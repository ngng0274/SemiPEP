import sys, os
# import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import argparse

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.data import Data as PygData
from torch_geometric.utils import k_hop_subgraph
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from asep.data.asepv1_dataset import PairData, AsEPv1Dataset, EmbeddingConfig
from model import SemiPEP_Target, SemiPEP_InstructorMLP
from utils import *

# Confidence computations
def compute_joint_confidence(conf_ag, conf_ab):
    return torch.sqrt(conf_ag * conf_ab)

def compute_node_confidence_L1(node_emb, node_pred, node_ref, instructor_model, device):
    l1 = nn.L1Loss(reduction='none')
    N, H = node_emb.size()
    node_conf = torch.zeros((N,1), device=device)
    for i in range(N):
        h = node_emb[i].view(1,H)
        p = node_pred[i].view(1,1)
        r = node_ref[i].view(1,1)
        hf = l1(p.squeeze(), r.squeeze()).view(1,1)
        inp = torch.cat([h, p, hf], dim=1)
        node_conf[i] = instructor_model(inp)
    return node_conf

def compute_local_confidence_khop(
    node_conf, node_pred, edge_index, num_hops, K
):
    N = node_pred.size(0)
    device = node_conf.device

    # Clamp K so it never exceeds number of nodes
    K = min(K, N)
    # Get top-K node indices
    _, topk_idx = torch.topk(node_pred.squeeze(), k=K, largest=True)

    sub_nodes_set = set()
    for idx in topk_idx:
        # Make sure root is a 1-D tensor of shape [1]
        root = torch.tensor([int(idx)], dtype=torch.long, device=device)
        neigh, _, _, _ = k_hop_subgraph(root, num_hops, edge_index, relabel_nodes=False)
        sub_nodes_set.update(neigh.tolist())

    # Fallback to global average if no nodes found
    if not sub_nodes_set:
        return node_conf.mean().view(1, 1)

    # Gather subgraph nodes and compute weighted mean
    sub = torch.tensor(sorted(sub_nodes_set), dtype=torch.long, device=device)
    conf_sub = node_conf[sub]           # [M,1]
    pred_sub = node_pred[sub].detach()  # [M,1]

    weight_sum = pred_sub.sum().clamp(min=1e-6)
    local_conf = (conf_sub * pred_sub).sum() / weight_sum

    return local_conf.view(1, 1)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=10, mode='min', verbose=False):
        self.patience, self.mode, self.verbose = patience, mode, verbose
        self.best, self.counter, self.early_stop = None, 0, False
    def __call__(self, metric):
        if self.best is None:
            self.best = metric
        else:
            improved = (metric < self.best) if self.mode=='min' else (metric > self.best)
            if improved:
                self.best = metric; self.counter=0
            else:
                self.counter+=1
                if self.counter>=self.patience:
                    self.early_stop=True

# Data helpers

def prepare_agab(data: PairData, device):
    return (
        data.x_g_one_hot.to(device), data.edge_index_g.to(device),
        data.x_b_one_hot.to(device), data.edge_index_b.to(device),
        data.x_g_pre_cal.to(device), data.x_b_pre_cal.to(device)
    )

def build_edge_label(data: PairData, device):
    ag_n, ab_n = data.y_g.size(0), data.y_b.size(0)
    mat = torch.zeros((ag_n,ab_n), device=device)
    for src,dst in zip(data.edge_index_bg[0], data.edge_index_bg[1]): mat[dst,src]=1.0
    return mat

def graph_level_repr(node_emb):
    return node_emb.mean(dim=0, keepdim=True)

# Pretrain with gradient accumulation

def pretrain_models(target_model: SemiPEP_Target,
                    instructor_model: SemiPEP_InstructorMLP,
                    labeled_ds: List[PairData],
                    unlabeled_ds: List[PairData],
                    opt_f, opt_g,
                    device,
                    pretrain_epochs_f=20,
                    pretrain_epochs_g=5,
                    batch_size=32,
                    k=2):
    loss_bce, loss_l1 = nn.BCELoss(), nn.L1Loss()
    target_model.to(device)
    # Pretrain target
    for ep in range(pretrain_epochs_f):
        target_model.train()
        L_acc, steps = 0.0, 0
        opt_f.zero_grad()
        for i,data in enumerate(labeled_ds):
            y_g = data.y_g.float().to(device)
            y_b = data.y_b.float().to(device)
            agab = prepare_agab(data,device)
            out_g,out_b,e_p,e_pt = target_model(*agab)
            edge_lbl = build_edge_label(data,device)
            loss = loss_bce(out_g.squeeze(),y_g) + loss_bce(out_b.squeeze(),y_b)
            loss += 10*(loss_bce(e_p,edge_lbl) + loss_bce(e_pt.t(),edge_lbl))
            L_acc += loss
            steps += 1
            if steps % batch_size == 0 or i == len(labeled_ds)-1:
                (L_acc/steps).backward()
                opt_f.step(); opt_f.zero_grad()
                L_acc, steps = 0.0, 0
        print(f"[Pretrain-F] Ep {ep+1}/{pretrain_epochs_f}")

    # Warmup instructor
    target_model.eval()
    with torch.no_grad():
        for data in unlabeled_ds:
            agab=prepare_agab(data,device)
            g,b,_,_ = target_model(*agab)
            data.y_g_pseudo = g.cpu(); data.y_b_pseudo = b.cpu()
    instructor_model.to(device)

    # Pretrain instructor with accumulation
    for ep in range(pretrain_epochs_g):
        instructor_model.train()
        L_acc, steps = 0.0, 0
        opt_g.zero_grad()
        for i,data in enumerate(labeled_ds+unlabeled_ds):
            agab=prepare_agab(data,device)
            with torch.no_grad():
                g_pred,b_pred,_,_ = target_model(*agab)
                h_ag = graph_level_repr(target_model.ag_graphenc(agab[4],agab[1]))
                h_ab = graph_level_repr(target_model.ab_graphenc(agab[5],agab[3]))
            y_g = (data.y_g if data.c.item()==1 else data.y_g_pseudo).float().to(device)
            y_b = (data.y_b if data.c.item()==1 else data.y_b_pseudo).float().to(device)
            m_g,m_b = y_g.mean().view(1,1), y_b.mean().view(1,1)
            hf_g = loss_l1(g_pred.squeeze(),y_g.squeeze()).view(1,1)
            hf_b = loss_l1(b_pred.squeeze(),y_b.squeeze()).view(1,1)
            gc_ag = instructor_model(torch.cat([h_ag,m_g,hf_g],dim=1))
            gc_ab = instructor_model(torch.cat([h_ab,m_b,hf_b],dim=1))
            nc_ag = compute_node_confidence_L1(target_model.ag_graphenc(agab[4],agab[1]), g_pred, y_g, instructor_model, device)
            nc_ab = compute_node_confidence_L1(target_model.ab_graphenc(agab[5],agab[3]), b_pred, y_b, instructor_model, device)
            lc_ag = compute_local_confidence_khop(nc_ag, g_pred, agab[1], k, 20)
            lc_ab = compute_local_confidence_khop(nc_ab, b_pred, agab[3], k, 20)
            conf_ag = instructor_model.alpha*gc_ag + (1-instructor_model.alpha)*lc_ag
            conf_ab = instructor_model.alpha*gc_ab + (1-instructor_model.alpha)*lc_ab
            c = data.c.float().to(device).view(1,1)
            loss = loss_bce(conf_ag,c) + loss_bce(conf_ab,c)
            L_acc += loss; steps += 1
            if steps % batch_size == 0 or i == len(labeled_ds+unlabeled_ds)-1:
                (L_acc/steps).backward()
                opt_g.step(); opt_g.zero_grad()
                L_acc, steps = 0.0, 0
        print(f"[Pretrain-G] Ep {ep+1}/{pretrain_epochs_g} ")

def train_semipep(
    target_model: SemiPEP_Target,
    instructor_model: SemiPEP_InstructorMLP,
    labeled_ds: List[PairData],
    unlabeled_ds: List[PairData],
    val_ds: List[PairData],
    opt_f: optim.Optimizer,
    opt_g: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    batch_size: int = 32,
    k: int = 2,
    p_start: float = 10.0,
    p_end: float = 90.0
) -> Tuple[SemiPEP_Target, SemiPEP_InstructorMLP]:
    """
    Dynamic threshold (percentile scheduling) training loop with gradient accumulation.
    """
    loss_bce = nn.BCELoss()
    sched_f = StepLR(opt_f, step_size=30, gamma=0.1)
    sched_g = StepLR(opt_g, step_size=30, gamma=0.1)
    stopper = EarlyStopping(patience=10, mode='min', verbose=True)

    for epoch in range(num_epochs):
        # 1. Pseudo-label 업데이트
        if epoch % 5 == 0:
            target_model.eval()
            with torch.no_grad():
                for data in unlabeled_ds:
                    agab = prepare_agab(data, device)
                    g_pred, b_pred, _, _ = target_model(*agab)
                    data.y_g_pseudo = g_pred.cpu()
                    data.y_b_pseudo = b_pred.cpu()

        # 2. Joint confidence 수집
        joint_vals = []
        target_model.eval(); instructor_model.eval()
        with torch.no_grad():
            for data in unlabeled_ds:
                agab = prepare_agab(data, device)
                g_pred, b_pred, _, _ = target_model(*agab)
                # graph representations
                h_ag = graph_level_repr(target_model.ag_graphenc(agab[4], agab[1]))
                h_ab = graph_level_repr(target_model.ab_graphenc(agab[5], agab[3]))
                # global metrics
                # # 실제 or pseudo 레이블 선택
                y_g = (data.y_g if data.c.item()==1 else data.y_g_pseudo).float().to(device)
                y_b = (data.y_b if data.c.item()==1 else data.y_b_pseudo).float().to(device)
                m_g = y_g.mean().view(1,1); m_b = y_b.mean().view(1,1)
                hf_g = nn.L1Loss()(g_pred.squeeze(), y_g.squeeze()).view(1,1)
                hf_b = nn.L1Loss()(b_pred.squeeze(), y_b.squeeze()).view(1,1)
                gc_ag = instructor_model(torch.cat([h_ag, m_g, hf_g], dim=1))
                gc_ab = instructor_model(torch.cat([h_ab, m_b, hf_b], dim=1))
                # local metrics
                nc_ag = compute_node_confidence_L1(
                    target_model.ag_graphenc(agab[4], agab[1]), g_pred, y_g, instructor_model, device)
                nc_ab = compute_node_confidence_L1(
                    target_model.ab_graphenc(agab[5], agab[3]), b_pred, y_b, instructor_model, device)
                lc_ag = compute_local_confidence_khop(nc_ag, g_pred, agab[1], k, 20)
                lc_ab = compute_local_confidence_khop(nc_ab, b_pred, agab[3], k, 20)
                # joint
                conf_ag = instructor_model.alpha * gc_ag + (1 - instructor_model.alpha) * lc_ag
                conf_ab = instructor_model.alpha * gc_ab + (1 - instructor_model.alpha) * lc_ab
                joint = compute_joint_confidence(conf_ag, conf_ab)
                joint_vals.append(joint.item())

        # 3. Dynamic gamma 설정
        p = p_start + (p_end - p_start) * epoch / (num_epochs - 1)
        gamma = float(np.percentile(joint_vals, p))

        # 4. D_double 구성
        D_double: List[PairData] = []
        for data in labeled_ds + unlabeled_ds:
            if data.c.item() == 1:
                D_double.append(data)
            else:
                agab = prepare_agab(data, device)
                g_pred, b_pred, _, _ = target_model(*agab)
                h_ag = graph_level_repr(target_model.ag_graphenc(agab[4], agab[1]))
                h_ab = graph_level_repr(target_model.ab_graphenc(agab[5], agab[3]))
                y_g = (data.y_g if data.c.item()==1 else data.y_g_pseudo).float().to(device)
                y_b = (data.y_b if data.c.item()==1 else data.y_b_pseudo).float().to(device)
                m_g = y_g.mean().view(1,1); m_b = y_b.mean().view(1,1)
                hf_g = nn.L1Loss()(g_pred.squeeze(), y_g.squeeze()).view(1,1)
                hf_b = nn.L1Loss()(b_pred.squeeze(), y_b.squeeze()).view(1,1)
                gc_ag = instructor_model(torch.cat([h_ag, m_g, hf_g], dim=1))
                gc_ab = instructor_model(torch.cat([h_ab, m_b, hf_b], dim=1))
                nc_ag = compute_node_confidence_L1(
                    target_model.ag_graphenc(agab[4], agab[1]), g_pred, y_g, instructor_model, device)
                nc_ab = compute_node_confidence_L1(
                    target_model.ab_graphenc(agab[5], agab[3]), b_pred, y_b, instructor_model, device)
                lc_ag = compute_local_confidence_khop(nc_ag, g_pred, agab[1], k, 20)
                lc_ab = compute_local_confidence_khop(nc_ab, b_pred, agab[3], k, 20)
                joint = compute_joint_confidence(
                    instructor_model.alpha * gc_ag + (1 - instructor_model.alpha) * lc_ag,
                    instructor_model.alpha * gc_ab + (1 - instructor_model.alpha) * lc_ab
                )
                if joint.item() >= gamma:
                    D_double.append(data)

        # 5. Instructor 업데이트 (gradient accumulation)
        instructor_model.train()
        opt_g.zero_grad()
        acc_loss, steps = 0.0, 0
        for i, data in enumerate(labeled_ds + unlabeled_ds):
            agab = prepare_agab(data, device)
            g_pred, b_pred, _, _ = target_model(*agab)
            h_ag = graph_level_repr(target_model.ag_graphenc(agab[4], agab[1]))
            h_ab = graph_level_repr(target_model.ab_graphenc(agab[5], agab[3]))
            y_g = (data.y_g if data.c.item()==1 else data.y_g_pseudo).float().to(device)
            y_b = (data.y_b if data.c.item()==1 else data.y_b_pseudo).float().to(device)
            m_g = y_g.mean().view(1,1); m_b = y_b.mean().view(1,1)
            hf_g = nn.L1Loss()(g_pred.squeeze(), y_g.squeeze()).view(1,1)
            hf_b = nn.L1Loss()(b_pred.squeeze(), y_b.squeeze()).view(1,1)
            gc_ag = instructor_model(torch.cat([h_ag, m_g, hf_g], dim=1))
            gc_ab = instructor_model(torch.cat([h_ab, m_b, hf_b], dim=1))
            nc_ag = compute_node_confidence_L1(
                target_model.ag_graphenc(agab[4], agab[1]), g_pred, y_g, instructor_model, device)
            nc_ab = compute_node_confidence_L1(
                target_model.ab_graphenc(agab[5], agab[3]), b_pred, y_b, instructor_model, device)
            lc_ag = compute_local_confidence_khop(nc_ag, g_pred, agab[1], k, 20)
            lc_ab = compute_local_confidence_khop(nc_ab, b_pred, agab[3], k, 20)
            conf_ag = instructor_model.alpha * gc_ag + (1 - instructor_model.alpha) * lc_ag
            conf_ab = instructor_model.alpha * gc_ab + (1 - instructor_model.alpha) * lc_ab
            c = data.c.float().to(device).view(1,1)
            loss = loss_bce(conf_ag, c) + loss_bce(conf_ab, c)
            acc_loss += loss; steps += 1
            if steps % batch_size == 0 or i == len(labeled_ds + unlabeled_ds) - 1:
                (acc_loss/steps).backward()
                opt_g.step(); opt_g.zero_grad()
                acc_loss, steps = 0.0, 0

        # 6. Target 업데이트 (gradient accumulation)
        target_model.train()
        opt_f.zero_grad()
        acc_loss, steps = 0.0, 0
        for i, data in enumerate(D_double):
            agab = prepare_agab(data, device)
            out_g, out_b, e_p, e_pt = target_model(*agab)
            edge_lbl = build_edge_label(data, device)
            loss = loss_bce(out_g.squeeze(), data.y_g.float().to(device))
            loss += loss_bce(out_b.squeeze(), data.y_b.float().to(device))
            loss += 10 * (loss_bce(e_p, edge_lbl) + loss_bce(e_pt.t(), edge_lbl))
            acc_loss += loss; steps += 1
            if steps % batch_size == 0 or i == len(D_double) - 1:
                (acc_loss/steps).backward()
                opt_f.step(); opt_f.zero_grad()
                acc_loss, steps = 0.0, 0

        # 7. 검증, 스케줄링, EarlyStopping
        sched_f.step(); sched_g.step()
        val_loss, met_ag, met_ab = evaluate(target_model, val_ds, device)
        stopper(val_loss)
        if stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        print(f"Epoch {epoch+1}/{num_epochs} |  | gamma={gamma:.4f} | val_loss={val_loss:.4f}")
        print(f'AUPRC_ag {met_ag[0]:.8f} AUROC_ag {met_ag[1]:.8f} MCC_ag {met_ag[4]:.8f} AUPRC_ab {met_ab[0]:.8f} AUROC_ab {met_ab[1]:.8f} MCC_ab {met_ab[4]:.8f}')
        

    return target_model, instructor_model

def evaluate(model, dataset, device):
    loss_BCE = torch.nn.BCELoss()
    val_loss = 0.0

    with torch.no_grad():
        for i in range(len(dataset)):
            ag_node_attr = dataset[i]['x_g_one_hot'].to(device)
            ag_edge_index = dataset[i]['edge_index_g'].to(device)
            ag_pre_cal = dataset[i]['x_g_pre_cal'].to(device)
            ag_label = dataset[i]['y_g'].float().to(device)

            ab_node_attr = dataset[i]['x_b_one_hot'].to(device)
            ab_edge_index = dataset[i]['edge_index_b'].to(device)
            ab_pre_cal = dataset[i]['x_b_pre_cal'].to(device)
            ab_label = dataset[i]['y_b'].float().to(device)

            edge_label = np.zeros((ag_label.shape[0], ab_label.shape[0]))

            inter_edge_index = dataset[i]['edge_index_bg']
            for edge_ind in range(len(inter_edge_index[0])):
                edge_label[inter_edge_index[1][edge_ind], inter_edge_index[0][edge_ind]] = 1.0
            edge_label = torch.tensor(edge_label, dtype=torch.float32).to(device)

            agab = [ag_node_attr, ag_edge_index, ab_node_attr, ab_edge_index, ag_pre_cal, ab_pre_cal]

            outputs = model(*agab)

            val_loss += loss_BCE((outputs[0]).squeeze(dim=1), ag_label) + loss_BCE((outputs[1]).squeeze(dim=1), ab_label) + 10 * (loss_BCE(outputs[2], edge_label)) + 10 * (loss_BCE(outputs[3].t(), edge_label))

            # evalution
            output_ag_test = torch.flatten(outputs[0]) if i == 0 else torch.cat((output_ag_test, torch.flatten(outputs[0])), dim=0)
            target_ag_test = ag_label.long() if i == 0 else torch.cat((target_ag_test, ag_label.long()), dim=0)
            output_ab_test = torch.flatten(outputs[1]) if i == 0 else torch.cat((output_ab_test, torch.flatten(outputs[1])), dim=0)
            target_ab_test = ab_label.long() if i == 0 else torch.cat((target_ab_test, ab_label.long()), dim=0)
        
        metrics_ag = evalution_prot(output_ag_test, target_ag_test, device)
        metrics_ab = evalution_prot(output_ab_test, target_ab_test, device)

        return val_loss / len(dataset), metrics_ag, metrics_ab

def test(model, device):
    with torch.no_grad():
        for i in range(len(test_dataset)):
            ag_node_attr = test_dataset[i]['x_g_one_hot'].to(device)
            ag_edge_index = test_dataset[i]['edge_index_g'].to(device)

            ag_pre_cal = test_dataset[i]['x_g_pre_cal'].to(device)
            ag_label = test_dataset[i]['y_g'].float().to(device)

            ab_node_attr = test_dataset[i]['x_b_one_hot'].to(device)
            ab_edge_index = test_dataset[i]['edge_index_b'].to(device)
            ab_pre_cal = test_dataset[i]['x_b_pre_cal'].to(device)
            ab_label = test_dataset[i]['y_b'].float().to(device)

            agab = [ag_node_attr, ag_edge_index, ab_node_attr, ab_edge_index, ag_pre_cal, ab_pre_cal]

            outputs = model(*agab)
            # evalution
            output_ag_test = torch.flatten(outputs[0]) if i == 0 else torch.cat((output_ag_test, torch.flatten(outputs[0])), dim=0)
            target_ag_test = ag_label.long() if i == 0 else torch.cat((target_ag_test, ag_label.long()), dim=0)
            output_ab_test = torch.flatten(outputs[1]) if i == 0 else torch.cat((output_ab_test, torch.flatten(outputs[1])), dim=0)
            target_ab_test = ab_label.long() if i == 0 else torch.cat((target_ab_test, ab_label.long()), dim=0)
        
        test_auprc_ag, test_auroc_ag, test_precision_ag, test_recall_ag, test_mcc_ag, test_f1_ag = evalution_prot(output_ag_test, target_ag_test, device)
        test_auprc_ab, test_auroc_ab, test_precision_ab, test_recall_ab, test_mcc_ab, test_f1_ab = evalution_prot(output_ab_test, target_ab_test, device)

        print("Antigen | Test | " + " AUPRC: " + str(test_auprc_ag) + " AUROC: " + str(test_auroc_ag) + " Precision: " + str(test_precision_ag) + " Recall: " + str(test_recall_ag) + " MCC: " + str(test_mcc_ag) + " F1: " + str(test_f1_ag))
        print("Antibody | Test | " + " AUPRC: " + str(test_auprc_ab) + " AUROC: " + str(test_auroc_ab) + " Precision: " + str(test_precision_ab) + " Recall: " + str(test_recall_ab) + " MCC: " + str(test_mcc_ab) + " F1: " + str(test_f1_ab))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0001) ## learning rate
    parser.add_argument('--batch_size', type=int, default=32) ## batch_size
    parser.add_argument('--num_epochs', type=int, default=100) ## num_epochs
    parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3') ## 사용할 gpu

    opt = parser.parse_args()

    device = torch.device('cuda:' + str(opt.gpu))

    print(opt)

    learning_rate = opt.lr
    epoch = opt.num_epochs
    batch_size = opt.batch_size

    # pre-calculated embeddings with AntiBERTy (via igfold) and ESM2
    config = EmbeddingConfig(
        node_feat_type=["pre_cal", "one_hot"],
        ab={"embeddininstructor_model": "igfold"},  # change this "esm2" for ESM2 embeddings
        ag={"embeddininstructor_model": "esm2"},
    )
    asepv1_dataset = AsEPv1Dataset(
        root="./data",   # replace with the path to the parent folder of downloaded AsEP
        name="AsEP",
        embedding_config=config,
    )

    # split_method either "epitope_ratio" or "epitope_group"
    split_idx = asepv1_dataset.get_idx_split(split_method="epitope_ratio")
    train_dataset = asepv1_dataset[split_idx['train']] # 1383
    val_dataset = asepv1_dataset[split_idx['val']] # 170
    test_dataset  = asepv1_dataset[split_idx['test']] # 170

    split_index = len(train_dataset) // 4
    labeled_dataset = train_dataset[:split_index]  # 1에 해당하는 부분 (약 25%)
    unlabeled_dataset = train_dataset[split_index:]   # 3에 해당하는 부분 (약 75%)

    # subset_3의 각 PairData 객체에서 'c' 값을 0으로 초기화
    for data in unlabeled_dataset:
        data.c = torch.zeros(1, dtype=torch.long)

    target_model = SemiPEP_Target()
    instructor_model = SemiPEP_InstructorMLP(64 + 2, 64)
    opt_f    = optim.Adam(target_model.parameters(), lr=opt.lr)
    opt_g    = optim.Adam(instructor_model.parameters(), lr=opt.lr)

    # 1) 사전 학습
    pretrain_models(target_model, instructor_model,
                    labeled_dataset, unlabeled_dataset,
                    opt_f, opt_g, device,
                    pretrain_epochs_f=20,
                    pretrain_epochs_g=5,
                    batch_size=opt.batch_size,
                    k=2)

    # 2) SemiPEP 학습
    target_model, instructor_model = train_semipep(
        target_model, instructor_model,
        labeled_dataset, unlabeled_dataset, val_dataset,
        opt_f, opt_g, device,
        num_epochs=opt.num_epochs,
        batch_size=opt.batch_size,
        k=2,
        p_start=10.0,
        p_end=100.0
    )
    test(target_model, device)

