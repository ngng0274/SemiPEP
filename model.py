import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, BatchNorm1d, Dropout, Sigmoid, Conv1d, LSTM, LayerNorm, ReLU, MultiheadAttention
from torch_geometric.nn import GINConv, GCNConv, GATConv, Linear

# Instructor model with learnable alpha
class SemiPEP_InstructorMLP(nn.Module):
    def __init__(self, agg_input_dim: int, hidden_dim: int, init_alpha: float = 0.5):
        super(SemiPEP_InstructorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(agg_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        raw = torch.logit(torch.tensor(init_alpha))
        self.alpha_raw = nn.Parameter(raw)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

# GIN    
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()

        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # GAT
        self.conv1 = GATConv(self.input_dim, hidden_dim, heads=10, dropout=dropout)
        self.conv2 = GATConv(hidden_dim*10, hidden_dim, dropout=dropout)

        self.fc = Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc(x))

        return x  # (node_num, hidden_dim)
    
class CrossAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(CrossAttention, self).__init__()
        #MultiHead
        self.MultiHead_1 = MultiheadAttention(embed_dim=input_size, num_heads=num_heads)

    def forward(self, input1, input2):
        input1 = input1.unsqueeze(0).transpose(0, 1)
        input2 = input2.unsqueeze(0).transpose(0, 1)
        output_1, attention_weights_1 = self.MultiHead_1(input1, input2, input2)
        attention_weights_1=attention_weights_1.squeeze(0)
        output_1 = output_1.transpose(0, 1).squeeze(0)

        return output_1, attention_weights_1

class SemiPEP_Target(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super(SemiPEP_Target, self).__init__()

        self.hidden_dim = 64
        self.esm_dim = 480
        self.ablang_dim = 512
        self.multiheads = 16
        self.dropout = dropout
        
        self.ag_graphenc = GraphEncoder(480, self.hidden_dim, self.dropout)
        self.ab_graphenc = GraphEncoder(512, self.hidden_dim, self.dropout)

        self.linear_ag = Linear(2 * self.hidden_dim, self.hidden_dim)
        self.linear_ab = Linear(2 * self.hidden_dim, self.hidden_dim)

        # CrossAttention
        self.ag_crossattention = CrossAttention(self.hidden_dim, self.multiheads)
        self.ab_crossattention = CrossAttention(self.hidden_dim, self.multiheads)
        self.ag_feed_forward = Sequential(
            Linear(self.hidden_dim, 4 * self.hidden_dim),
            ReLU(),
            Linear(4 * self.hidden_dim, self.hidden_dim)
        )
        self.ag_norm1 = LayerNorm(self.hidden_dim)
        self.ag_norm2 = LayerNorm(self.hidden_dim)
        self.ab_feed_forward = Sequential(
            Linear(self.hidden_dim, 4 * self.hidden_dim),
            ReLU(),
            Linear(4 * self.hidden_dim, self.hidden_dim)
        )
        self.ab_norm1 = LayerNorm(self.hidden_dim)
        self.ab_norm2 = LayerNorm(self.hidden_dim)

        self.ag_linearsigmoid_linear = Linear(self.hidden_dim, 1)
        self.ag_linearsigmoid_sigmoid = Sigmoid()
        self.ab_linearsigmoid_linear = Linear(self.hidden_dim, 1)
        self.ab_linearsigmoid_sigmoid = Sigmoid()


    def forward(self, *agab):
        ag_x = agab[0]
        ag_edge_index = agab[1]
        ab_x = agab[2]
        ab_edge_index = agab[3]
        ag_pre_cal = agab[4]
        ab_pre_cal = agab[5]

        # Ag
        ag_emb = self.ag_graphenc(ag_pre_cal, ag_edge_index)
        
        # Ab
        ab_emb = self.ab_graphenc(ab_pre_cal, ab_edge_index)
        
        # CrossAttention
        ag_attention, ag_attention_weights = self.ag_crossattention(ag_emb, ab_emb)
        ab_attention, ab_attention_weights = self.ab_crossattention(ab_emb, ag_emb)
        ag_res1 = self.ag_norm1(ag_emb + ag_attention)
        ab_res1 = self.ab_norm1(ab_emb + ab_attention)
        ag_res2 = self.ag_norm2(ag_res1 + self.ag_feed_forward(ag_res1))
        ab_res2 = self.ab_norm2(ab_res1 + self.ab_feed_forward(ab_res1))

        ag_out_0 = self.ag_linearsigmoid_linear(ag_res2)
        ag_out = self.ag_linearsigmoid_sigmoid(ag_out_0)
        ab_out_0 = self.ab_linearsigmoid_linear(ab_res2)
        ab_out = self.ab_linearsigmoid_sigmoid(ab_out_0)
        
        return ag_out, ab_out, ag_attention_weights, ab_attention_weights
    
