
import torch
import torch.nn.functional as F
import torch_geometric

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_convs=2, p_drop=0.5):
        super().__init__()
        assert num_convs > 0

        self.convs = torch.nn.ModuleList([
            torch_geometric.nn.GCNConv(num_node_features, hidden_dim),
        ])
        self.convs.extend([
            torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_convs - 1)
        ])
        self.gate = torch.nn.Linear(hidden_dim, 1)
        self.readout = torch_geometric.nn.GlobalAttention(self.gate)
        self.final = torch.nn.Linear(hidden_dim, 1)
        self.p_drop = p_drop

    def forward(self, graph):
        H = graph.x.clone()
        for conv in self.convs:
            H = conv(H, graph.edge_index)
            H = F.relu(H)
            H = F.dropout(H, self.p_drop, training=self.training)
        R = self.readout(H)
        out = self.final(R)
        return out
