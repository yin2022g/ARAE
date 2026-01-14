from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from layers import GNNLayer
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor


class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj, active=True)
        z = self.gnn_5(z, adj, active=True)
        z_hat = self.gnn_6(z, adj, active=False)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class GCN_a(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.1):
        super(GCN_a, self).__init__()
        self.dropout = dropout
        self.gc1 = GCNConv(
            nfeat, nhid, bias=True, add_self_loops=False, normalize=False
        )
        self.gc2 = GCNConv(
            nhid, nclass, bias=True, add_self_loops=False, normalize=False
        )

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GCN_emb(nn.Module):
    def __init__(self, nfeat, nhid, nclass, threshold=0.1, dropout=0.1):
        super(GCN_emb, self).__init__()
        self.dropout = dropout
        self.threshold = threshold
        nclass = int(nclass)
        self.gc1 = GCNConv(nfeat, nhid, bias=True, add_self_loops=True, normalize=True)
        self.gc2 = GCNConv(nhid, nclass, bias=True, add_self_loops=True, normalize=True)

    def forward(self, x, adj, save=False):
        adj = self.att_coef(x, adj)
        if save:
            dense_adj = adj.to_dense()
            df = pd.DataFrame(dense_adj.cpu().numpy())
            df.to_csv("adj_1.csv", index=False, header=False)

        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        adj = self.att_coef(x, adj)
        x = self.gc2(x, adj)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def att_coef(self, features, adj, t=0):
        with torch.no_grad():
            sim_features = features
            row, col = adj.coo()[:2]
            n_total = sim_features.size(0)
            if sim_features.size(1) > 512 or row.size(0) > 5e5:
                batch_size = int(1e8 // sim_features.size(1))
                bepoch = row.size(0) // batch_size + (row.size(0) % batch_size > 0)
                sims = []
                for i in range(bepoch):
                    st = i * batch_size
                    ed = min((i + 1) * batch_size, row.size(0))
                    sims.append(
                        F.cosine_similarity(
                            sim_features[row[st:ed]], sim_features[col[st:ed]]
                        )
                    )
                sims = torch.cat(sims, dim=0)
            else:
                sims = F.cosine_similarity(sim_features[row], sim_features[col])

            mask = torch.logical_and(sims >= self.threshold, row != col)
            row = row[mask]
            col = col[mask]
            sims = sims[mask]
            graph_size = torch.Size((n_total, n_total))
            new_adj = SparseTensor(
                row=row, col=col, value=sims, sparse_sizes=graph_size
            )
        return new_adj.cuda()


class GAT_DAGA_gcn_gcn_emb_other(nn.Module):
    def __init__(
        self, ninput, n_hid_1, n_hid_2, n_z_l, nclass, dropout, threshold, v=1
    ):
        super(GAT_DAGA_gcn_gcn_emb_other, self).__init__()
        self.dropout = dropout
        self.GCN_emb = GCN_emb(ninput, n_hid_1, n_z_l, threshold=threshold)
        self.gcn = GCN_a(ninput, n_hid_1, n_z_l)
        self.decoder = IGAE_decoder(n_z_l, n_hid_2, n_hid_1, ninput)
        self.gcn_decoder = GCN_a(n_z_l, n_hid_1, ninput)
        self.cluster_layer = Parameter(
            torch.Tensor(nclass, 2 * n_z_l), requires_grad=True
        )
        self.v = v
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.cluster_layer)
        self.GCN_emb.initialize()
        self.gcn.initialize()
        self.gcn_decoder.initialize()

    def forward(self, x, adj_new, adj_l):
        emb_gcn = self.GCN_emb(x, adj_new)
        h = self.gcn(x, adj_new)
        z_hat, z_hat_adj = self.decoder(h, adj_l)
        z_hat_emb = self.gcn_decoder(emb_gcn, adj_new)
        z = torch.cat((h, emb_gcn), 1)
        q = 1.0 / (
            (
                1.0
                + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), dim=2)
                / self.v
            )
            + 1e-8
        )
        q = q ** (self.v + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z_hat, q, z, z_hat_emb
