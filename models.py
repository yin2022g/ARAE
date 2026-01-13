from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from layers import GNNLayer
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.1):
        super(GCN, self).__init__()
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


class GCN_d(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_input):
        super(GCN_d, self).__init__()
        self.gnn_3 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_4 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_5 = GNNLayer(gae_n_dec_3, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_3(z_igae, adj, active=True)
        z = self.gnn_4(z, adj, active=True)
        z_hat = self.gnn_5(z, adj, active=False)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class SEWGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, threshold=0.1, dropout=0.1):
        super(SEWGCN, self).__init__()
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


class ARAE(nn.Module):
    def __init__(
        self,
        ninput,
        sew_n_hid_1,
        n_hid_1,
        n_hid_2,
        n_z_l,
        nclass,
        n_decon,
        dropout,
        alpha,
        threshold,
        v=1,
    ):
        super(ARAE, self).__init__()
        self.dropout = dropout
        self.SEWGCN_encoder = SEWGCN(
            nfeat=ninput, nhid=sew_n_hid_1, nclass=n_z_l, threshold=threshold
        )
        self.SEWGCN_decoder = GCN(n_z_l, sew_n_hid_1, ninput)

        self.GCN_encoder = GCN(ninput, n_hid_1, n_z_l)
        self.GCN_decoder = GCN_d(n_z_l, n_hid_2, n_hid_1, ninput)
        self.cluster_layer = Parameter(
            torch.Tensor(nclass, 2 * n_z_l + 8), requires_grad=True
        )
        self.v = v
        self.weight1 = Parameter(torch.FloatTensor(n_decon, 8))
        self.weight2 = Parameter(torch.FloatTensor(8, n_decon))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.cluster_layer)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        self.SEWGCN_encoder.initialize()
        self.GCN_encoder.initialize()
        self.SEWGCN_decoder.initialize()

    def forward(self, x, adj_l, dec, adj_new, save=False):
        emb_gcn = self.SEWGCN_encoder(x, adj_new)
        z_hat_emb = self.SEWGCN_decoder(emb_gcn, adj_new)
        h = self.GCN_encoder(x, adj_new)
        z_hat, z_hat_adj = self.GCN_decoder(h, adj_l)

        emb_d = F.dropout(dec, self.dropout, self.training)
        emb_d = torch.mm(emb_d, self.weight1)
        emb_d = torch.mm(adj_l, emb_d)
        dec_hat = F.dropout(emb_d, self.dropout, self.training)
        dec_hat = torch.mm(dec_hat, self.weight2)
        dec_hat = torch.mm(adj_l, dec_hat)

        z = torch.cat((h, emb_gcn, emb_d), 1)
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
        return z_hat, q, z, z_hat_emb, dec_hat
