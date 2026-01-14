from __future__ import print_function, division
import os
import numpy as np
import scanpy as sc
import os
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from models import GAT_DAGA_gcn_gcn_emb_other
from config import Config
import torch.optim as optim
from utils import target_distribution, preprocess_adj, mclust_R, set_seed
import scipy.sparse as sp
from torch_sparse import SparseTensor
from preprocess import process_mush_muso, preprocess_adj, get_adata_slide_stereo_data
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"


def train_DAGA_v3(
    update_interval,
    tol,
    filepath,
    dataset,
    n_hid_1,
    n_hid_2,
    n_z_l,
    nclass,
    dropout,
    lr,
    weight_decay,
    seed,
    epochs,
    ls_1,
    ls_2,
    ls_3,
    threshold,
):
    set_seed(seed)
    adata = get_adata_slide_stereo_data(filepath, dataset)
    adata = process_mush_muso(adata)
    print(adata)
    features = torch.FloatTensor(adata.obsm["feat"].copy()).cuda()
    adj = adata.obsm["adj"]
    adj = preprocess_adj(adj)

    adj_new = np.array(adj)
    adj_new = sp.coo_matrix(adj_new)
    col = torch.from_numpy(adj_new.col).type(torch.long)
    row = torch.from_numpy(adj_new.row).type(torch.long)
    data = torch.from_numpy(adj_new.data).type(torch.float)
    adj_l = torch.sparse.FloatTensor(
        torch.stack([row, col], dim=0), data, torch.Size(adj_new.shape)
    ).cuda()
    adj_new = SparseTensor(row=row, col=col, value=data, sparse_sizes=adj_new.shape)
    adj_new = adj_new.cuda()

    model = GAT_DAGA_gcn_gcn_emb_other(
        features.shape[1], n_hid_1, n_hid_2, n_z_l, nclass, dropout, threshold, v=1
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.cuda()

    with torch.no_grad():
        _, _, z_tmp, _ = model(features, adj_new, adj_l)
    kmeans = KMeans(n_clusters=nclass, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z_tmp.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

    model.train()
    optimizer.zero_grad()
    for epoch in range(epochs):
        if epoch % update_interval == 0:
            _, q, _, _ = model(features, adj_new, adj_l)
            p = target_distribution(q.data)

        z_hat, q, _, z_hat_emb = model(features, adj_new, adj_l)
        loss_gcn = F.mse_loss(z_hat, features, reduction="mean")
        loss_gcn_emb = F.mse_loss(z_hat_emb, features, reduction="mean")
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        loss = ls_1 * loss_gcn + ls_2 * loss_gcn_emb + ls_3 * kl_loss
        print("loss: ", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, q_f, emb_d, _ = model(features, adj_new, adj_l)

    adata.obsm["daga"] = emb_d.to("cpu").detach().numpy()
    adata.write_h5ad("/path/to/output/" + dataset + "/results_pre.h5ad")
    adata = mclust_R(adata, used_obsm="daga", num_cluster=nclass)

    adata.write_h5ad("/path/to/output/" + dataset + "/results.h5ad")

    plotname = "/path/to/output/" + dataset + "_pred.png"
    plt.rcParams["figure.figsize"] = (8, 5)

    ax = sc.pl.embedding(
        adata,
        basis="spatial",
        color="mclust",
        s=30,
        show=False,
        title="model",
    )
    ax.axis("off")
    plt.savefig(plotname)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())  # 0
    print(torch.cuda.get_device_name(0))
    os.environ["R_HOME"] = "/path/to/lib/R"
    config_file = "./config/Stereo-seq_MoB.ini"
    config = Config(config_file)
    adata = sc.read_h5ad("./data/Stereo-seq_MoB/results_pre.h5ad")
    adata = mclust_R(adata, used_obsm="daga", num_cluster=7)
    train_DAGA_v3(
        update_interval=3,
        tol=1e-3,
        filepath=config["filepath"],
        dataset=config["dataset"],
        n_hid_1=config["n_hid_1"],
        n_hid_2=config["n_hid_2"],
        n_z_l=config["n_z_l"],
        nclass=config["nclass"],
        dropout=config["dropout"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        seed=config["seed"],
        epochs=config["epochs"],
        ls_1=config["ls_1"],
        ls_2=config["ls_2"],
        ls_3=config["ls_3"],
        threshold=config["threshold"],
    )
