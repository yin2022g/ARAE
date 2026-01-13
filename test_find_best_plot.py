from __future__ import print_function, division
import pandas as pd
import numpy as np
import scanpy as sc
import os
from sklearn import metrics
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from models import ARAE
from config import Config
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import target_distribution, eva, load_data, Clustering, set_seed
import scipy.sparse as sp
from torch_sparse import SparseTensor


def eva_test(y_true, y_pred, epoch=0):
    nmi = metrics.normalized_mutual_info_score(
        y_true, y_pred, average_method="arithmetic"
    )
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    print(epoch, ", nmi {:.8f}".format(nmi), ", ari {:.8f}".format(ari))
    return ari, nmi


def tmp_plot(section_id, ARI, key):
    adata = sc.read("/path/to/output/" + section_id + "_results.h5ad")
    plt.rcParams["figure.figsize"] = (8, 8)
    plotname = "/path/to/output/" + section_id + "_" + key + "_domain_pred.png"
    sc.pl.spatial(adata, color=[key], title="model (ARI=%.2f)" % ARI, show=False)
    plt.savefig(plotname)
    sc.pp.neighbors(adata, use_rep="daga")
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (8, 8)
    plotname = "/path/to/output/" + section_id + "_" + key + "_umap.png"
    sc.pl.umap(adata, color=[key], title="model (ARI=%.2f)" % ARI, show=False)
    plt.savefig(plotname)
    sc.tl.paga(adata, groups="ground_truth")
    plt.rcParams["figure.figsize"] = (6, 5)
    sc.pl.paga_compare(
        adata,
        legend_fontsize=10,
        frameon=False,
        size=20,
        title=section_id + "_model",
        legend_fontoutline=2,
        show=False,
    )
    plotname = "/path/to/output/" + section_id + "_" + key + "_trajectory_compare.png"
    plt.savefig(plotname)


def train_DAGA_v3(
    update_interval,
    filepath,
    dataset,
    sew_n_hid_1,
    n_hid_1,
    n_hid_2,
    n_z_l,
    nclass,
    n_decon,
    dropout,
    alpha,
    lr,
    weight_decay,
    seed,
    epochs,
    ls_1,
    ls_2,
    ls_3,
    ls_4,
    n_neighbors_a,
    threshold,
):
    set_seed(seed)
    adata, features, _, adj_l, label = load_data(filepath, dataset, n_neighbors_a)
    res_dec = adata.obsm["res_dec"]
    print("Size of res_dec: ", res_dec.shape)
    adj_new = np.array(adj_l)
    adj_new = sp.coo_matrix(adj_new)
    col = torch.from_numpy(adj_new.col).type(torch.long)
    row = torch.from_numpy(adj_new.row).type(torch.long)
    data = torch.from_numpy(adj_new.data).type(torch.float)

    adj_new = SparseTensor(row=row, col=col, value=data, sparse_sizes=adj_new.shape)
    adj_new = adj_new.cuda()

    features = np.array(features)
    features = torch.FloatTensor(features)
    adj_l = np.array(adj_l)
    adj_l = torch.FloatTensor(adj_l)
    label = np.array(label)
    label = torch.LongTensor(label)
    res_dec = np.array(res_dec)
    res_dec = torch.FloatTensor(res_dec)

    adj_l = adj_l.cuda()
    features = features.cuda()
    label = label.cuda()
    res_dec = res_dec.cuda()

    model = ARAE(
        features.shape[1],
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
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.cuda()

    with torch.no_grad():
        _, _, z_tmp, _, _ = model(features, adj_l, res_dec, adj_new, save=False)
    kmeans = KMeans(n_clusters=nclass, n_init=20, random_state=42)
    y_pred = kmeans.fit_predict(z_tmp.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

    eva(label, y_pred, "init by z")

    model.train()
    optimizer.zero_grad()
    y_pred_last = y_pred
    for epoch in range(epochs):
        if epoch % update_interval == 0:
            _, q, _, _, _ = model(features, adj_l, res_dec, adj_new)
            p = target_distribution(q.data)
            res2 = q.data.cpu().numpy().argmax(1)
            _, _ = eva(label, res2, str(epoch) + "Z-q")

        z_hat, q, z, z_hat_emb, dec_hat = model(features, adj_l, res_dec, adj_new)

        loss_gcn = F.mse_loss(z_hat, features, reduction="mean")
        loss_gcn_emb = F.mse_loss(z_hat_emb, features, reduction="mean")
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        loss_dec = F.mse_loss(dec_hat, res_dec, reduction="mean")
        loss = ls_1 * loss_gcn + ls_2 * kl_loss + ls_3 * loss_gcn_emb + ls_4 * loss_dec
        print("loss: ", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z_hat_f, q_f, z_f, z_hat_emb_f, _ = model(
            features, adj_l, res_dec, adj_new, save=False
        )

    res2_f = q_f.data.cpu().numpy().argmax(1)
    ari_z, nmi_z = eva(label, res2_f, "Z-q")
    kmeans3 = KMeans(n_clusters=nclass, n_init=20, random_state=42)
    y_pred3 = kmeans3.fit_predict(z_f.data.cpu().numpy())
    ari_k, nmi_k = eva(label, y_pred3, "kmeans-z")

    adata.obsm["daga"] = z_f.to("cpu").detach().numpy()
    adata.obsm["z_hat_f"] = z_hat_f.to("cpu").detach().numpy()

    sc.pp.neighbors(adata, use_rep="daga")
    sc.tl.umap(adata)
    refinement = True
    Clustering(adata=adata, n_clusters=nclass, refinement=refinement)

    adata.obs["refine"] = adata.obs["refine"].astype("int").astype("category")
    adata.obs["ground_truth"] = adata.obs["ground_truth"].astype("category")
    adata.write_h5ad("/path/to/output/" + dataset + "_results.h5ad")

    ARI, NMI = eva(label, adata.obs["mclust"], "mclust-z")
    list1 = [dataset, ARI, NMI, threshold, ls_1, ls_2, ls_3, ls_4]
    data = pd.DataFrame([list1])
    data.to_csv(
        "/path/to/output/find_best_plot.csv",
        mode="a",
        header=False,
        index=False,
    )
    tmp_plot(dataset, ARI, "mclust")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.set_device(2)
    os.environ["R_HOME"] = "/path/to/envs/DRAE/lib/R"
    n_neighbors_a = 3
    config_file = "./config/151675.ini"
    config = Config(config_file)
    train_DAGA_v3(
        update_interval=3,
        tol=1e-3,
        filepath=config["filepath"],
        dataset="151675",
        sew_n_hid_1=config["sew_n_hid_1"],
        n_hid_1=config["n_hid_1"],
        n_hid_2=config["n_hid_2"],
        n_z_l=config["n_z_l"],
        nclass=config["nclass"],
        n_decon=config["n_decon"],
        dropout=config["dropout"],
        alpha=config["alpha"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        seed=config["seed"],
        epochs=config["epochs"],
        ls_1=config["ls_1"],
        ls_2=config["ls_2"],
        ls_3=config["ls_3"],
        ls_4=config["ls_4"],
        n_neighbors_a=n_neighbors_a,
        threshold=config["threshold"],
    )
