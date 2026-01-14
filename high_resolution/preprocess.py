import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def preprocess_data(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def construct_interaction_KNN(adata, n_neighbors=12):
    position = adata.obsm["spatial"]
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adata.obsm["graph_neigh"] = interaction

    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm["adj"] = adj
    print("Graph constructed!")


def get_feature(adata):
    adata_Vars = adata[:, adata.var["highly_variable"]]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:,]
    else:
        feat = adata_Vars.X[:,]
    adata.obsm["feat"] = feat


def process_mush_muso(adata):
    if "adj" not in adata.obsm.keys():
        construct_interaction_KNN(adata)
    if "feat" not in adata.obsm.keys():
        get_feature(adata)
    return adata


def get_adata_slide_stereo_data(filepath, dataset):
    counts_file = filepath + "/RNA_counts.tsv"
    coor_file = filepath + "/position.tsv"
    counts = pd.read_csv(counts_file, sep="\t", index_col=0)
    coor_df = pd.read_csv(coor_file, sep="\t")
    print(counts.shape, coor_df.shape)
    counts.columns = ["Spot_" + str(x) for x in counts.columns]
    coor_df.index = coor_df["label"].map(lambda x: "Spot_" + str(x))
    coor_df = coor_df.loc[:, ["x", "y"]]
    coor_df.head()
    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    adata
    coor_df = coor_df.loc[adata.obs_names, ["y", "x"]]
    adata.obsm["spatial"] = coor_df.to_numpy()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    used_barcode = pd.read_csv(filepath + "/used_barcodes.txt", sep="\t", header=None)
    used_barcode = used_barcode[0]
    adata = adata[used_barcode,]
    adata
    sc.pp.filter_genes(adata, min_cells=50)
    print("After flitering: ", adata.shape)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm["label_CSL"] = label_CSL


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized
