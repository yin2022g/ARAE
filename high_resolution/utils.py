import os, csv, re
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch
from sklearn import metrics
from torch_cluster import knn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
from preprocess import permutation, add_contrastive_label
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import random
from sklearn.mixture import BayesianGaussianMixture


def set_seed(seed=0):
    print(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def target_distribution(q):
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p


def process_data_to_adata(filepath, dataset):
    adata = sc.read_visium(
        filepath, count_file="filtered_feature_bc_matrix.h5", load_images=True
    )
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    df_meta = pd.read_csv(filepath + "/metadata.tsv", sep="\t")
    adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
    adata = adata[~pd.isnull(adata.obs["ground_truth"])]
    res_dec = pd.read_csv(
        "/home/duwenhui/DAGA/data/" + dataset + "/deconProp_best.tsv", sep="\t"
    )
    res_dec = res_dec.set_index(res_dec.index)
    res = res_dec.index.intersection(adata.obs_names)
    res_dec = res_dec.loc[res, :]
    adata = adata[res, :]
    adata.obsm["res_dec"] = res_dec
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    return adata


def load_data_new(filepath, dataset, n_neighbors_a):
    adata = sc.read_visium(
        filepath, count_file="filtered_feature_bc_matrix.h5", load_images=True
    )
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    df_meta = pd.read_csv(filepath + "/metadata.tsv", sep="\t")
    adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
    adata = adata[~pd.isnull(adata.obs["ground_truth"])]
    adata.var_names = [i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    construct_adj_l(adata, n_neighbors_a)
    adj_l = preprocess_adj(adata.obsm["adj"])
    add_contrastive_label(adata)
    if "highly_variable" in adata.var.columns:
        adata_Vars = adata[:, adata.var["highly_variable"]]
    else:
        adata_Vars = adata
    features = pd.DataFrame(
        adata_Vars.X.toarray()[:,],
        index=adata_Vars.obs.index,
        columns=adata_Vars.var.index,
    )
    features_a = permutation(adata_Vars.X.toarray()[:,])
    category = pd.Categorical(adata.obs["ground_truth"])
    label = category.codes
    return adata, features, features_a, adj_l, label


def load_data(filepath, dataset, n_neighbors_a):
    adata = process_data_to_adata(filepath, dataset)
    construct_adj_l(adata, n_neighbors_a)
    adj_l = preprocess_adj(adata.obsm["adj"])
    add_contrastive_label(adata)
    if "highly_variable" in adata.var.columns:
        adata_Vars = adata[:, adata.var["highly_variable"]]
    else:
        adata_Vars = adata
    features = pd.DataFrame(
        adata_Vars.X.toarray()[:,],
        index=adata_Vars.obs.index,
        columns=adata_Vars.var.index,
    )
    features_a = permutation(adata_Vars.X.toarray()[:,])
    print("size of features: ", features.shape)
    print("Size of Input: ", adata_Vars.shape)
    category = pd.Categorical(adata.obs["ground_truth"])
    label = category.codes
    return adata, features, features_a, adj_l, label


def calculate_distance(x):
    assert isinstance(x, np.ndarray) and x.ndim == 2
    x_square = np.expand_dims(np.einsum("ij,ij->i", x, x), axis=1)
    y_square = x_square.T
    distances = np.dot(x, x.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, distances)
    distances.flat[:: distances.shape[0] + 1] = 0.0
    np.sqrt(distances, distances)
    return distances


def get_knn_pyg(adata):
    position = adata.obsm["spatial"]
    position = torch.FloatTensor(position)
    edge_index = knn(position, position, k=6)
    return edge_index


def get_data(feats, adata):
    edge_index = get_knn_pyg(adata)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = add_self_loops(edge_index)[0]
    feats = np.array(feats)
    feats = torch.tensor(feats, dtype=torch.float)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    data = Data(x=feats, edge_index=edge_index, adj_t=adj.t())
    return data


def construct_adj_l(adata, n_neighbors=3):
    position = adata.obsm["spatial"]
    distance_matrix = calculate_distance(position.astype(np.float64))
    n_spot = distance_matrix.shape[0]
    adata.obsm["distance_matrix"] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
    adata.obsm["graph_neigh"] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm["adj"] = adj
    return adj


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj) + sp.eye(adj.shape[0])
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def BGMM(adata, n_cluster, used_obsm="daga"):
    knowledge = BayesianGaussianMixture(
        n_components=n_cluster,
        weight_concentration_prior_type="dirichlet_process",  ##'dirichlet_process' or dirichlet_distribution'
        weight_concentration_prior=50,
    ).fit(adata.obsm[used_obsm])

    method = "mclust"
    labels = knowledge.predict(adata.obsm[used_obsm]) + 1
    adata.obs[method] = labels
    adata.obs[method] = adata.obs[method].astype("int")
    adata.obs[method] = adata.obs[method].astype("category")
    return adata


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="daga", random_seed=2020):
    import rpy2.robjects as robjects
    from rpy2.robjects import r

    if used_obsm not in adata.obsm.keys():
        raise ValueError(f"'{used_obsm}' not found in adata.obsm")
    data = adata.obsm[used_obsm]
    if sp.issparse(data):
        print("Converting sparse matrix to dense...")
        data = data.toarray()
    data_np = np.array(data, dtype=np.float64, copy=True)
    if np.isnan(data_np).any() or np.isinf(data_np).any():
        print("⚠️ Warning: Data contains NaNs or Infs. Filling NaNs with 0.")
        data_np = np.nan_to_num(data_np)
    n_rows, n_cols = data_np.shape
    print(f"Data shape: {n_rows} rows, {n_cols} columns")
    flat_data = data_np.flatten()
    r_vec = robjects.FloatVector(flat_data)

    robjects.globalenv["r_vec"] = r_vec
    robjects.globalenv["n_rows"] = n_rows
    robjects.globalenv["n_cols"] = n_cols
    robjects.globalenv["n_clust"] = num_cluster
    robjects.globalenv["r_seed"] = random_seed
    robjects.globalenv["model_name"] = modelNames

    r_script = """
    library(mclust)
    set.seed(r_seed)
    
    data_mat <- matrix(r_vec, nrow = n_rows, ncol = n_cols, byrow = TRUE)
    
    colnames(data_mat) <- paste0("Dim", 1:n_cols)
    
    res <- Mclust(data_mat, G = n_clust, modelNames = model_name, verbose = FALSE)
    
    if (is.null(res)) {
        stop("Mclust returned NULL")
    }
    m_class <- res$classification
    """

    print("Running Mclust in R...")
    try:
        r(r_script)
    except Exception as e:
        print("❌ Error executing R script.")
        print(r("geterrmessage()"))
        raise e

    mclust_res = np.array(robjects.globalenv["m_class"])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int")
    adata.obs["mclust"] = adata.obs["mclust"].astype("category")

    r("rm(r_vec, data_mat, res)")

    print("Mclust finished successfully.")
    return adata


def Clustering(adata, n_clusters=7, radius=50, refinement=False):
    adata = mclust_R(adata=adata, num_cluster=n_clusters, used_obsm="daga")
    if refinement:
        new_type = refine_label(adata=adata, radius=radius, key="mclust")
        adata.obs["refine"] = new_type


def refine_label(adata, radius=25, key="mclust"):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm["spatial"]
    distance_matrix = calculate_distance(position.astype(np.float64))
    n_cell = distance_matrix.shape[0]
    for i in range(n_cell):
        vec = distance_matrix[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def knn_cosine_similarity(features, k):
    inner_product = torch.matmul(features, features.transpose(0, 1))
    norm = torch.sqrt(torch.sum(torch.square(features), dim=1))
    norm_matrix = torch.matmul(norm.reshape(-1, 1), norm.reshape(1, -1))
    cosine_similarity = torch.div(inner_product, norm_matrix + 1e-8)
    cosine_similarity.fill_diagonal_(-float("inf"))
    top_k_similarities, top_k_indices = torch.topk(cosine_similarity, k=k, dim=1)
    num_nodes = cosine_similarity.shape[0]
    knn_matrix = torch.zeros((num_nodes, num_nodes))
    knn_matrix = knn_matrix.cuda()
    knn_matrix.scatter_(1, top_k_indices, top_k_similarities)
    return knn_matrix


def cosine_similarity(features, threshold):
    inner_product = torch.matmul(features, features.transpose(0, 1))
    norm = torch.sqrt(torch.sum(torch.square(features), dim=1))
    norm_matrix = torch.matmul(norm.reshape(-1, 1), norm.reshape(1, -1))
    cosine_similarity = torch.div(inner_product, norm_matrix + 1e-8)
    cosine_similarity[cosine_similarity < threshold] = 0
    cosine_similarity[cosine_similarity >= threshold] = 1
    cosine_similarity.fill_diagonal_(1)
    return cosine_similarity


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def knbrsloss(H, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(
        H.cpu().detach().numpy()
    )
    _, indices = nbrs.kneighbors(H.cpu().detach().numpy())
    f = lambda x: torch.exp(x / 1.0)
    refl_sim = f(sim(H, H))
    V = torch.zeros((H.shape[0], k)).cuda()
    for i in range(H.shape[0]):
        for j in range(k):
            V[i][j] += refl_sim[i][indices[i][j + 1]]
    ret = -torch.log(V.sum(1) / (refl_sim.sum(1) - refl_sim.diag()))
    ret = ret.mean()
    return ret
