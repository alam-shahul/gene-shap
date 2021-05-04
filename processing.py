import scanpy as sc

def parse_pbmc_dataset(filepath):
    """Parse gene expression data and labels from PBMC dataset.

    Args:
        filepath: path to PBMC dataset

    Returns:
        A tuple of (data, (labels_1, labels_2, labels_3), genes)   
    """

    data = sc.read_h5ad(filepath)
    labels_1 = data.obs["celltype.l1"]
    labels_2 = data.obs["celltype.l2"]
    labels_3 = data.obs["celltype.l3"]
    dataframe = sc.get.var_df(data)
    genes = dataframe.index.values

    return data.X, (labels_1, labels_2, labels_3), genes

if __name__ == "__main__":
    data, labels, genes = parse_pbmc_dataset("pbmc_multimodal.h5ad")
