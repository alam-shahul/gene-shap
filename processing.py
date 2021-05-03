import scanpy as sc

def parse_pbmc_dataset(filepath):
    """Parse gene expression data and labels from PBMC dataset.

    Args:
        filepath: path to PBMC dataset

    Returns:
        A tuple of (data,   
    """

    data = sc.read_h5ad(filepath)
    data.obs["celltype.l1"]
    dataframe = sc.get.var_df(data)

    return dataframe

if __name__ == "__main__":
    dataframe = parse_pbmc_dataset("pbmc_multimodal.h5ad")
