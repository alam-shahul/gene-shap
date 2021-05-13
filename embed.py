from vae.train import VAE, trainVAE

import scipy.sparse
from pathlib import Path

if __name__ == "__main__":
    latent_dimension = 10
    batch_size = 100

    embeddings_directory = Path("embeddings")
    embeddings_directory.mkdir(parents=True, exist_ok=True)
    data = scipy.sparse.load_npz("raw_gene_expression.npz").toarray()
    trainVAE(data, str(embeddings_directory), "pbmc")

    model = VAE(input_dim=data.shape[1], latent_dim=latent_dimension)
    model.load_state_dict(torch.load("embeddings/pbmc.pt")["model_state_dict"])
    
    results = np.zeros((data.shape[0], latent_dimension))
    for batch in range(len(data) // batch_size + 1):
       batch_data = data[batch * batch_size : (batch+1) * batch_size].astype(np.float32)
       padded_batch_data = np.zeros((batch_size, data.shape[1])).astype(np.float32)
       padded_batch_data[:len(batch_data)] = batch_data
       _, latent_z, *_ = model.forward(padded_batch_data)
       results[batch * batch_size : (batch+1) * batch_size] = latent_z.detach()[:len(batch_data)]
       print(batch)
    np.save(embeddings_directory / "scvis_embeddings.npy", results)

