import plotly.express as px
from sklearn.decomposition import PCA
from pathlib import Path
import pickle as pk
import numpy as np
import pandas as pd

latent_dir = "data/output/explain/latent"

latents = []

for p in Path(latent_dir).iterdir():
    with open(p, "rb") as f:
        latents.append(pk.load(f))


latents = np.concatenate(latents, 0)
latent_cols = [
    f"latent_{i}" for i in range(
        latents.shape[1]-1)
]
epoch_loss = ["epoch_loss"]
n_components = 3

df = pd.DataFrame(
    latents, columns=latent_cols + epoch_loss)


pca = PCA(n_components=n_components)
components = pca.fit_transform(df[latent_cols])

total_var = pca.explained_variance_ratio_.sum() * 100

labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'Median Price'

fig = px.scatter_matrix(
    components,
    color=df.epoch_loss,
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)

fig.write_html("test.html")
