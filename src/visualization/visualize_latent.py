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
col_names = [
    f"latent_{i}" for i in range(
        latents.shape[1])
]
n_components = 3

df = pd.DataFrame(
    latents, columns=col_names)


pca = PCA(n_components=n_components)
components = pca.fit_transform(df)

total_var = pca.explained_variance_ratio_.sum() * 100

labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'Median Price'

fig = px.scatter_matrix(
    components,
    # color=boston.target,
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)

fig.write_html("test.html")
