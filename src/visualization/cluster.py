from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import pyarrow.parquet as pq
import os
from pathlib import Path
from hdbscan import HDBSCAN
import pandas as pd


def hdbscan_cluster(
    df: pd.DataFrame,
    min_cluster_size: int = 10,
    gen_min_span_tree: bool = True
):

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        gen_min_span_tree=gen_min_span_tree)
    clusterer.fit(df)

    return clusterer.labels_, clusterer.probabilities_


run_id = '4d8ddb41e7f340c182a6a62699502d9f'
score_path = os.path.join("data/output/score", run_id)
cols = ['mask_press', 'resp_flow', 'delivered_volum',
        'mask_press_mu', 'resp_flow_mu', 'delivered_volum_mu',
        'mask_press_se', 'resp_flow_se', 'delivered_volum_se',
        'epoch_id']
scores = []

for f in Path(score_path).iterdir():
    df = pq.read_table(f, columns=cols).to_pandas()
    df["file_name"] = os.path.basename(f)[:15]
    scores.append(df)

scores = pd.concat(scores)

scores = scores.loc[
    scores.delivered_volum != -32768, :]

scores_grouped = scores.groupby(["file_name", "epoch_id"])
means = scores_grouped.mean()
stds = scores_grouped.std()
mins = scores_grouped.min()
maxs = scores_grouped.max()
q05 = scores_grouped.quantile(0.05)
q95 = scores_grouped.quantile(0.95)

scores = pd.concat([means, stds, mins, maxs, q05, q95],
                   axis=1).dropna().reset_index()

scaler = RobustScaler().fit(scores.iloc[:, 2:])
score_scaled = scaler.transform(scores.iloc[:, 2:])


# clusterer
clusterer = HDBSCAN(
    # min_cluster_size=min_cluster_size,
    min_cluster_size=30,
    gen_min_span_tree=True)
clusterer.fit(score_scaled)

labels_ = clusterer.labels_

clusterer = KMeans(n_clusters=5, random_state=0).fit(score_scaled)
labels_ = clusterer.predict(score_scaled)

scores["labels"] = labels_

scores.groupby("labels").agg({"file_name": len})

# fit pca
pca = PCA(n_components=3)
pca.fit(score_scaled)
components = pca.transform(score_scaled)
# create df for visualization
pca_columns = [f"PC{i+1}" for i in range(3)]
components = pd.DataFrame(
    components, columns=pca_columns
).reset_index()
# components = pd.concat(
#     [train_df.reset_index(), components], axis=1)
total_var = pca.explained_variance_ratio_.sum() * 100
labels = {str(i): f"PC {i+1}" for i in range(3)}
labels['color'] = 'class'
# fit latent
pca_fig = px.scatter_matrix(
    components,
    color=labels_.astype("str"),
    dimensions=pca_columns,
    labels=clusterer.labels_,
    title=f'Run: {run_id}; Total Explained Variance: {total_var:.2f}%'  # ,
    # hover_name="file_name",
    # hover_data=["epoch_loss", "epoch"]
)

pca_fig.write_html("test.html")
