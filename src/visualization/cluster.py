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
