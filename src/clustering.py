import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# these are counts with a long tail so they get log transformed first
SKEWED = ["reviews_per_day", "new_cards_per_day", "median_seconds"]


def prepare(table, features):
    X = table[features].copy()
    for col in SKEWED:
        if col in X:
            X[col] = np.log1p(X[col])
    return StandardScaler().fit_transform(X)


def try_k_values(X, k_values=range(2, 9), seed=42):
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit_methods(X, k, seed=42):
    labels = {
        "k-means": KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X),
        "hierarchical (ward)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
        "gaussian mixture": GaussianMixture(n_components=k, n_init=5, random_state=seed).fit_predict(X),
    }
    return labels


def compare_methods(X, labels):
    rows = []
    for name, lab in labels.items():
        rows.append({
            "method": name,
            "silhouette (higher better)": silhouette_score(X, lab),
            "davies-bouldin (lower better)": davies_bouldin_score(X, lab),
            "calinski-harabasz (higher better)": calinski_harabasz_score(X, lab),
            "cluster sizes": sorted(np.bincount(lab).tolist(), reverse=True),
        })
    return pd.DataFrame(rows)


def stability(X, k, runs=20, seed=42):
    # fit k-means on two random halves, label everyone with both models,
    # and see how much the two labellings agree (1 = identical, 0 = random)
    rng = np.random.default_rng(seed)
    scores = []
    for r in range(runs):
        idx = rng.permutation(len(X))
        half_a, half_b = idx[: len(X) // 2], idx[len(X) // 2:]
        km_a = KMeans(n_clusters=k, n_init=10, random_state=r).fit(X[half_a])
        km_b = KMeans(n_clusters=k, n_init=10, random_state=r).fit(X[half_b])
        scores.append(adjusted_rand_score(km_a.predict(X), km_b.predict(X)))
    return np.array(scores)


def profile_table(table, features, labels):
    z = (table[features] - table[features].mean()) / table[features].std()
    z["cluster"] = labels
    means = z.groupby("cluster").mean()
    means["size"] = pd.Series(labels).value_counts().sort_index()
    return means
