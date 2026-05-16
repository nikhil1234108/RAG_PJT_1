import os
import json
import sys
import numpy as np
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CLUSTER_CACHE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "clusters.json")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "article_embeddings.npy")
ARTICLES_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "articles")

def extract_article_embeddings(articles_dir: str = ARTICLES_DIR) -> Tuple[np.ndarray,List[str]]:
    from vectorstore.vector_store import get_embeddings

    url_ids = []
    texts = []
    total_characters = 0

    for fname in sorted(os.listdir(articles_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(articles_dir, fname), encoding = "utf-8", errors="replace") as f:
            text = f.read().strip()

        if text:
            url_ids.append(fname.replace(".txt",""))
            texts.append(text)
            total_characters += len(text)

    if not texts:
        raise ValueError(f"No .txt articles found in {articles_dir}")

    if os.path.exists(EMBEDDINGS_PATH):
        vectors = np.load(EMBEDDINGS_PATH)
        print(f"[embeddings] using cached article embeddings: {EMBEDDINGS_PATH}")
        print(f"shape: {vectors.shape}")
        return vectors, url_ids

    print(f"[articles] loaded {len(texts)} full articles ({total_characters} characters)")
    print("[embeddings] building article-level embeddings")
    embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents(texts), dtype = np.float32)
    print(f"shape: {vectors.shape}")

    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, vectors)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")

    return vectors, url_ids

def find_optimal_k(vectors: np.ndarray,k_range=range(2,11))->Tuple[int, Dict[str, Dict[int, float]]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    inertias:Dict[int, float] = {}
    silhouette_scores:Dict[int, float] = {}
    max_k = min(len(vectors), max(k_range))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        inertias[k] = round(float(km.inertia_),2)
        silhouette_scores[k] = round(float(silhouette_score(vectors, labels)), 4)
        print(f"k={k}, inertia={inertias[k]:.2f}, silhouette={silhouette_scores[k]:.4f}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"best_k by silhouette score: {best_k}")
    return best_k, {
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }

def run_kmeans(vectors:np.ndarray, url_ids:List[str],n_clusters:int = None) -> Dict[str, Any]:
    from sklearn.cluster import KMeans
    metrics = None
    if n_clusters is None:
        n_clusters, metrics = find_optimal_k(vectors)

    print(f"running kmeans with {n_clusters} clusters")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)

    clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
    for url_id, label in zip(url_ids, labels.tolist()):
        clusters[label].append(url_id)

    return {
        "n_clusters": n_clusters,
        "labels": labels.tolist(),
        "url_ids": url_ids,
        "clusters": {f"cluster_{k}": v for k, v in clusters.items()},
        "centroids": km.cluster_centers_.tolist(),
        "inertia": round(float(km.inertia_),2),
        "elbow_data":{str(k): v for k, v in metrics["inertias"].items()} if metrics else None,
        "silhouette_scores":{str(k): v for k, v in metrics["silhouette_scores"].items()} if metrics else None,
        "selection_metric": "silhouette_score",
    }

def run_umap(vectors:np.ndarray, url_ids:List[str],labels:List[int]=None) -> Dict[str, Any]:
    try:
        import umap
        print("running umap from 384 dim -> 2 dim..")
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            n_jobs=1,
        )
        embedding = reducer.fit_transform(vectors)
        method = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        print("umap-learn not found — falling back to PCA...")
        embedding = PCA(n_components=2, random_state=42).fit_transform(vectors)
        method = "PCA"

    return {
        "method": method,
        "url_ids": url_ids,
        "x": embedding[:, 0].tolist(),
        "y": embedding[:, 1].tolist(),
        "labels": labels if labels else [0] * len(url_ids),
    }

def save_cluster_results(kmeans_results:dict, umap_results:dict):
    os.makedirs(os.path.dirname(CLUSTER_CACHE), exist_ok = True)
    with open(CLUSTER_CACHE, "w") as f:
        json.dump({"kmeans":kmeans_results, "umap":umap_results}, f, indent=2)
    print(f"Saved results to {CLUSTER_CACHE}")

def load_cluster_results() ->dict:
    if not os.path.exists(CLUSTER_CACHE):
        return {}
    with open(CLUSTER_CACHE, "r") as f:
        return json.load(f)

def build_clusters(article_dir: str = ARTICLES_DIR):
    print("=" * 50)
    print("BUILDING ARTICLE-LEVEL CLUSTERS")
    print("=" * 50)

    vectors, urlids = extract_article_embeddings(article_dir)
    kmeans_results = run_kmeans(vectors, urlids)
    umap_results = run_umap(vectors, urlids, labels = kmeans_results["labels"])
    save_cluster_results(kmeans_results, umap_results)
    print(f"\nDone. k={kmeans_results['n_clusters']} clusters, "
          f"{len(urlids)} articles.")
    return kmeans_results, umap_results


if __name__ == "__main__":
    km, umap = build_clusters()

    for cid, articles in km["clusters"].items():
        print(f"[clustering] {cid}: {len(articles)} articles")


