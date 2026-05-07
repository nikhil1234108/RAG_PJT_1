import argparse
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vectorstore.clustering import build_clusters, load_cluster_results

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "visualizations")
MPL_CONFIG_DIR = os.path.join(PROJECT_ROOT, "data", "matplotlib_cache")
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)


def get_cluster_results(rebuild: bool = False) -> Dict[str, Any]:
    results = {} if rebuild else load_cluster_results()
    if results:
        print("[visualization] using cached cluster results")
        return results

    print("[visualization] cluster cache not found; building clusters first")
    kmeans_results, umap_results = build_clusters()
    return {"kmeans": kmeans_results, "umap": umap_results}


def make_points(umap_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels = umap_results.get("labels") or [0] * len(umap_results["url_ids"])
    return [
        {
            "url_id": url_id,
            "x": x,
            "y": y,
            "cluster": f"cluster_{label}",
        }
        for url_id, x, y, label in zip(
            umap_results["url_ids"],
            umap_results["x"],
            umap_results["y"],
            labels,
        )
    ]


def save_matplotlib_seaborn_plot(points: List[Dict[str, Any]], output_path: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(points)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="cluster",
        palette="tab10",
        s=95,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    ax.set_title("Article UMAP Clusters")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[visualization] saved Matplotlib/Seaborn PNG: {output_path}")


def save_plotly_plot(points: List[Dict[str, Any]], output_path: str) -> None:
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(points)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_name="url_id",
        hover_data={"x": ":.3f", "y": ":.3f", "cluster": True},
        title="Article UMAP Clusters",
        template="plotly_white",
    )
    fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "white"}})
    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Cluster",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[visualization] saved Plotly HTML: {output_path}")


def get_elbow_points(kmeans_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    elbow_data = kmeans_results.get("elbow_data")
    if not elbow_data:
        return []

    return [
        {"k": int(k), "inertia": inertia}
        for k, inertia in sorted(elbow_data.items(), key=lambda item: int(item[0]))
    ]


def get_silhouette_points(kmeans_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    scores = kmeans_results.get("silhouette_scores")
    if not scores:
        return []

    return [
        {"k": int(k), "silhouette_score": score}
        for k, score in sorted(scores.items(), key=lambda item: int(item[0]))
    ]


def save_elbow_matplotlib_plot(points: List[Dict[str, Any]], output_path: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(points)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="k", y="inertia", marker="o", linewidth=2.5, ax=ax)
    ax.set_title("KMeans Elbow Method")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(df["k"].tolist())
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[visualization] saved elbow PNG: {output_path}")


def save_silhouette_matplotlib_plot(points: List[Dict[str, Any]], output_path: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(points)
    best_row = df.loc[df["silhouette_score"].idxmax()]
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="k", y="silhouette_score", marker="o", linewidth=2.5, ax=ax)
    ax.scatter(
        [best_row["k"]],
        [best_row["silhouette_score"]],
        s=180,
        color="#d62728",
        zorder=5,
        label=f"best k={int(best_row['k'])}",
    )
    ax.set_title("KMeans Silhouette Scores")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(df["k"].tolist())
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[visualization] saved silhouette PNG: {output_path}")


def save_elbow_plotly_plot(points: List[Dict[str, Any]], output_path: str) -> None:
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(points)
    fig = px.line(
        df,
        x="k",
        y="inertia",
        markers=True,
        title="KMeans Elbow Method",
        template="plotly_white",
    )
    fig.update_traces(line={"width": 3}, marker={"size": 10})
    fig.update_layout(
        xaxis_title="Number of clusters (k)",
        yaxis_title="Inertia",
        xaxis={"dtick": 1},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[visualization] saved elbow HTML: {output_path}")


def save_silhouette_plotly_plot(points: List[Dict[str, Any]], output_path: str) -> None:
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(points)
    best_row = df.loc[df["silhouette_score"].idxmax()]
    fig = px.line(
        df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="KMeans Silhouette Scores",
        template="plotly_white",
    )
    fig.add_scatter(
        x=[best_row["k"]],
        y=[best_row["silhouette_score"]],
        mode="markers+text",
        marker={"size": 15, "color": "#d62728"},
        text=[f"best k={int(best_row['k'])}"],
        textposition="top center",
        name="Best k",
    )
    fig.update_traces(line={"width": 3}, marker={"size": 10}, selector={"mode": "lines+markers"})
    fig.update_layout(
        xaxis_title="Number of clusters (k)",
        yaxis_title="Silhouette score",
        xaxis={"dtick": 1},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"[visualization] saved silhouette HTML: {output_path}")


def visualize_umap(rebuild: bool = False) -> None:
    results = get_cluster_results(rebuild=rebuild)
    kmeans_results = results["kmeans"]
    umap_results = results["umap"]
    points = make_points(umap_results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_matplotlib_seaborn_plot(
        points,
        os.path.join(OUTPUT_DIR, "umap_clusters.png"),
    )
    save_plotly_plot(
        points,
        os.path.join(OUTPUT_DIR, "umap_clusters.html"),
    )

    elbow_points = get_elbow_points(kmeans_results)
    if elbow_points:
        save_elbow_matplotlib_plot(
            elbow_points,
            os.path.join(OUTPUT_DIR, "kmeans_elbow.png"),
        )
        save_elbow_plotly_plot(
            elbow_points,
            os.path.join(OUTPUT_DIR, "kmeans_elbow.html"),
        )
    else:
        print("[visualization] no elbow_data found; run with --rebuild to generate it")

    silhouette_points = get_silhouette_points(kmeans_results)
    if silhouette_points:
        save_silhouette_matplotlib_plot(
            silhouette_points,
            os.path.join(OUTPUT_DIR, "kmeans_silhouette.png"),
        )
        save_silhouette_plotly_plot(
            silhouette_points,
            os.path.join(OUTPUT_DIR, "kmeans_silhouette.html"),
        )
    else:
        print("[visualization] no silhouette_scores found; run with --rebuild to generate them")

    print(f"[visualization] plotted {len(points)} articles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cached or freshly built UMAP clusters.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild article embeddings/clusters before plotting.",
    )
    args = parser.parse_args()
    visualize_umap(rebuild=args.rebuild)
