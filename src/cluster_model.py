import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
import umap


def _savefig_or_show(fig, name, save_dir):
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
        fig.savefig(Path(save_dir) / f"{name}.png", bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def cluster_subscribers(df, seed=42, save_dir=None):
    data = df.drop(columns=['y'], errors='ignore').copy()

    num_cols = data.select_dtypes(include=['number']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()

    # StandardScaler on numerical features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[num_cols])

    # OneHotEncoder on categorical features (no false ordinal relationships by label encoding)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(data[cat_cols])

    # X_all: scaled numericals + one-hot categoricals
    X_all = np.hstack([X_num, X_cat])

    # Correlation heatmap — numerical features only
    corr_df = pd.DataFrame(X_num, columns=num_cols).corr(method='spearman')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True, fmt='.2f', cmap='coolwarm', center=0,
        linewidths=0.5, ax=ax, annot_kws={'size': 9}
    )
    ax.set_title('Feature Correlation (Spearman) — Numerical Features (Subscribers)', fontsize=13)
    plt.tight_layout()
    _savefig_or_show(fig, 'correlation_heatmap', save_dir)

    def plot_correlation_graph(corr_df, save_dir=None):
        feature_pairs = [
            ('age',      'balance'),
            ('age',      'day'),
            ('duration', 'balance'),
            ('campaign', 'day'),
            ('duration', 'campaign'),
        ]
        edges = [(u, v, corr_df.loc[u, v]) for u, v in feature_pairs]

        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        pos = nx.circular_layout(G)

        positive_edges = [(u, v) for u, v, w in edges if w >= 0]
        negative_edges = [(u, v) for u, v, w in edges if w < 0]
        pos_widths = [abs(G[u][v]['weight']) * 40 for u, v in positive_edges]
        neg_widths = [abs(G[u][v]['weight']) * 40 for u, v in negative_edges]

        fig, ax = plt.subplots(figsize=(8, 7))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='steelblue', ax=ax)
        nx.draw_networkx_labels(G, pos, font_color='white', font_size=9, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=positive_edges, width=pos_widths,
                               edge_color='green', ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=negative_edges, width=neg_widths,
                               edge_color='red', style='dashed', ax=ax)
        edge_labels = {(u, v): f'{w:+.2f}' for u, v, w in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

        ax.set_title('Feature Correlation Graph', fontsize=13)
        ax.axis('off')
        plt.tight_layout()
        _savefig_or_show(fig, 'correlation_graph', save_dir)

    plot_correlation_graph(corr_df, save_dir)
    
    # Elbow method — find optimal k
    wcss = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=seed)
        kmeans.fit(X_all)
        wcss.append(kmeans.inertia_)
    drops       = [wcss[i] - wcss[i + 1] for i in range(len(wcss) - 1)]
    second_diff = [drops[i] - drops[i + 1] for i in range(len(drops) - 1)]
    elbow_k     = second_diff.index(max(second_diff)) + 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid()
    ax.plot(range(1, 20), wcss, linewidth=2, color='steelblue', marker='o')
    ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=1.5, label=f'Elbow at k={elbow_k}')
    ax.annotate(
        f'Elbow\nk={elbow_k}',
        xy=(elbow_k, wcss[elbow_k - 1]),
        xytext=(elbow_k + 1.5, wcss[elbow_k - 1] + (wcss[0] - wcss[-1]) * 0.1),
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red', fontsize=11, fontweight='bold'
    )
    ax.set_title('Elbow Method — All Features', fontsize=13)
    ax.set_xlabel('K (number of clusters)')
    ax.set_xticks(range(1, 20))
    ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
    ax.legend()
    plt.tight_layout()
    _savefig_or_show(fig, 'elbow_method', save_dir)
    print(f"Elbow detected at k={elbow_k}")
    
    # Silhouette scores - find optimal k
    silhouette_scores = []
    for k in range(2, 20):
        labels = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=seed).fit_predict(X_all)
        silhouette_scores.append(silhouette_score(X_all, labels))

    silhouet_k = silhouette_scores.index(max(silhouette_scores)) + 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(2, 20), silhouette_scores, marker='o', linewidth=2, color='steelblue')
    ax.axvline(x=silhouet_k, color='red', linestyle='--', linewidth=1.5, label=f'Best k={silhouet_k}')
    ax.set_title('Silhouette Score by K — All Features')
    ax.set_xlabel('K'); ax.set_ylabel('Silhouette Score')
    ax.set_xticks(range(2, 20))
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    _savefig_or_show(fig, 'silhouette_scores', save_dir)
    print(f"Highest silhouette scores at k={silhouet_k}")
    
    best_k = 2
    
    # DR reduces X_all to 2 and 3 components — KMeans then clusters those reduced features
    k_values     = [best_k, 2, 3]
    method_names = ['PCA', 'tSNE', 'UMAP']

    reducers_2d = {
        'PCA':  PCA(n_components=2).fit_transform(X_all),
        'tSNE': TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_all),
        'UMAP': umap.UMAP(n_components=2, random_state=seed).fit_transform(X_all),
    }
    reducers_3d = {
        'PCA':  PCA(n_components=3).fit_transform(X_all),
        'tSNE': TSNE(n_components=3, random_state=seed, perplexity=30).fit_transform(X_all),
        'UMAP': umap.UMAP(n_components=3, random_state=seed).fit_transform(X_all),
    }

    # 2×3 grid — best_k clusters on 2-component (row 1) and 3-component (row 2)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'best_k={best_k} — Clustered on 2 Features (row 1) and 3 Features (row 2)', fontsize=14)
    for col, name in enumerate(method_names):
        X2     = reducers_2d[name]
        lbl2   = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', random_state=seed).fit_predict(X2)
        score2 = silhouette_score(X2, lbl2)
        ax2    = fig.add_subplot(2, 3, col + 1)
        ax2.scatter(X2[:, 0], X2[:, 1], c=lbl2, cmap='viridis', s=15, alpha=0.7)
        ax2.set_title(f'{name} 2 features  |  silhouette={score2:.3f}')
        ax2.set_xlabel(f'{name} 1')
        ax2.set_ylabel(f'{name} 2')

        X3     = reducers_3d[name]
        lbl3   = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', random_state=seed).fit_predict(X3)
        score3 = silhouette_score(X3, lbl3)
        ax3    = fig.add_subplot(2, 3, col + 4, projection='3d')
        ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=lbl3, cmap='viridis', s=10, alpha=0.6)
        ax3.set_title(f'{name} 3 features  |  silhouette={score3:.3f}')
        ax3.set_xlabel(f'{name} 1')
        ax3.set_ylabel(f'{name} 2')
        ax3.set_zlabel(f'{name} 3')
    plt.tight_layout()
    _savefig_or_show(fig, 'dr_clustering_2d3d', save_dir)

    # Cluster profiling — UMAP 2-component labels with best_k
    umap_2d      = reducers_2d['UMAP']
    profile_lbl  = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', random_state=seed).fit_predict(umap_2d)
    profiled     = data.copy()
    profiled['cluster'] = profile_lbl
    profiled['cluster'] = profiled['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1'})
    colors = ['#4e9fc4', '#f4a261']

    # Plot 1 — Numerical means per cluster (simple bar chart)
    means = profiled.groupby('cluster')[num_cols].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    means.T.plot(kind='bar', ax=ax, color=colors, edgecolor='white', width=0.6)
    ax.set_title('Average Value per Feature — Cluster 0 vs Cluster 1', fontsize=13)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Mean value')
    ax.set_xticklabels(num_cols, rotation=0)
    ax.legend(title='')
    ax.grid(axis='y', alpha=0.4)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=8, padding=2)
    plt.tight_layout()
    _savefig_or_show(fig, 'cluster_means', save_dir)

    # Plot 2 — Categorical breakdown per cluster (one subplot per feature)
    n_cat = len(cat_cols)
    fig, axs = plt.subplots(1, n_cat, figsize=(6 * n_cat, 5))
    fig.suptitle('Category Breakdown per Cluster (%) — Cluster 0 vs Cluster 1', fontsize=13)
    if n_cat == 1:
        axs = [axs]
    for ax, col in zip(axs, cat_cols):
        pct = (profiled.groupby(['cluster', col])
                       .size()
                       .groupby(level=0)
                       .apply(lambda s: s / s.sum() * 100)
                       .unstack(fill_value=0))
        pct.T.plot(kind='bar', ax=ax, color=colors, edgecolor='white', width=0.7)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel('% of cluster')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='')
        ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    _savefig_or_show(fig, 'cluster_categories', save_dir)

    # Summary
    print("\n Cluster sizes")
    print(profiled['cluster'].value_counts().to_string())
    print("\nNumerical means per cluster ")
    print(means.round(1).to_string())