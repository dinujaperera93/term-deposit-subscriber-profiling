import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (recall_score, make_scorer, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, silhouette_score)


def _savefig_or_show(fig, name, save_dir):
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
        fig.savefig(Path(save_dir) / f"{name}.png", bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def cluster_subscribers(df, save_dir=None):
    # numerical columns in order: age(0), balance(1), day(2), duration(3), campaign(4)
    df = df.select_dtypes(include=['number'])
    # Bring all the variables to the same magnitude
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    X = np.array(df)

    # Combined figure: 2D pairs (top 2 rows) + 3D triplets (bottom 2 rows)
    # column order: age(0), balance(1), day(2), duration(3), campaign(4)
    pairs = [
        (0, 1, 'age',      'balance'),
        (0, 3, 'age',      'duration'),
        (0, 4, 'age',      'campaign'),
        (3, 4, 'duration', 'campaign'),
        (4, 2, 'campaign', 'day'),
        (3, 2, 'duration', 'day'),
    ]
    triplets = [
        (0, 1, 3, 'age',      'balance',  'duration'),
        (1, 3, 4, 'balance',  'duration', 'campaign'),
        (0, 3, 4, 'age',      'duration', 'campaign'),
        (4, 2, 3, 'campaign', 'day',      'duration'),
    ]
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle('Feature Scatter Plots — 2D Pairs and 3D Triplets (Standardised)', fontsize=16)

    for i, (xi, yi, xlabel, ylabel) in enumerate(pairs):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.scatter(X[:, xi], X[:, yi], s=25, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')

    for i, (xi, yi, zi, xlabel, ylabel, zlabel) in enumerate(triplets):
        ax = fig.add_subplot(4, 3, 7 + i, projection='3d')
        ax.scatter(X[:, xi], X[:, yi], X[:, zi], s=20, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(f'{xlabel}/{ylabel}/{zlabel}')

    plt.tight_layout()
    _savefig_or_show(fig, 'scatter_pairs', save_dir)

    # Use all 5 numerical features for clustering: age, balance, day, duration, campaign
    fig, axs = plt.subplots(3, 6, figsize=(30, 12))
    fig.suptitle('KMeans Clustering Analysis: age, balance, day, duration, campaign', fontsize=20)
    axs = axs.flatten()

    for k in range(2, 20):
        model = KMeans(n_clusters=k, init='k-means++', n_init='auto')
        cluster_labels = model.fit_predict(X)

        axs[k-2].scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap='viridis')
        axs[k-2].set_title(f'k={k}')
        centroids = model.cluster_centers_[:, :2]
        axs[k-2].scatter(centroids[:, 0], centroids[:, 1], marker='o', c='yellow', s=200, edgecolors='black', label='Centroids')

    _savefig_or_show(fig, 'kmeans_k2_to_k20', save_dir)

    # Set up the subplot layout
    fig, axs = plt.subplots(3, 6, figsize=(30, 12))
    fig.suptitle('KMeans Clustering with Silhouette Score', fontsize=20)
    axs = axs.flatten()

    for k in range(2, 20):
        model = KMeans(n_clusters=k, init='k-means++', n_init='auto')
        cluster_labels = model.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        axs[k-2].scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap='viridis')
        axs[k-2].set_title(f'k={k}, Silhouette Score: {score:.4f}')
        centroids = model.cluster_centers_[:, :2]
        axs[k-2].scatter(centroids[:, 0], centroids[:, 1], marker='o', c='yellow', s=200, edgecolors='black', label='Centroids')

    plt.tight_layout()
    _savefig_or_show(fig, 'silhouette_scores', save_dir)

    model = KMeans(n_clusters=2, init='k-means++', n_init='auto')
    cluster_labels = model.fit_predict(X)

    df1 = pd.DataFrame(df[cluster_labels == 0])
    print(df1.describe())

    df2 = pd.DataFrame(df[cluster_labels == 1])
    print(df2.describe())

    # Scatter plots for correlated feature pairs (from correlation map)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('KMeans Clustering - Correlated Feature Pairs', fontsize=16)

    # age vs balance (corr: 0.08)
    axs[0].scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap='viridis')
    axs[0].set_title('age vs balance (corr: 0.08)')
    axs[0].set_xlabel('age')
    axs[0].set_ylabel('balance')

    # age vs duration (corr: -0.04)
    axs[1].scatter(X[:, 0], X[:, 3], c=cluster_labels, s=50, cmap='viridis')
    axs[1].set_title('age vs duration (corr: -0.04)')
    axs[1].set_xlabel('age')
    axs[1].set_ylabel('duration')

    # day vs campaign (corr: 0.17)
    axs[2].scatter(X[:, 2], X[:, 4], c=cluster_labels, s=50, cmap='viridis')
    axs[2].set_title('day vs campaign (corr: 0.17)')
    axs[2].set_xlabel('day')
    axs[2].set_ylabel('campaign')
    
    # age vs campaign (corr: 0.02)
    axs[0].scatter(X[:, 0], X[:, 4], c=cluster_labels, s=50, cmap='viridis')
    axs[0].set_title('age vs campaign (corr: 0.02)')
    axs[0].set_xlabel('age')
    axs[0].set_ylabel('campaign')

    # day vs duration (corr: -0.03)
    axs[1].scatter(X[:, 2], X[:, 3], c=cluster_labels, s=50, cmap='viridis')
    axs[1].set_title('day vs duration (corr: -0.03)')
    axs[1].set_xlabel('day')
    axs[1].set_ylabel('duration')

    # balance vs duration (corr: 0.17)
    axs[2].scatter(X[:, 1], X[:, 3], c=cluster_labels, s=50, cmap='viridis')
    axs[2].set_title('balance vs duration (corr: 0.17)')
    axs[2].set_xlabel('balance')
    axs[2].set_ylabel('duration')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig_or_show(fig, 'correlated_pairs', save_dir)

    # The elbow method
    wcss = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    fig = plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(range(1, 20), wcss, linewidth=2, color="blue", marker="8")
    plt.xlabel("K")
    plt.xticks([x for x in range(1, 20)])
    plt.ylabel("WCSS")
    _savefig_or_show(fig, 'elbow_method', save_dir)
    print(wcss)

    plot_correlation_graph(save_dir)


def plot_3d_scatter(X, cluster_labels, save_dir=None):
    # column order: age(0), balance(1), day(2), duration(3), campaign(4)
    # 4 most meaningful triplets based on feature importance and correlations
    triplets = [
        (0, 1, 3, 'age',      'balance',  'duration'),
        (1, 3, 4, 'balance',  'duration', 'campaign'),
        (0, 3, 4, 'age',      'duration', 'campaign'),
        (4, 2, 3, 'campaign', 'day',      'duration'),
    ]
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('3D Scatter Plots — Most Meaningful Feature Triplets (k=2)', fontsize=15)

    for i, (xi, yi, zi, xlabel, ylabel, zlabel) in enumerate(triplets):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.scatter(X[:, xi], X[:, yi], X[:, zi],
                   c=cluster_labels, s=20, cmap='viridis', alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(f'{xlabel} / {ylabel} / {zlabel}')

    plt.tight_layout()
    _savefig_or_show(fig, '3d_scatter', save_dir)


def plot_correlation_graph(save_dir=None):
    edges = [
        ('age',      'balance',  0.08),
        ('age',      'duration', -0.04),
        ('age',      'campaign',  0.02),
        ('duration', 'campaign', -0.09),
        ('campaign', 'day',       0.17),
        ('duration', 'day',      -0.03),
    ]

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
