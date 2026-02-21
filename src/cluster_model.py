import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    X = np.array(df)
    column_indices = df.columns
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], s=25)
    plt.title('Scatter Plot of age vs balance (Standardised)')
    plt.xlabel('age')
    plt.ylabel('balance')
    _savefig_or_show(fig, 'scatter_age_balance', save_dir)

    # Use all 5 numerical features for clustering: age, balance, day, duration, campaign
    X_2 = X
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('KMeans Clustering Analysis: age, balance, day, duration, campaign', fontsize=20)
    axs = axs.flatten()

    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init='auto')
        cluster_labels = model.fit_predict(X_2)

        axs[k-2].scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap='viridis')
        axs[k-2].set_title(f'k={k}')
        centroids = model.cluster_centers_[:, :2]
        axs[k-2].scatter(centroids[:, 0], centroids[:, 1], marker='o', c='yellow', s=200, edgecolors='black', label='Centroids')

    _savefig_or_show(fig, 'kmeans_k2_to_k6', save_dir)

    # Set up the subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('KMeans Clustering with Silhouette Score', fontsize=16)
    axs = axs.flatten()

    for k in range(2, 6):
        model = KMeans(n_clusters=k, n_init='auto')
        cluster_labels = model.fit_predict(X_2)

        score = silhouette_score(X_2, cluster_labels)

        axs[k-2].scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, cmap='viridis')
        axs[k-2].set_title(f'k={k}, Silhouette Score: {score:.4f}')
        centroids = model.cluster_centers_[:, :2]
        axs[k-2].scatter(centroids[:, 0], centroids[:, 1], marker='o', c='yellow', s=200, edgecolors='black', label='Centroids')

    plt.tight_layout()
    _savefig_or_show(fig, 'silhouette_scores', save_dir)

    model = KMeans(n_clusters=2, n_init='auto')
    cluster_labels = model.fit_predict(X_2)

    df1 = pd.DataFrame(df[cluster_labels == 0])
    print(df1.describe())

    df2 = pd.DataFrame(df[cluster_labels == 1])
    print(df2.describe())

    # Scatter plots for correlated feature pairs (from correlation map)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig_or_show(fig, 'correlated_pairs', save_dir)

    # The elbow method
    wcss = []
    for k in range(1, 7):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(X_2)
        wcss.append(kmeans.inertia_)
    fig = plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(range(1, 7), wcss, linewidth=2, color="blue", marker="8")
    plt.xlabel("K")
    plt.xticks([x for x in range(1, 7)])
    plt.ylabel("WCSS")
    _savefig_or_show(fig, 'elbow_method', save_dir)
    print(wcss)
