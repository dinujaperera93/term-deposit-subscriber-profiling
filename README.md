# Term Deposit Marketing Prediction and Subscriber Segmentation

An end-to-end machine learning project that predicts **which banking customers are likely to subscribe to a term deposit** and segments confirmed subscribers into meaningful customer groups for deeper business insight.

The solution combines:

- A **two-layer supervised learning pipeline** for campaign targeting
- **Subscriber segmentation using clustering**
- Careful treatment of **class imbalance, preprocessing, and data leakage**

---

# Business Problem

Banks often run marketing campaigns to promote financial products such as **term deposits**. Contacting every customer is expensive and inefficient, while conversion rates remain low.

Marketing teams therefore need a way to identify:

1. **Which customers should be contacted first**
2. **Which contacted customers are most likely to subscribe**

This project develops a machine learning pipeline that helps marketing teams focus their efforts on customers with the highest probability of subscribing.

---

# Project Summary

The project implements a **two-layer modelling strategy** followed by **subscriber clustering**.

### Layer 1 – Pre-call Targeting
Uses only customer demographic and financial information that is available **before a campaign call** to predict which customers should be contacted.

### Layer 2 – Post-call Follow-up
Uses campaign information collected **after initial contact** to predict which customers are most likely to convert.

### Subscriber Segmentation
Customers who subscribed are grouped into clusters to identify different **types of subscribers**, helping banks design more targeted marketing strategies.

This structure reflects how real marketing campaigns operate in practice.

---

# Dataset

The dataset used in this project is the **Term Deposit Marketing dataset**, which contains information collected from banking marketing campaigns.

Each record represents a customer and the outcome of the marketing campaign.

## Target Variable

| Variable | Description |
|---------|-------------|
| y | Whether the client subscribed to a term deposit |

Values:
- **yes** → client subscribed  
- **no** → client did not subscribe  

## Feature Groups

### Pre-call features
These features are available before contacting the customer.

- age  
- job  
- marital  
- education  
- default  
- balance  
- housing  
- loan  

### Post-call features
These become available only after the campaign interaction.

- contact  
- day  
- month  
- duration  
- campaign  

Separating these feature groups prevents **data leakage** and ensures the model reflects realistic business usage.

---

# Project Workflow

```
Raw Marketing Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning and Preprocessing
        │
        ├── Handle encoded missing values ("unknown")
        ├── Outlier treatment
        ├── Train/validation/test split
        ├── Feature encoding
        └── Feature scaling
        │
        ▼
Two-Layer Supervised Modelling
        │
        ├── Model 1: Pre-call targeting
        └── Model 2: Post-call follow-up
        │
        ▼
Model Evaluation and Interpretation
        │
        ▼
Subscriber Segmentation
        │
        ├── Dimensionality reduction
        ├── KMeans clustering
        └── Cluster interpretation
        │
        ▼
Business Insights
```

---

# Step-by-Step Explanation

## 1. Data Loading

The dataset is first loaded and organized into **pre-call** and **post-call** feature groups. This ensures that the first model only uses information that would realistically be available before contacting customers.

---

## 2. Exploratory Data Analysis

Exploratory analysis was conducted to understand:

- Feature distributions
- Relationships between variables
- Class imbalance in the target variable
- Missing or encoded values such as `"unknown"`

The analysis revealed a strong **class imbalance**, meaning that far fewer customers subscribe compared to those who do not.

---

## 3. Data Preprocessing

A structured preprocessing pipeline was implemented.

### Key preprocessing steps

- **Stratified dataset splitting**  
  Ensures that class proportions remain consistent across training and testing datasets.

- **Handling missing-like values**  
  Values labeled `"unknown"` were treated carefully rather than automatically removed.

- **Outlier treatment**  
  Extreme values were clipped using an interquartile range approach.

- **Categorical encoding**  
  Categorical variables were converted into numerical representations.

- **Feature scaling**  
  Numerical variables were standardized to support model training.

---

## 4. Two-Layer Supervised Modelling

### Model 1 – Pre-call Targeting

Purpose:  
Predict which customers should be contacted before a marketing call.

Features used:
- demographic information
- financial information
- account characteristics

Business question answered:

> Which customers should the bank call?

This helps reduce unnecessary outreach and lowers marketing costs.

---

### Model 2 – Post-call Follow-up

Purpose:  
Predict which contacted customers are most likely to subscribe.

Features used:
- pre-call features
- campaign interaction features

Business question answered:

> Which customers should remain a priority after initial contact?

This allows marketing teams to allocate resources efficiently during campaigns.

---

## 5. Model Development and Optimization

Several modelling approaches were explored and compared.

### Model development strategy

- **LazyPredict**
  - quickly evaluates many baseline classifiers

- **Ensemble model comparison**
  - identifies strong candidate algorithms

- **Hyperparameter optimization**
  - performed using **Hyperopt with the TPE algorithm**

- **Final model**
  - **LightGBM**, chosen for its strong performance on imbalanced datasets and ability to capture nonlinear relationships.

---

## 6. Model Evaluation

Evaluation focuses on metrics that are meaningful for marketing campaigns.

### Key evaluation metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Because the dataset is imbalanced, **recall for actual subscribers** is particularly important. Missing potential subscribers could reduce campaign effectiveness.

---

# Subscriber Segmentation

Beyond prediction, the project explores **customer segmentation among subscribers**.

### Objective

Identify different types of customers who subscribe to term deposits.

### Clustering process

1. Filter only subscribed customers
2. Encode categorical variables using One-Hot Encoding
3. Scale numerical features using StandardScaler
4. Apply dimensionality reduction techniques such as:
   - PCA
   - t-SNE
   - UMAP
5. Determine optimal k using the Elbow method and Silhouette scores
6. Perform clustering using **KMeans** on the full feature space
7. Analyze cluster characteristics via z-score heatmap and categorical breakdown charts

---

### Cluster Profiles

KMeans with k=6 partitions the 2,896 confirmed subscribers into six behaviorally
distinct segments. Interpretation draws on three sources of evidence: the **cluster
size distribution**, the **z-score heatmap** (deviation from the overall subscriber
mean on each numerical feature), and the **categorical breakdown bar charts**.

---

#### Cluster 0 — Young Moderate Subscribers

**Size:** 779 customers (26.9%)

**Characteristics:**
- Young customers (average age ≈ 33)
- Lower account balances (≈ €893)
- Moderate call duration (≈ 568 s)
- Few campaign contacts (2.0)
- Contacted early in the month (day ≈ 8)

**Interpretation:**
Cluster 0 represents younger customers with lower balances who are receptive to standard outreach. Their defining characteristic is being contacted very early in the month (day ≈ 8), suggesting campaign timing plays a role in their conversion.

**Business implication:**
Target this segment with standard campaign outreach in the first week of the month.

---

#### Cluster 1 — Highly Engaged Subscribers

**Size:** 561 customers (19.4%)

**Characteristics:**
- Moderate age (≈ 39)
- Very long call duration (≈ 1,257 seconds)
- Moderate balances (≈ €1,082)
- Average campaign attempts (2.4)

**Interpretation:**
Cluster 1 consists of mid-age customers whose defining trait is a very long average call duration of approximately 21 minutes — the highest of any cluster. This indicates they need thorough product explanation and discussion before committing to a decision.

**Business implication:**
Assign experienced agents capable of conducting in-depth product discussions. Do not end calls prematurely — extended engagement is a positive signal for this segment.

---

#### Cluster 2 — Affluent Subscribers

**Size:** 100 customers (3.5%)

**Characteristics:**
- Extremely high balances (≈ €12,796)
- Average call duration (≈ 609 s)
- Moderate age (≈ 43)
- Few campaign contacts (2.2)

**Interpretation:**
Cluster 2 represents a small but financially significant group of affluent customers with exceptionally high account balances (≈ €12,796 — more than ten times the overall subscriber average). They subscribe with few contacts and moderate call durations, indicating they are financially confident and require little persuasion.

**Business implication:**
Prioritise this segment for premium or high-yield term deposit offers and relationship-based outreach. Despite being the smallest cluster, their deposit value per customer is disproportionately high.

---

#### Cluster 3 — Persistence-Driven Subscribers

**Size:** 159 customers (5.5%)

**Characteristics:**
- Highest number of campaign contacts (≈ 9.5)
- Moderate balances (≈ €1,277)
- Moderate age (≈ 40)
- Medium call duration (≈ 788 s)

**Interpretation:**
Cluster 3 includes customers who eventually subscribe but only after a high number of repeated contacts — approximately 9.5 on average, nearly five times the overall mean. This suggests initial resistance or difficulty in reaching them, but persistence ultimately drives conversion.

**Business implication:**
Implement structured multi-touch follow-up campaigns for customers who do not convert on early contacts. Consider automated follow-up sequences (SMS or email) to reduce the cost of repeated live outreach.

---

#### Cluster 4 — Older Subscribers

**Size:** 599 customers (20.7%)

**Characteristics:**
- Oldest age group (≈ 55)
- Moderate balances (≈ €1,571)
- Shorter call durations (≈ 529 s)
- Few campaign contacts (1.9)

**Interpretation:**
Cluster 4 consists of the oldest subscriber group, characterised by quick and confident decision-making. They subscribe with among the fewest campaign contacts (1.9) and below-average call durations, suggesting familiarity with savings products and little need for persuasion.

**Business implication:**
Prioritise older customers early in campaigns. A concise, rate-focused message is sufficient — over-engineering the sales approach is unnecessary for this segment.

---

#### Cluster 5 — Late-Month Young Subscribers

**Size:** 698 customers (24.1%)

**Characteristics:**
- Younger customers (≈ 34)
- Lower call duration (≈ 469 s)
- Campaign contacts occur later in the month (day ≈ 24)
- Moderate balances (≈ €1,252)
- Fewest campaign contacts of any cluster (1.7)

**Interpretation:**
Cluster 5 contains younger customers who are the most efficient to convert — they have the shortest average call duration (≈ 469 s) and the fewest campaign contacts (1.7) of any cluster, while being contacted late in the month (day ≈ 24). This suggests they are predisposed to subscribe and respond quickly when reached at the right time.

**Business implication:**
Target younger customers with concise, timely outreach in the final week of the month. High-volume, low-effort contact in this window maximises campaign conversion efficiency.

---

### Overall Business Conclusions

From the clustering results, several key insights emerge:

- Subscriber behavior is heterogeneous, forming six meaningful segments.
- Different segments subscribe for different reasons, including financial capacity, marketing persistence, demographic factors, and call engagement.
- A small group of high-balance customers forms a particularly valuable segment.
- Some customers respond quickly, while others require multiple marketing contacts.
- Marketing strategies should therefore be tailored to each subscriber segment rather than applying a single campaign strategy.

**Final Summary:**
The clustering analysis identifies six distinct subscriber segments characterised by differences in age, financial balance, call engagement, and marketing exposure. These insights suggest that targeted marketing strategies tailored to specific customer segments could significantly improve term deposit campaign efficiency and conversion rates.

---

# Repository Structure

```text
.
├── data/
│   └── term-deposit-marketing-2020.csv
├── figures/
│   └── supervised model outputs
├── figures_Clustering/
│   ├── cluster_categories.png
│   ├── cluster_interpretation_heatmap.png
│   ├── cluster_means.png
│   ├── cluster_size_distribution.png
│   ├── correlation_graph.png
│   ├── correlation_heatmap.png
│   ├── dr_clustering_2d3d.png
│   ├── elbow_method.png
│   └── silhouette_scores.png
├── src/
│   ├── __init__.py
│   ├── cluster_model.py
│   └── two_layer_model.py
├── analysis.ipynb
├── analysis.py
├── config.py
├── main.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

# Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Hyperopt
- Matplotlib
- Seaborn
- DuckDB
- Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone https://github.com/dinujaperera93/3KLtANr8ONizlTsO
cd 3KLtANr8ONizlTsO
```

Install dependencies:

```bash
pip install -e .
```

---

# Running the Project

### Run the analysis notebook

Open the notebook to explore the full workflow:

```
analysis.ipynb
```

### Run the main script

```bash
python main.py
```

---

# Key Contributions

This project demonstrates the ability to:

- Translate a real business problem into a machine learning workflow
- Design a **two-stage predictive modelling pipeline**
- Prevent **data leakage** through careful feature separation
- Address **class imbalance in real-world datasets**
- Apply **Bayesian hyperparameter optimization**
- Combine **supervised prediction with unsupervised clustering**
- Communicate technical insights in a business-relevant context

---

# Conclusion

This project illustrates how machine learning can support smarter marketing strategies in the banking sector.

By predicting which customers are most likely to subscribe to term deposits, marketing teams can allocate resources more efficiently and improve campaign outcomes. The addition of subscriber segmentation provides deeper insight into customer behavior and supports more targeted future campaigns.

For hiring managers and recruiters, this repository demonstrates my ability to:

- design practical machine learning solutions for real business problems  
- build structured data science workflows from data exploration to model evaluation  
- apply advanced modelling techniques while maintaining business interpretability  

Overall, the project reflects both technical proficiency and the ability to translate machine learning into actionable business insight.