from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
# Set matplotlib theme to bmh
mpl.style.use('bmh')
# Setwd to file directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# set fig size
plt.rcParams['figure.figsize'] = [10, 8]

# 1. Load and preprocess data

import pandas as pd
df = pd.read_csv('../tables/Hec_50.csv')
# Create U_G color index
df['U_G'] = df['U_v2'] - df['G_v2']

df = df[["logSFR_total", "logM_total", "Metal_v2","Activity_class", "U_G", "logsSFR"]].dropna()
print(df.describe())

inertias = []

r_seed = 54  # Set random seed for reproducibility

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=r_seed)
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png', bbox_inches='tight', dpi=800)
plt.close()

kmeans = KMeans(n_clusters=4, random_state=r_seed)  # Set n_clusters based on elbow method
clusters = kmeans.fit_predict(df)

# Add cluster labels to original data
df['cluster'] = clusters
# plot the clusters in the original space without PCA
plt.scatter(df['logSFR_total'], df['logM_total'], c=df['cluster'])
plt.colorbar(label='Cluster')
plt.xlabel('logSFR_total')
plt.ylabel('logM_total')
plt.title('Galaxy Clusters in Original Space without PCA')
plt.savefig('clusters_original_space.png', bbox_inches='tight', dpi=800)
plt.close()


# histogram of clusters describe()
df.groupby('cluster').size().plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Number of galaxies')
plt.title('Number of Galaxies per Cluster')
plt.savefig('clusters_histogram.png', bbox_inches='tight', dpi=800)
plt.close()

# describe the clusters
print(df.groupby('cluster').describe())
# Describe the clusters per column
for col in df.columns[:-1]:  # Exclude 'cluster' column
    print(f"Cluster {col} statistics:")
    print(df.groupby('cluster')[col].describe())
    print("\n")

# violin plot of clusters based on the above describe()
import seaborn as sns

for col in df.columns[:-1]:  # Exclude 'cluster' column
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=col, data=df)
    plt.title(f'Violin plot of {col} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(col)
    plt.savefig(f'violin_plot_{col}.png', bbox_inches='tight', dpi=800)
    plt.close()

