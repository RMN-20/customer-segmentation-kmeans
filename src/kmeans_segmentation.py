import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/Dataset.csv')
print(df.head())
print(df.info())
print(df.describe())

df = df.drop('CustomerID', axis=1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender']) 

print("\nEncoded Data:\n", df.head())

X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled features shape:", X_scaled.shape)

inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels

print("\nClustered Data:\n", df.head())

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, 
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    palette='Set1', 
    s=100
)
plt.title('Customer Segments based on Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

print("\nCluster Centers (scaled):")
print(kmeans.cluster_centers_)

original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_profiles = pd.DataFrame(original_centers, columns=X.columns)
cluster_profiles['Cluster'] = cluster_profiles.index

def interpret(row):
    age, income, score = row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)']
    if income < 40 and score > 60:
        return "Young, low income, high spending (Impulsive buyers)"
    elif income > 75 and score < 30:
        return "Older, high income, low spending (Careful spenders)"
    elif income > 90 and score > 80:
        return "Young, high income, high spending (Target customers 💰)"
    elif 40 <= income <= 60 and 40 <= score <= 60:
        return "Average group"
    elif income < 25 and score < 30:
        return "Young, low income, low spending"
    else:
        return "Others"

cluster_profiles['Interpretation'] = cluster_profiles.apply(interpret, axis=1)

summary = cluster_profiles[['Cluster', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Interpretation']]
summary = summary.round(0).astype({'Age': 'int', 'Annual Income (k$)': 'int', 'Spending Score (1-100)': 'int'})

print("\nCluster Profiles (Original Scale):\n")
print(summary.to_string(index=False))

os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/clustered_customers.csv', index=False)