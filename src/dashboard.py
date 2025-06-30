import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

df = pd.read_csv('outputs/clustered_customers.csv')

st.title("Customer Segmentation Dashboard (K-Means)")

st.sidebar.header("Filter Customers")
selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(df['Cluster'].unique().tolist()) + ["All"])

if selected_cluster != "All":
    filtered_df = df[df['Cluster'] == selected_cluster]
else:
    filtered_df = df

st.subheader("Segmentation by Income vs Spending")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=80, ax=ax1)
ax1.set_title("Customer Clusters")
st.pyplot(fig1)

st.subheader("Number of Customers per Cluster")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x='Cluster', palette='Set2', ax=ax2)
ax2.set_title("Customer Count by Cluster")
st.pyplot(fig2)

st.subheader("Cluster Profiles Summary")
cluster_profiles = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(0).reset_index()
st.dataframe(cluster_profiles)

with st.expander("View Raw Data"):
    st.dataframe(df)