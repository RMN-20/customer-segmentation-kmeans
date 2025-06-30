# Customer Segmentation using K-Means Clustering

This project implements **K-Means clustering** for customer segmentation and provides a visual dashboard using **Streamlit**. It is designed to help understand customer behavior based on their **age**, **annual income**, and **spending score**.

## Objective

To segment customers into distinct groups using unsupervised learning, helping businesses make informed decisions about targeted marketing strategies.

## Project Structure

```
project/
│
├── data/
│   └── Dataset.csv                   # Input dataset
│
├── outputs/
│   └── clustered_customers.csv       # Segmented output with cluster labels
│
├── kmeans_segmentation.py           # Main script for clustering and profiling
├── dashboard.py                     # Streamlit dashboard for visualization
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/RMN-20/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Place your `Dataset.csv` file inside the `data/` folder. Make sure it contains at least the following columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

### 4. Run the Clustering Script

```bash
python kmeans_segmentation.py
```

This script:
- Preprocesses the dataset
- Applies K-Means clustering
- Saves the clustered results to `outputs/clustered_customers.csv`
- Displays cluster visualizations and interpretations

### 5. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open the local URL provided by Streamlit to explore customer segments interactively.

## Visualizations

- **Elbow Method** to determine optimal k
- **Scatter plots** of clusters
- **Cluster-wise customer counts**
- **Mean profiles of each cluster**

## Interpretation Logic

The model interprets each cluster based on patterns of income and spending, classifying them into types like:
- Impulsive buyers
- Careful spenders
- Target customers

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Streamlit

## Author

**Narmadha**  
Computer Science and Data Science Student  
[LinkedIn](https://www.linkedin.com/in/narmadha20/)

---

Feel free to fork or contribute. This project was built as part of a hands-on learning initiative in unsupervised machine learning.
