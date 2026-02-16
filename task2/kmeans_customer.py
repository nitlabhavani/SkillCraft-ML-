import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Annual Income (k$)": [15, 16, 17, 18, 19, 40, 42, 43, 44, 45, 60, 62, 63, 65, 68],
    "Spending Score (1-100)": [39, 81, 6, 77, 40, 50, 55, 60, 65, 70, 20, 25, 30, 35, 40]
}

df = pd.DataFrame(data)

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

print("\nClustered Customer Data:\n")
print(df)

plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], 
            c=df["Cluster"], cmap='viridis')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")

plt.show()
