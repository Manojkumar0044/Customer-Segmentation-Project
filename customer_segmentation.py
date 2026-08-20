# Customer Segmentation Project
# Method: K-Means Clustering

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Customer_Segmentation_Data.csv")

features = ["Age","Annual_Income","Spending_Score",
            "Purchase_Frequency","Average_Order_Value"]

X = StandardScaler().fit_transform(df[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
df["Cluster"] = kmeans.fit_predict(X)

score = silhouette_score(X, df["Cluster"])
print("Silhouette Score:", round(score, 3))

summary = df.groupby("Cluster")[features].mean()
print(summary)

df.to_csv("Customer_Segmentation_Output.csv", index=False)
