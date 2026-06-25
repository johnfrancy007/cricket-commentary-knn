import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA

# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================
file_path = "processed_customer_support_data.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find '{file_path}'. Please make sure you generated the CSV first.")

print("Loading preprocessed dataset...")
df = pd.read_csv(file_path)

# Separate features (X) and target label (y)
# Drop the original string tracking columns, leaving only numerical features
X = df.drop(columns=['clean_conversation', 'issue_complexity', 'target_label'])
y = df['target_label']

# Dynamically map the target encoded numbers back to their original names for plotting
target_names = df.groupby('target_label')['issue_complexity'].first().sort_index().tolist()

# Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. FEATURE SCALING & KNN TRAINING
# ==========================================
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training KNN Classifier (k=5)...")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# ==========================================
# 3. SAVE MODEL COMPONENTS
# ==========================================
print("\nSaving model assets for deployment...")
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("-> Saved 'knn_model.pkl' and 'scaler.pkl'")

# ==========================================
# 4. EVALUATE & VISUALIZE RESULTS
# ==========================================
print("\nEvaluating model metrics...")
y_pred = knn.predict(X_test_scaled)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Create visual directory to store plots
os.makedirs("visualizations", exist_ok=True)

# Plot 1: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('KNN Model - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png')
plt.close()
print("-> Saved Confusion Matrix plot to 'visualizations/confusion_matrix.png'")

# Plot 2: 2D Decision Boundary Projection (PCA)
print("Generating 2D visual projection of clusters...")
# Since we have 300+ features, we project down to 2 dimensions to visually see the separation
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', alpha=0.7, edgecolors='k')
plt.legend(handles=scatter.legend_elements()[0], labels=target_names, title="Classes")
plt.title('2D PCA Projection of KNN Test Data Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('visualizations/data_clusters_2d.png')
plt.close()
print("-> Saved Data Clusters plot to 'visualizations/data_clusters_2d.png'")

print("\n🎉 Pipeline execution complete!")