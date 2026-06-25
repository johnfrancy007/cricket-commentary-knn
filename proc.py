import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. LOAD THE DATA
# ==========================================
# Update this with the actual path to your downloaded parquet file
file_path = "train-00000-of-00001.parquet"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_path}. Please place the file in your working directory.")

print("Loading dataset...")
df = pd.read_parquet(file_path)

print(f"Dataset successfully loaded! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns available:", df.columns.tolist())

# ==========================================
# 2. CLEAN & DEFINE TARGET
# ==========================================
# Drop rows where critical columns are missing
df = df.dropna(subset=['conversation', 'issue_complexity'])

# Clean text: lowercase and remove leading/trailing whitespaces
df['clean_conversation'] = df['conversation'].str.lower().str.strip()

# Initialize LabelEncoder for our target column (e.g., predicting 'issue_complexity')
target_encoder = LabelEncoder()
df['target'] = target_encoder.fit_transform(df['issue_complexity'])

print("\nTarget classes identified:")
for index, class_label in enumerate(target_encoder.classes_):
    print(f" Class {index}: {class_label}")

# ==========================================
# 3. TEXT VECTORIZATION (TF-IDF)
# ==========================================
print("\nVectorizing text conversations using TF-IDF...")
# We limit to max_features=500 because high-dimensional sparse text vectors 
# can degrade KNN performance due to the 'curse of dimensionality'.
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_text_features = tfidf.fit_transform(df['clean_conversation']).toarray()

# Convert text features to a DataFrame
X_text_df = pd.DataFrame(X_text_features, columns=[f"tfidf_{i}" for i in range(X_text_features.shape[1])])

# ==========================================
# 4. ENCODE CATEGORICAL FEATURES (Optional Extra Context)
# ==========================================
# Let's one-hot encode categorical metadata columns to help the KNN model
categorical_cols = ['product_category', 'customer_sentiment']
X_categorical_df = pd.get_dummies(df[categorical_cols], drop_first=True).reset_index(drop=True)

# Combine textual features and categorical metadata features
X = pd.concat([X_text_df, X_categorical_df], axis=1)
y = df['target'].values

print(f"Final Feature Matrix Shape: {X.shape}")

# ==========================================
# 5. SPLIT AND SCALE DATA
# ==========================================
# Split into 80% Training and 20% Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# CRITICAL FOR KNN: Normalize features so distances are evaluated equally
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. TRAIN AND EVALUATE KNN MODEL
# ==========================================
print("\nTraining KNN Classifier...")
# k=5 is a standard starting point
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = knn.predict(X_test_scaled)

# Evaluate Results
accuracy = accuracy_score(y_test, y_pred)
print(f"\n==========================================")
print(f"KNN MODEL PERFORMANCE")
print(f"==========================================")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))