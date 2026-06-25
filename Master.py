import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Define Asset Paths
PARQUET_FILE = "train-00000-of-00001.parquet"
CSV_FILE = "processed_customer_support_data.csv"
MODEL_PATH = "knn_model.pkl"
SCALER_PATH = "scaler.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
COLUMNS_PATH = "model_columns.pkl"

# =====================================================================
# STAGE 1: DATA PROCESSING & CLEANING
# =====================================================================
def run_stage_1_processing():
    if not os.path.exists(PARQUET_FILE):
        raise FileNotFoundError(f"❌ Error: Could not find '{PARQUET_FILE}'. Make sure it is in this folder!")

    print("\n--- STAGE 1: PROCESSING RAW PARQUET DATA ---")
    df = pd.read_parquet(PARQUET_FILE)

    # Clean text and handle missing values
    df = df.dropna(subset=['conversation', 'issue_complexity'])
    df['clean_conversation'] = df['conversation'].str.lower().str.strip()

    # Encode target label (issue_complexity)
    target_encoder = LabelEncoder()
    df['target_label'] = target_encoder.fit_transform(df['issue_complexity'])

    # Convert text to numerical features (TF-IDF Vectorization)
    print("Vectorizing conversation text...")
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_features = tfidf.fit_transform(df['clean_conversation'].fillna('')).toarray()

    # Turn TF-IDF array into a DataFrame
    tfidf_df = pd.DataFrame(tfidf_features, columns=[f"word_feature_{i}" for i in range(tfidf_features.shape[1])])

    # Standardize string values to title case for reliable user input mapping later
    df['product_category'] = df['product_category'].fillna('Unknown').str.strip().str.title()
    df['customer_sentiment'] = df['customer_sentiment'].fillna('Neutral').str.strip().str.title()

    # One-hot encode product categories and sentiments
    categorical_df = pd.get_dummies(df[['product_category', 'customer_sentiment']], drop_first=True).reset_index(drop=True)

    # Combine text and categorical data together
    processed_df = pd.concat([
        df[['clean_conversation', 'issue_complexity', 'target_label']].reset_index(drop=True),
        categorical_df,
        tfidf_df
    ], axis=1)

    # Save to CSV
    processed_df.to_csv(CSV_FILE, index=False)
    print(f"✅ Success! Generated '{CSV_FILE}'\n")

# =====================================================================
# STAGE 2: KNN MODEL TRAINING & ASSET EXPORT
# =====================================================================
def run_stage_2_training():
    print("--- STAGE 2: TRAINING THE KNN MODEL ---")
    df = pd.read_csv(CSV_FILE)

    # Separate features (X) and target (y)
    X = df.drop(columns=['clean_conversation', 'issue_complexity', 'target_label'])
    y = df['target_label']

    # Split into sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Standardizing features (Scaling)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    # Export all 4 required deployment assets
    print("Saving model and feature deployment assets...")
    joblib.dump(list(X.columns), COLUMNS_PATH)
    
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    # Fit on original text to ensure identical feature vocabulary sizes
    raw_df = pd.read_parquet(PARQUET_FILE)
    tfidf.fit(raw_df['conversation'].dropna().str.lower())
    joblib.dump(tfidf, TFIDF_PATH) 

    joblib.dump(knn, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Success! All deployment assets built.\n")

# =====================================================================
# STAGE 3: LIVE INTERACTIVE PREDICTION LOOP
# =====================================================================
def process_realtime_input(conversation_text, sentiment, product_cat, tfidf, scaler, model_columns):
    clean_text = str(conversation_text).lower().strip()
    text_features = tfidf.transform([clean_text]).toarray()
    X_text_df = pd.DataFrame(text_features, columns=[f"word_feature_{i}" for i in range(300)])
    
    # Safely convert to python string and call title case
    clean_product_cat = str(product_cat).strip().title()
    clean_sentiment = str(sentiment).strip().title()

    raw_cat_df = pd.DataFrame([{
        'product_category': clean_product_cat,
        'customer_sentiment': clean_sentiment
    }])
    X_categorical_df = pd.get_dummies(raw_cat_df)
    
    X_combined = pd.concat([X_text_df, X_categorical_df], axis=1)
    X_final = X_combined.reindex(columns=model_columns, fill_value=0)
    return scaler.transform(X_final)

def run_stage_3_cli():
    print("--- STAGE 3: STARTING LIVE PREDICTION INTERFACE ---")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    
    class_mapping = {0: "Low Complexity", 1: "Medium Complexity", 2: "High Complexity"}
    
    print("==========================================================")
    print("       🛒 RETAIL CUSTOMER SUPPORT - KNN PREDICTION CLI     ")
    print("==========================================================")
    print("Type 'exit' at any prompt to quit the application.\n")
    
    while True:
        print("-" * 58)
        user_chat = input("📝 Enter Customer Chat Segment:\n> ")
        if user_chat.strip().lower() == 'exit': 
            break
            
        print("\nEnter Sentiment (Negative, Neutral, Positive):")
        sentiment = input("> ").strip()
        if sentiment.lower() == 'exit': 
            break
        
        print("\nEnter Category (Warranty, Shipping, Refund, Login And Account, Cancellations And Returns, Order Shopping, Payment Options):")
        product_cat = input("> ").strip()
        if product_cat.lower() == 'exit': 
            break
        
        try:
            processed_input = process_realtime_input(user_chat, sentiment, product_cat, tfidf, scaler, model_columns)
            pred_class = model.predict(processed_input)[0]
            confidence_dist = model.predict_proba(processed_input)[0]
            
            result_label = class_mapping.get(pred_class, f"Unknown ({pred_class})")
            print("\n==========================================================")
            print(f"🔮 PREDICTED COMPLEXITY: {result_label.upper()}")
            print(f"➡ Nearest Neighbor Consensus: {np.max(confidence_dist)*100:.1f}%")
            print("==========================================================\n")
        except Exception as e:
            print(f"❌ Real-time processing error: {str(e)}")
            print("Tip: Make sure you use one of the provided categories listed above.\n")

    print("\n👋 Exiting Master Application. Goodbye!")

if __name__ == "__main__":
    try:
        run_stage_1_processing()
        run_stage_2_training()
        run_stage_3_cli()
    except Exception as error:
        print(f"\n❌ Pipeline failed: {str(error)}")
