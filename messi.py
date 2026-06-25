import os
import sys
import joblib
import pandas as pd
import numpy as np

# =====================================================================
# 1. INITIALIZATION & ASSET LOADING
# =====================================================================
MODEL_PATH = "knn_model.pkl"
SCALER_PATH = "scaler.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
COLUMNS_PATH = "model_columns.pkl"

def load_assets():
    """Loads saved model, scaler, and aligns the textual vocabulary."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, TFIDF_PATH, COLUMNS_PATH]):
        print("❌ Error: Missing deployment assets ('knn_model.pkl', 'scaler.pkl', 'tfidf_vectorizer.pkl', or 'model_columns.pkl').")
        print("Please run your updated 'train_knn.py' script first to generate these files.")
        sys.exit(1)
        
    print("⏳ Loading model components...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    
    # Hardcoded classification mappings based on dataset classes
    class_mapping = {0: "Low Complexity", 1: "Medium Complexity", 2: "High Complexity"}
    
    print("✅ System assets successfully initialized!\n")
    return model, scaler, tfidf, model_columns, class_mapping

# =====================================================================
# 2. INPUT PROCESSING ENGINE
# =====================================================================
def process_realtime_input(conversation_text, sentiment, product_cat, tfidf, scaler, model_columns):
    """Transforms raw user CLI inputs into a matching numerical feature vector."""
    
    # A. Handle the text vectorization
    clean_text = str(conversation_text).lower().strip()
    text_features = tfidf.transform([clean_text]).toarray()
    X_text_df = pd.DataFrame(text_features, columns=[f"word_feature_{i}" for i in range(300)])
    
    # B. Handle One-Hot Encoded Categorical Features dynamically
    raw_cat_df = pd.DataFrame([{
        'product_category': product_cat,
        'customer_sentiment': sentiment
    }])
    X_categorical_df = pd.get_dummies(raw_cat_df)
    
    # C. Merge and dynamically fix column shape mismatches
    X_combined = pd.concat([X_text_df, X_categorical_df], axis=1)
    
    # This aligns the columns perfectly to what the model expects, inserting 0 for unselected options
    X_final = X_combined.reindex(columns=model_columns, fill_value=0)
    
    # D. Scale features uniformly
    return scaler.transform(X_final)

# =====================================================================
# 3. INTERACTIVE TERMINAL LOOP
# =====================================================================
def run_cli():
    model, scaler, tfidf, model_columns, class_mapping = load_assets()
    
    print("==========================================================")
    print("       🛒 RETAIL CUSTOMER SUPPORT - KNN PREDICTION CLI     ")
    print("==========================================================")
    print("Type 'exit' at any prompt to quit the interface.\n")
    
    while True:
        print("-" * 58)
        # Input 1: The Chat Conversation Text
        user_chat = input("📝 Enter Customer Chat Segment:\n> ")
        if user_chat.strip().lower() == 'exit':
            break
            
        # Input 2: Sentiment Select
        print("\nEnter Sentiment (Negative, Neutral, Positive):")
        sentiment = input("> ").strip()
        if sentiment.lower() == 'exit': break
        
        # Input 3: Product Category
        print("\nEnter Category (e.g., Warranty, Shipping, Refund, Login and Account):")
        product_cat = input("> ").strip()
        if product_cat.lower() == 'exit': break
        
        print("\n⏳ Running pipeline distance calculations...")
        
        try:
            # Vectorize input data
            processed_input = process_realtime_input(user_chat, sentiment, product_cat, tfidf, scaler, model_columns)
            
            # Predict
            pred_class = model.predict(processed_input)[0]
            confidence_dist = model.predict_proba(processed_input)[0] # Grab neighborhood votes ratio
            
            # Display Results
            result_label = class_mapping.get(pred_class, f"Unknown Class ({pred_class})")
            print("\n==========================================================")
            print("🔮 INFERENCE RESULT:")
            print(f"➡ Predicted Issue Complexity: {result_label.upper()}")
            print(f"➡ Nearest Neighbor Consensus: {np.max(confidence_dist)*100:.1f}%")
            print("==========================================================\n")
            
        except Exception as e:
            print(f"❌ Error compiling pipeline input: {str(e)}")
            print("Please ensure your training pipeline executed correctly.\n")

    print("\n👋 Exiting Real-time CLI Interface. Goodbye!")

if __name__ == "__main__":
    run_cli()