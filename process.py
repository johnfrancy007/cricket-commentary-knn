import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load the original file
file_path = "train-00000-of-00001.parquet"
print("Reading original parquet file...")
df = pd.read_parquet(file_path)

# 2. Clean text and handle missing values
df = df.dropna(subset=['conversation', 'issue_complexity'])
df['clean_conversation'] = df['conversation'].str.lower().str.strip()

# 3. Encode the target (issue_complexity)
target_encoder = LabelEncoder()
df['target_label'] = target_encoder.fit_transform(df['issue_complexity'])

# 4. Convert text to numerical features (TF-IDF)
print("Processing and vectorizing text...")
tfidf = TfidfVectorizer(max_features=300, stop_words='english')
tfidf_features = tfidf.fit_transform(df['clean_conversation']).toarray()

# Turn TF-IDF array into a clean DataFrame
tfidf_df = pd.DataFrame(tfidf_features, columns=[f"word_feature_{i}" for i in range(tfidf_features.shape[1])])

# 5. One-hot encode other categorical categories
categorical_df = pd.get_dummies(df[['product_category', 'customer_sentiment']], drop_first=True).reset_index(drop=True)

# 6. Combine everything back together into a single dataset
processed_df = pd.concat([
    df[['clean_conversation', 'issue_complexity', 'target_label']].reset_index(drop=True),
    categorical_df,
    tfidf_df
], axis=1)

# 7. EXPORT TO CSV
output_filename = "processed_customer_support_data.csv"
processed_df.to_csv(output_filename, index=False)

print(f"\n🎉 Success! Your processed file has been created and saved as: '{output_filename}'")
print(f"Total rows: {processed_df.shape[0]} | Total features: {processed_df.shape[1]}")