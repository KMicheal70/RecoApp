import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
df = pd.read_csv('ratings_Electronics (1).csv', names=['userId', 'productId', 'rating', 'timestamp'])

df = df.iloc[:5000,0:]
# Data Cleaning & EDA
df['userId'] = df['userId'].astype(str)
df['productId'] = df['productId'].astype(str)
df['rating'] = df['rating'].astype(float)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Filter out ratings outside 1-5 range
df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
df = df.drop_duplicates()

# Feature Engineering
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day

# Number of ratings per user
user_ratings_count = df.groupby('userId').size()
df['user_ratings_count'] = df['userId'].map(user_ratings_count)

# Number of ratings per product
product_ratings_count = df.groupby('productId').size()
df['product_ratings_count'] = df['productId'].map(product_ratings_count)

# Save cleaned dataset
df.to_csv('cleaned_amazon_reviews_with_features.csv', index=False)

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

# Train the SVD model
svd = SVD()
svd.fit(trainset)

# Make predictions on the test set and evaluate
predictions = svd.test(testset)
print("SVD RMSE:", accuracy.rmse(predictions))

# Content-Based Filtering
user_product_matrix = df.pivot(index='userId', columns='productId', values='rating').fillna(0)
product_similarity = cosine_similarity(user_product_matrix.T)
product_similarity_df = pd.DataFrame(product_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)
