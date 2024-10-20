from flask import Flask, render_template, request, redirect, session
import pandas as pd
from surprise import Reader, Dataset, SVD
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load your cleaned dataset
df = pd.read_csv('cleaned_amazon_reviews_with_features.csv')


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train SVD model
svd = SVD()
svd.fit(trainset)

def get_user_recommendations(user_id, num_recommendations=5):
    all_product_ids = df['productId'].unique()
    recommendations = []

    for product_id in all_product_ids:
        if not df[(df['userId'] == user_id) & (df['productId'] == product_id)].empty:
            continue
        predicted_rating = svd.predict(user_id, product_id).est
        recommendations.append((product_id, predicted_rating))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:num_recommendations]

@app.route('/')
def index():
    return render_template('sign-in.html')

@app.route('/sign_in', methods=['POST'])
def sign_in():
    user_id = request.form['user_id']
    
    # Validate user ID (you can add a proper validation mechanism here)
    if user_id:
        session['user_id'] = user_id  # Store user_id in session
        return redirect('/recommendations')
    
    return redirect('/')  # Redirect back to sign-in if user_id is invalid

@app.route('/recommendations')
def recommendations():
    user_id = session.get('user_id')
    recommendations = []
    if user_id:
        recommendations = get_user_recommendations(user_id)
    return render_template('recommendations.html', recommendations=recommendations, user_id=user_id)

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    user_id = session.get('user_id')
    product_id = request.form['product_id']
    rating = request.form['rating']
    
    # Get the current timestamp
    now = datetime.now()
    
    # Create a new DataFrame for the new rating
    new_rating = pd.DataFrame({
        'userId': [user_id],
        'productId': [product_id],
        'rating': [rating],
        'timestamp': [now],
        'year': [now.year],
        'month': [now.month],
        'day': [now.day]
    })

    # Append the new rating to the DataFrame
    global df
    df = pd.concat([df, new_rating], ignore_index=True)
    
    # Save the updated DataFrame to CSV
    df.to_csv('cleaned_amazon_reviews_with_features.csv', index=False)

    # Retrain the SVD model with the updated DataFrame
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    return redirect('/recommendations')

@app.route('/rate_product/<int:product_id>', methods=['GET'])
def rate_product(product_id):
    user_id = session.get('user_id')
    if user_id:
        return render_template('rate_product.html', product_id=product_id, user_id=user_id)
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port, debug=True)
