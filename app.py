import difflib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

app = Flask(__name__)

def load_data():
    books = pd.read_csv('Books.csv')
    rating = pd.read_csv('Ratings.csv')
    user = pd.read_csv('Users.csv')
    return books, rating, user

def recommend_books(bookname):
    books, rating, user = load_data()

    ratings_with_name = rating.merge(books, on='ISBN')

    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().astype('float').reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings'] >= 350].sort_values('avg_rating', ascending=False).head(50)
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 100
    read = x[x].index
    filtered = ratings_with_name[ratings_with_name['User-ID'].isin(read)]

    y = filtered.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous = y[y].index
    final = filtered[filtered['Book-Title'].isin(famous)]

    if final.empty:
        return []

    p = final.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    p.fillna(0, inplace=True)

    cosarr = cosine_similarity(p)

    if bookname in p.index:
        i = np.where(p.index == bookname)[0][0]
        similar = sorted(list(enumerate(cosarr[i])), key=lambda x: x[1], reverse=True)[1:6]
        recommendations = [p.index[i[0]] for i in similar]
        return recommendations

    return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    bookname = request.form['bookname']
    recommended_books = recommend_books(bookname)
    return render_template('result.html', books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)
