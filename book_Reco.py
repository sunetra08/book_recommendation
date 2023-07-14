import difflib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv('Books.csv')
rating = pd.read_csv('Ratings.csv')
user = pd.read_csv('Users.csv')

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
    print("No data available for recommendations.")
    exit()

p = final.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
p.fillna(0, inplace=True)

cosarr = cosine_similarity(p)

bookname = input("Enter a book name: ")

if bookname in p.index:
    i = list(p.index).index(bookname)
    similar = sorted(list(enumerate(cosarr[i])), key=lambda x: x[1], reverse=True)[1:6]
    for j in similar:
        print(p.index[j[0]])
else:
    matches = difflib.get_close_matches(bookname, p.index, cutoff=0.5)
    if len(matches) > 0:
        print("Did you mean one of these?")
        for match in matches:
            print(match)
    else:
        print("Book not found in recommendations.")