import pandas as pd
import numpy as np
import ast
import json
import nltk
from nltk.stem.porter import PorterStemmer

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
# print(credits.head(1))

movies = movies.merge(credits,on='title')

#genres,id,keywords,title,overview,
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#remove missing data
movies.dropna(inplace=True)
# print(movies.head(1))

#check duplication 
movies.duplicated().sum()


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres']  = movies['genres'].apply(convert)
movies['keywords']  = movies['keywords'].apply(convert)

# print(movies.head(1))

# only 3 name of actors
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
        else: 
            break
        counter+=1
    return L

# print(movies['cast'].apply(convert3))
movies['cast'] = movies['cast'].apply(convert3)


#find director
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
# print(movies['crew'])

#overview
movies['overview'] = movies['overview'].apply(lambda x:x.split())
# print(movies['overview'])

#remove spaces

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# print(movies['genres'])
# print(movies['keywords'])
# print(movies['cast'])
# print(movies['crew'])

# new column name is tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# print(movies['tags'])

# new data frame
new_df = movies[['movie_id','title','tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
# print(new_df['tags'][0])

# convert all into lower case
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
# print(new_df.head())

#remove similar words
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
# convert text to vector (bag of words)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()
# print(vectors[0])
# print(cv.get_feature_names_out()[:50])


#similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# print(similarity[0])

#sorted 
# sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

#recommend
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')


################################

# build dictionary: title -> list of top-5 recommended titles
recs = {}
for idx in range(len(new_df)):   # iterate over row positions (0…N-1)
    distances = similarity[idx]
    movie_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]   # skip itself

    recs[new_df.iloc[idx]['title']] = [
        new_df.iloc[i[0]]['title'] for i in movie_list
    ]

# write to file
with open("movies_recs.json", "w", encoding="utf-8") as f:
    json.dump(recs, f, ensure_ascii=False, indent=2)

print(f"movies_recs.json written; contains {len(recs)} movies")














