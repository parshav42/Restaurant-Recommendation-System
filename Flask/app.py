import sys
import csv
csv.field_size_limit(sys.maxsize)

import pandas as pd
import string
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# ---------- LOAD DATA ----------
df = pd.read_csv("../Dataset/zomato.csv", engine='python', on_bad_lines='skip')

# ---------- CLEAN ----------
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df = df[df['rate'] != 'NEW']
df = df[df['rate'] != '-']

df['rate'] = df['rate'].str.replace('/5', '')
df['rate'] = df['rate'].astype(float)

# 🔥 reduce size early
df = df.sample(5000).reset_index(drop=True)

# ---------- TEXT CLEAN ----------
df['reviews_list'] = df['reviews_list'].str.lower()

def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return ""

df['reviews_list'] = df['reviews_list'].apply(remove_punctuation)

# ---------- MODEL ----------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['reviews_list'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# ---------- FUNCTION ----------
def recommend_model(name):
    name = name.lower().strip()

    matches = [i for i in indices.index if name in i.lower()]

    if len(matches) == 0:
        return ["No restaurant found"]

    idx = indices[matches[0]]

    # ✅ FIX HERE
    sim_scores = list(enumerate(cosine_sim[idx].flatten()))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    restaurant_indices = [i[0] for i in sim_scores]

    # safe check
    restaurant_indices = [i for i in restaurant_indices if i < len(df)]

    result = df['name'].iloc[restaurant_indices].tolist()

 # remove duplicates
    result = list(dict.fromkeys(result))

    return result

# ---------- FLASK ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        name = request.form['restaurant']
        result = recommend_model(name)
        return render_template('result.html', restaurants=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)