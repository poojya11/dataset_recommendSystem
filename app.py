from flask import Flask, render_template, request
import pickle
import pandas as pd
from difflib import get_close_matches

# Load model
with open("reco_model.pkl", "rb") as f:
    data = pickle.load(f)

knn = data["model"]
df = data["df"]
scaler = data["scaler"]
features_scaled = data["features_scaled"]

def get_best_match(name_input):
    names = df['Dataset_name'].tolist()
    match = get_close_matches(name_input, names, n=1, cutoff=0.4)
    return match[0] if match else None

def get_recommendations_by_name(name):
    try:
        index = df[df['Dataset_name'].str.lower() == name.lower()].index[0]
    except IndexError:
        return []

    distances, indices = knn.kneighbors([features_scaled[index]])
    recs = []
    for i in indices[0][1:]:  # skip input dataset
        recs.append({
            'name': df.iloc[i]['Dataset_name'],
            'link': df.iloc[i]['Dataset_link'],
            'upvotes': df.iloc[i]['Upvotes'],
            'usability': df.iloc[i]['Usability']
        })
    return recs

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error = None
    input_name = ""

    if request.method == 'POST':
        input_name = request.form['dataset_name']
        matched = get_best_match(input_name)
        if matched:
            recommendations = get_recommendations_by_name(matched)
            input_name = matched
        else:
            error = "No match found for your input."

    return render_template("index.html", recs=recommendations, input_name=input_name, error=error)

if __name__ == "__main__":
    app.run(debug=True)
