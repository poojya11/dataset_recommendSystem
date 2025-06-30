from flask import Flask, render_template, request
import pickle
import os

# Load model, vectorizer, and dataframe
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("dataframe.pkl", "rb") as f:
    df = pickle.load(f)

# Recommendation function
def recommend(query, top_n=5):
    query_vec = vectorizer.transform([query.lower()])
    distances, indices = knn_model.kneighbors(query_vec, n_neighbors=top_n)
    results = df.iloc[indices[0]].copy()
    results['Similarity'] = (1 - distances[0]).round(3)
    
    # Return list of dictionaries for rendering
    return results[['Dataset_name', 'Author_name', 'Dataset_link', 'Similarity']].to_dict(orient='records')

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    query = ''
    error = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            recommendations = recommend(query)
        else:
            error = "Please enter a search term."

    return render_template('index.html', recs=recommendations, query=query, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
