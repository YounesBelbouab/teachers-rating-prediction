from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()

# Charger le modèle et le vectorizer
model = joblib.load("modele_stars.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

@app.post("/api/predic")
async def predict(request: Request):
    data = await request.json()

    # Fusionner les champs texte
    text = ' '.join([
        data.get("description", ""),
        ' '.join([d["title"] for d in data.get("diplomas", [])]),
        ' '.join([d["level"] for d in data.get("diplomas", [])]),
        ' '.join([e["company"] for e in data.get("experiences", [])]),
        ' '.join([e["city"] for e in data.get("experiences", [])])
    ])

    # Créer un DataFrame et transformer
    df = pd.DataFrame([{"text_combined": text}])
    X = tfidf.transform(df["text_combined"])

    # Prédire
    prediction = model.predict(X)[0]
    return {"predicted_numberOfStars": round(float(prediction), 2)}
