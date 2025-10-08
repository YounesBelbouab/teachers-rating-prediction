from fastapi import FastAPI, Request
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import hstack

app = FastAPI()

# Charger le modèle et le vectorizer
model = joblib.load("modele_stars(1).pkl")
tfidf = joblib.load("tfidf_vectorizer(1).pkl")

@app.post("/api/predict")
async def predict(request: Request):
    data = await request.json()
    user_id = data.get("id", "temp")

    # === 🔹 Construction du mini DataFrame simulé ===
    rows = []

    # Description principale
    rows.append({
        "id": user_id,
        "source": "main",
        "description": data.get("description", ""),
        "numberOfStars": np.nan
    })

    # Diplômes
    for d in data.get("diplomas", []):
        rows.append({
            "id": user_id,
            "source": "diploma",
            "title": d.get("title", ""),
            "level": d.get("level", ""),
            "institution": d.get("institution", ""),
            "numberOfStars": np.nan
        })

    # Expériences
    for e in data.get("experiences", []):
        rows.append({
            "id": user_id,
            "source": "experience",
            "company": e.get("company", ""),
            "city": e.get("city", ""),
            "duration": e.get("duration", ""),
            "numberOfStars": np.nan
        })

    # Cours passés
    for c in data.get("pastCourses", []):
        rows.append({
            "id": user_id,
            "source": "pastcourse",
            "course_code": c.get("course_code", ""),
            "course_level": c.get("course_level", ""),
            "numberOfStars": c.get("numberOfStars", np.nan)
        })

    df_all = pd.DataFrame(rows)

    # === 🔹 Calcul des colonnes d’agrégation ===
    df_all["numberOfStars"] = pd.to_numeric(df_all.get("numberOfStars", np.nan), errors="coerce")

    nombre_exp = df_all[df_all["source"] == "experience"].groupby("id").size().rename("nombre_experiences")
    nb_cours = df_all[df_all["source"] == "pastcourse"].groupby("id").size().rename("nb_cours")
    moyenne_notes = df_all[df_all["source"] == "pastcourse"].groupby("id")["numberOfStars"].mean().rename("moyenne_notes")

    df_features = pd.concat([nombre_exp, nb_cours, moyenne_notes], axis=1).fillna(0)

    coef_exp, coef_cours, coef_stars = 0.3, 0.3, 0.4
    max_exp = max(df_features["nombre_experiences"].max(), 1)
    max_cours = max(df_features["nb_cours"].max(), 1)

    df_features["score_reputation"] = (
        (coef_exp * (df_features["nombre_experiences"] / max_exp)) +
        (coef_cours * (df_features["nb_cours"] / max_cours)) +
        (coef_stars * (df_features["moyenne_notes"] / 5))
    ) * 100

    df_features["score_reputation"] = df_features["score_reputation"].round(2)

    # === 🔹 Fusion avec les données principales ===
    df_all = df_all.merge(df_features, on="id", how="left")

    # === 🔹 Préparation du texte combiné ===
    text_cols = ['firstname', 'lastname', 'city', 'description', 'title',
                 'company', 'level', 'institution', 'course_level']
    df_all[text_cols] = df_all[text_cols].fillna('')
    text_combined = df_all[text_cols].agg(' '.join, axis=1).iloc[0]
    X_text = tfidf.transform([text_combined])

    # === 🔹 Variables numériques ===
    num_cols = ['duration', 'nombre_experiences', 'nb_cours', 'moyenne_notes', 'score_reputation']
    df_all[num_cols] = df_all[num_cols].fillna(0)
    X_num = sparse.csr_matrix(df_all[num_cols].iloc[0].values.reshape(1, -1))

    # === 🔹 Combinaison finale ===
    X_final = hstack([X_text, X_num])

    # === 🔹 Prédiction ===
    prediction = model.predict(X_final)[0]
    return {"predicted_numberOfStars": round(float(prediction), 2)}
