from fastapi import FastAPI, Request
import joblib
import pandas as pd
import numpy as np
import json
import re
import unicodedata
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from catboost import CatBoostRegressor
import catboost

app = FastAPI()

model = joblib.load("best_cat.pkl")
tfidf = joblib.load("tfidf_vectorizer_xgb.pkl")


def nettoyer_texte(texte):
    if not isinstance(texte, str):
        return "Inconnu"
    texte = texte.strip().lower()
    texte = ''.join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')
    texte = re.sub(r'[^a-z0-9\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte if texte else "inconnu"


def convert_duration_to_months(text):
    if pd.isna(text) or str(text).strip() == "" or text == "Inconnu":
        return np.nan
    text = str(text).lower().strip()
    years = re.search(r"(\d+)\s*an", text)
    months = re.search(r"(\d+)\s*mois", text)
    total = 0
    if years:
        total += int(years.group(1)) * 12
    if "demi" in text and not months:
        total += 6
    if months:
        total += int(months.group(1))
    if total > 0:
        return int(total)
    match = re.findall(r"\d{4}", text)
    if match:
        start_year = int(match[0])
        end_year = None
        if len(match) > 1:
            end_year = int(match[1])
        elif "présent" in text or "present" in text:
            end_year = datetime.now().year
        if end_year:
            return int((end_year - start_year) * 12)
    date_pattern = re.findall(r"(\d{2}/\d{4})", text)
    if date_pattern:
        try:
            start_date = datetime.strptime(date_pattern[0], "%m/%Y")
            if len(date_pattern) > 1:
                end_date = datetime.strptime(date_pattern[1], "%m/%Y")
            elif "présent" in text or "present" in text:
                end_date = datetime.now()
            else:
                return np.nan
            diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            return max(diff, 1)
        except:
            pass
    if re.fullmatch(r"\d{4}", text):
        return 12
    return np.nan


def rename_year_columns(df):
    new_columns = {}
    for col in df.columns:
        if 'year' in col.lower():
            new_columns[col] = col.lower().replace('year', 'years')
    return df.rename(columns=new_columns)


def calculate_duration_in_months(df):
    # Ensure columns exist before processing
    if "start_date" not in df.columns:
        df["start_date"] = "Inconnu"
    if "end_date" not in df.columns:
        df["end_date"] = "Inconnu"

    def safe_parse_date(date_str):
        if pd.isna(date_str) or str(date_str).strip() == "" or str(date_str) == "Inconnu":
            return None
        text = str(date_str).lower().strip()
        if "present" in text or "présent" in text:
            return datetime.now()
        try:
            return parser.parse(text, fuzzy=True, dayfirst=True)
        except Exception:
            return None

    df["start_date_parsed"] = df["start_date"].apply(safe_parse_date)
    df["end_date_parsed"] = df["end_date"].apply(safe_parse_date)

    def compute_months(row):
        if pd.isna(row["start_date_parsed"]) or pd.isna(row["end_date_parsed"]):
            return np.nan
        diff = relativedelta(row["end_date_parsed"], row["start_date_parsed"])
        total_months = diff.years * 12 + diff.months
        return max(total_months, 0)

    df["duration"] = df.apply(compute_months, axis=1)
    df = df.drop(columns=["start_date", "end_date", "start_date_parsed", "end_date_parsed"])
    df["duration"] = df["duration"].fillna(0).astype(int)
    return df


def remplacer_nan_par_inconnu(df, colonne=None):
    if colonne is None or colonne not in df.columns:
        return df
    if pd.api.types.is_numeric_dtype(df[colonne]):
        df[colonne] = df[colonne].fillna(0)
    else:
        df[colonne] = df[colonne].fillna("Inconnu")
        df[colonne] = df[colonne].replace("", "Inconnu")
    return df


@app.post("/api/predict")
async def predict(request: Request):
    data = await request.json()
    user_id = data.get("id", "temp")

    # === Nettoyage de base des valeurs ===
    clean_data = {k: (v if v is not None else "Inconnu") for k, v in data.items()}
    firstname = clean_data.get("firstname", "Inconnu").strip()
    lastname = clean_data.get("lastname", "Inconnu").strip()
    city = clean_data.get("city", "Inconnu").strip()
    description = clean_data.get("description", "Inconnu").strip()

    # === Construction du DataFrame global ===
    base_rows, exp_rows, dip_rows, course_rows = [], [], [], []

    base_rows.append({
        "id": user_id,
        "firstname": firstname,
        "lastname": lastname,
        "city": city,
        "description": description,
        "source": "base"
    })

    for d in clean_data.get("diplomas", []):
        dip_rows.append({
            "id": user_id,
            "firstname": firstname,
            "lastname": lastname,
            "title": d.get("title", "Inconnu"),
            "level": d.get("level", "Inconnu"),
            "institution": d.get("institution", "Inconnu"),
            "source": "diploma"
        })

    for e in clean_data.get("experiences", []):
        exp_rows.append({
            "id": user_id,
            "firstname": firstname,
            "lastname": lastname,
            "title": e.get("title", "Inconnu"),
            "company": e.get("company", "Inconnu"),
            "city": e.get("city", e.get("location", e.get("country", "Inconnu"))),
            "duration": e.get("duration", e.get("dates", "Inconnu")),
            "description": e.get("description", "Inconnu"),
            "source": "experience"
        })

    for c in clean_data.get("pastCourses", []):
        course_rows.append({
            "id": user_id,
            "firstname": firstname,
            "lastname": lastname,
            "title": c.get("title", "Inconnu"),
            "numberOfStars": c.get("numberOfStars", np.nan),
            "course_code": c.get("course_code", "Inconnu"),
            "course_level": c.get("course_level", "Inconnu"),
            "description": c.get("description", c.get("course_description", "Inconnu")),
            "start_date": c.get("start_date", "Inconnu"),
            "end_date": c.get("end_date", "Inconnu"),
            "source": "pastcourse"
        })

    df_base = pd.DataFrame(base_rows)
    df_exp = pd.DataFrame(exp_rows)
    df_dip = pd.DataFrame(dip_rows)
    df_course = pd.DataFrame(course_rows)

    # Process experiences
    if "duration" not in df_exp.columns:
        df_exp["duration"] = "Inconnu"

    df_exp["duration"] = df_exp["duration"].apply(convert_duration_to_months)
    df_exp["duration"] = df_exp["duration"].fillna(0).astype(int)

    for col in ["description", "city", "company"]:
        if col not in df_exp.columns:
            df_exp[col] = "Inconnu"
        df_exp[col] = df_exp[col].fillna("Inconnu")

    df_exp = df_exp.dropna(subset=["duration"])
    df_exp["duration"] = df_exp["duration"].fillna(0).astype(int)

    # Process diplomas
    df_dip = remplacer_nan_par_inconnu(df_dip, "institution")

    # Process courses
    df_course = rename_year_columns(df_course)

    # Calculate duration only once
    df_course = calculate_duration_in_months(df_course)
    df_course = remplacer_nan_par_inconnu(df_course, "duration")

    # Combine all dataframes
    df_all = pd.concat([df_base, df_exp, df_dip, df_course], ignore_index=True, sort=False).fillna("Inconnu")

    df_all["numberOfStars"] = pd.to_numeric(df_all.get("numberOfStars", np.nan), errors="coerce")

    # Calculate features
    nombre_exp = df_all[df_all["source"] == "experience"].groupby("id").size().rename("nombre_experiences")
    nb_cours = df_all[df_all["source"] == "pastcourse"].groupby("id").size().rename("nb_cours")
    moyenne_notes = df_all[df_all["source"] == "pastcourse"].groupby("id")["numberOfStars"].mean().rename(
        "moyenne_notes")

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
    df_all = df_all.merge(df_features, on="id", how="left")

    text_input = \
    df_all[df_all["id"] == user_id][['description', 'company']].fillna('').astype(str).agg(' '.join, axis=1).iloc[0]
    X_text = tfidf.transform([text_input])

    features_row = df_all[df_all["id"] == user_id].iloc[0]

    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    X_num_array = np.array([[safe_float(features_row['duration']),
                             safe_float(features_row['nombre_experiences']),
                             safe_float(features_row['nb_cours']),
                             safe_float(features_row['moyenne_notes']),
                             safe_float(features_row['score_reputation'])]])

    X_num = csr_matrix(X_num_array)

    X_final = hstack([X_text, X_num])

    # Make prediction
    prediction = model.predict(X_final)[0]
    return {"gradeAverage": round(float(prediction), 2)}