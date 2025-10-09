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

# Load pre-trained model and vectorizer
model = joblib.load("best_cat.pkl")
tfidf = joblib.load("tfidf_vectorizer_xgb.pkl")


# === Helper Functions ===

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
        return max(diff.years * 12 + diff.months, 0)

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
        df[colonne] = df[colonne].fillna("Inconnu").replace("", "Inconnu")
    return df


# === NEW: Function to assign coefficient based on diploma 'level' ===
def get_diplome_coef(level):
    if pd.isna(level) or str(level).strip().lower() == "inconnu" or str(level).strip() == "":
        return np.nan
    text = str(level).lower()
    if any(word in text for word in ["doctorat", "phd", "doctor", "dr ", "dr.", "dr ing", "professeur", "professor"]):
        return 5
    if any(word in text for word in
           ["master", "maîtrise", "maitrise", "maître", "maitre", "post-graduation", "post diplome", "post-diplome"]):
        return 4
    if any(word in text for word in ["licence", "diplôme professionnel", "diplome professionnel", "professionnel"]):
        return 3
    if any(word in text for word in
           ["diplôme", "diplome", "état", "formation", "certificat", "certificate", "certifié", "certifie",
            "aptitude"]):
        return 2
    if any(word in text for word in ["secondaire", "senior"]):
        return 1
    return np.nan


# === API Endpoint ===

@app.post("/api/predict")
async def predict(request: Request):
    data = await request.json()
    user_id = data.get("id", "temp")

    # === 1. Basic value cleaning ===
    clean_data = {k: (v if v is not None else "Inconnu") for k, v in data.items()}

    # === 2. Building DataFrames from JSON sections ===
    base_rows, exp_rows, dip_rows, course_rows = [], [], [], []

    base_rows.append({
        "id": user_id, "description": clean_data.get("description", "Inconnu"), "source": "base"
    })

    for d in clean_data.get("diplomas", []):
        dip_rows.append({
            "id": user_id, "title": d.get("title", "Inconnu"), "level": d.get("level", "Inconnu"),
            "institution": d.get("institution", "Inconnu"), "source": "diploma"
        })

    for e in clean_data.get("experiences", []):
        exp_rows.append({
            "id": user_id, "title": e.get("title", "Inconnu"), "company": e.get("company", "Inconnu"),
            "duration": e.get("duration", e.get("dates", "Inconnu")),
            "description": e.get("description", "Inconnu"), "source": "experience"
        })

    for c in clean_data.get("pastCourses", []):
        course_rows.append({
            "id": user_id, "title": c.get("title", "Inconnu"), "numberOfStars": c.get("numberOfStars", np.nan),
            "course_level": c.get("course_level", "Inconnu"),
            "description": c.get("description", c.get("course_description", "Inconnu")),
            "start_date": c.get("start_date", "Inconnu"), "end_date": c.get("end_date", "Inconnu"),
            "source": "pastcourse"
        })

    # === 3. Preprocessing and Combining DataFrames ===
    df_exp = pd.DataFrame(exp_rows)
    if not df_exp.empty:
        df_exp["duration"] = df_exp["duration"].apply(convert_duration_to_months).fillna(0).astype(int)

    df_course = pd.DataFrame(course_rows)
    if not df_course.empty:
        df_course = calculate_duration_in_months(df_course)

    df_all = pd.concat([
        pd.DataFrame(base_rows), df_exp, pd.DataFrame(dip_rows), df_course
    ], ignore_index=True).fillna("Inconnu")

    df_all["numberOfStars"] = pd.to_numeric(df_all["numberOfStars"], errors='coerce')

    # === 4. Feature Engineering ===

    # NEW: Calculate diploma coefficient from 'level' text
    mask_dip = df_all["source"] == "diploma"
    df_all["diplome_coef"] = np.nan
    if mask_dip.any():
        df_all.loc[mask_dip, "diplome_coef"] = df_all.loc[mask_dip, "level"].apply(get_diplome_coef)
    df_all['diplome_coef'] = pd.to_numeric(df_all['diplome_coef'], errors='coerce').fillna(0)

    # Aggregate features for the user
    nombre_exp = (df_all["source"] == "experience").sum()
    nb_cours = (df_all["source"] == "pastcourse").sum()
    moyenne_notes = df_all["numberOfStars"].mean()
    total_duration = df_all[df_all["source"] == 'experience']['duration'].sum()
    max_diplome_coef = df_all["diplome_coef"].max()

    df_features = pd.DataFrame([{
        "nombre_experiences": nombre_exp,
        "nb_cours": nb_cours,
        "moyenne_notes": moyenne_notes,
        "total_duration": total_duration,
        "max_diplome_coef": max_diplome_coef
    }]).fillna(0)

    # Calculate reputation score
    coef_exp, coef_cours, coef_stars = 0.3, 0.3, 0.4
    df_features["score_reputation"] = (
                                              (coef_exp * (df_features["nombre_experiences"] / max(nombre_exp, 1))) +
                                              (coef_cours * (df_features["nb_cours"] / max(nb_cours, 1))) +
                                              (coef_stars * (df_features["moyenne_notes"] / 5))
                                      ) * 100
    df_features = df_features.fillna(0)

    # === 5. Preparing Data for the Model ===

    # MODIFIED: Combine all specified text columns
    text_cols = ['description', 'company', 'title', 'level', 'institution', 'course_level']
    text_input = ' '.join(str(df_all[col].fillna('').agg(' '.join)) for col in text_cols)
    X_text = tfidf.transform([nettoyer_texte(text_input)])

    # Get the single row of aggregated features
    features_row = df_features.iloc[0]

    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # MODIFIED: Use aggregated features including the new diploma coefficient
    X_num_array = np.array([[
        safe_float(features_row['total_duration']),
        safe_float(features_row['nombre_experiences']),
        safe_float(features_row['nb_cours']),
        safe_float(features_row['moyenne_notes']),
        safe_float(features_row['score_reputation']),
        safe_float(features_row['max_diplome_coef'])
    ]])

    X_num = csr_matrix(X_num_array)
    X_final = hstack([X_text, X_num])

    # === 6. Make Prediction ===
    prediction = model.predict(X_final)[0]
    return {"gradeAverage": round(float(prediction), 2)}