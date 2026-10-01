import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

N = 20000  # exemples par source (augmentez si votre PC le permet)

# Français : critiques de films Allociné (0 = négatif, 1 = positif)
fr = load_dataset("tblard/allocine", split="train").to_pandas()
fr = fr.rename(columns={"review": "text"})[["text", "label"]]
fr = fr.sample(N, random_state=42)

# Anglais : critiques longues IMDB
en_long = load_dataset("stanfordnlp/imdb", split="train").to_pandas()[["text", "label"]]
en_long = en_long.sample(N // 2, random_state=42)

# Anglais : phrases courtes SST-2 (aide pour les textes courts)
en_short = load_dataset("nyu-mll/glue", "sst2", split="train").to_pandas()
en_short = en_short.rename(columns={"sentence": "text"})[["text", "label"]]
en_short = en_short.sample(N // 2, random_state=42)

df = pd.concat([fr, en_long, en_short]).dropna()
df["text"] = df["text"].str.lower()
print("Total exemples :", len(df), "| répartition :", df["label"].value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000,
                              strip_accents="unicode", sublinear_tf=True)),
    ("svm", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
])

model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "models/svm_model_multi.pkl")
print("Modèle enregistré : models/svm_model_multi.pkl")