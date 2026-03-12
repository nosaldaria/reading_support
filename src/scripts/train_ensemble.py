import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import nltk
from nltk.tokenize import sent_tokenize

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

MODEL_DIR = "../../models"
os.makedirs(MODEL_DIR, exist_ok=True)


df_sentences = pd.read_csv(
    "/Users/daria/PycharmProjects/reading-support-system/data/ukrtb.csv"
)

def pisarek_score(text):
    words = re.findall(r"\b[а-яіїєґ]+\b", text.lower())
    if not words:
        return 100.0
    complex_words = sum(len(w) > 7 for w in words)
    return 100 - 10 * (complex_words / len(words))

def aggregate_to_paragraphs(df, sentences_per_para=4):
    paragraphs = []

    for level in [1, 2, 3]:
        level_df = df[df["level"] == level]

        for i in range(0, len(level_df), sentences_per_para):
            chunk = level_df.iloc[i:i+sentences_per_para]
            if len(chunk) < sentences_per_para:
                continue

            text = " ".join(chunk["text"].tolist())
            words = re.findall(r"\b[а-яіїєґ]+\b", text.lower())

            paragraphs.append({
                "text": text,
                "level": level,
                "pisarek": pisarek_score(text),
                "sent_count": len(sent_tokenize(text)),
                "asl": len(words) / max(len(sent_tokenize(text)), 1),
                "awl": np.mean([len(w) for w in words]) if words else 0,
                "ttr": len(set(words)) / max(len(words), 1),
                "long_words": sum(len(w) >= 8 for w in words) / max(len(words), 1)
            })

    return pd.DataFrame(paragraphs)

df = aggregate_to_paragraphs(df_sentences)
print("Абзаців:", len(df))


le = LabelEncoder()
y = le.fit_transform(df["level"])
print("Класи:", le.classes_)

MODEL_NAME = "youscan/ukr-roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

def mean_pooling(out, mask):
    t = out.last_hidden_state
    mask = mask.unsqueeze(-1).float()
    return (t * mask).sum(1) / mask.sum(1)

def bert_embeddings(texts, batch_size=16):
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = bert(**tokens)
            emb = mean_pooling(out, tokens["attention_mask"])
            embs.append(emb.cpu().numpy())

    return np.vstack(embs)

print("BERT embeddings...")
X_bert = bert_embeddings(df["text"].tolist())
print("BERT shape:", X_bert.shape)

X_read = df[["asl", "awl", "ttr", "long_words", "pisarek", "sent_count"]].values

scaler = StandardScaler()
X_read = scaler.fit_transform(X_read)

pca = PCA(n_components=32, random_state=42)
X_bert = pca.fit_transform(X_bert)

X = np.hstack([X_bert, X_read])
print("Final features:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(acc, 3))
print("\nReport:")
print(classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred)
))

pipeline = {
    "model": model,
    "bert_model_name": MODEL_NAME,
    "pca_bert": pca,
    "read_scaler": scaler,
    "label_encoder": le
}

joblib.dump(pipeline, f"{MODEL_DIR}/paragraph_classifier.pkl", compress=5)
print("\n МОДЕЛЬ ЗБЕРЕЖЕНА")
