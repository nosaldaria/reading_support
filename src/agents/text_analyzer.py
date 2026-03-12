import numpy as np
import re
import os
import warnings
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# Завантаження ресурсів NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TextAnalyzerAgent:
    def __init__(self, model_path="models/paragraph_classifier.pkl"):
        self.model = None
        self.levels = ["легкий", "середній", "складний"]
        self.device = torch.device("cpu")

        if os.path.exists(model_path):
            try:
                import joblib
                self.pipeline = joblib.load(model_path)
                self.model = self.pipeline["model"]

                bert_name = self.pipeline["bert_model_name"]

                # self.bert_tokenizer = self.pipeline["bert_tokenizer"]
                # self.bert_model = self.pipeline["bert_model"].to(self.device).eval()


                self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
                self.bert_model = AutoModel.from_pretrained(bert_name).to(self.device).eval()

                self.pca_bert = self.pipeline["pca_bert"]
                self.read_scaler = self.pipeline["read_scaler"]
                # self.feature_selector = self.pipeline["feature_selector"]

                self.feature_selector = None
                self.label_encoder = self.pipeline["label_encoder"]

                print("Повний пайплайн моделі завантажено")
                print(f"Класи моделі: {self.label_encoder.classes_}")
                print(
                    f"Очікувані фічі моделі: {self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 'невідомо'}"
                )
            except Exception as e:
                print(f"Помилка моделі: {e}")
                self.model = None
        else:
            print("Модель відсутня — використовуємо правила")
            self.model = None

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)

    def get_bert_embeddings(self, text):
        inputs = self.bert_tokenizer(
            text, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            emb = self.mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()

        bert_pca = self.pca_bert.transform(emb)
        print(f"  BERT shape: {emb.shape} → PCA: {bert_pca.shape}")
        return bert_pca

    def paragraph_features(self, text):
        words = re.findall(r"\b[а-яіїєґ]+\b", text.lower())
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            print("  Resource punkt_tab відсутній, використовується простий split по крапках.")
            sentences = text.split('.')

        asl = len(words) / max(len(sentences), 1)
        awl = np.mean([len(w) for w in words]) if words else 0
        ttr = len(set(words)) / max(len(words), 1)
        long_words = sum(len(w) >= 8 for w in words) / max(len(words), 1)
        pisarek = 100 - 10 * (sum(len(w) > 7 for w in words) / max(len(words), 1))
        sent_count = len(sentences)

        features = np.array([[asl, awl, ttr, long_words, pisarek, sent_count]])
        print(f"  Readability: ASL={asl:.1f}, AWL={awl:.1f}, Pisarek={pisarek:.1f}")
        return features

    def prepare_features(self, text):
        print("  Підготовка фіч...")
        bert_features = self.get_bert_embeddings(text)
        read_features = self.paragraph_features(text)

        # Масштабуємо метрики читабельності
        read_scaled = self.read_scaler.transform(read_features)

        # Перевірка: чи існує селектор фіч у цій версії моделі
        if self.feature_selector is not None:
            read_processed = self.feature_selector.transform(read_scaled)
            print("  Використано Feature Selector")
        else:
            read_processed = read_scaled
            print("  Feature Selector відсутній, крок пропущено")

        X_final = np.hstack([bert_features, read_processed])
        print(f"  Фінальні фічі: {X_final.shape}")
        return X_final

    def rule_based_fallback(self, text):
        words = re.findall(r"\b[а-яіїєґ]+\b", text.lower())
        total_chars = len(text)

        if len(words) <= 10 or total_chars <= 65:
            return "легкий", 0.8
        elif len(words) >= 16 or total_chars >= 112:
            return "складний", 0.8
        else:
            return "середній", 0.7

    def analyze(self, text):
        if not text or len(text.strip()) < 10:
            return {"predicted_level": "середній", "confidence": 0.5, "stats": {"ukr_words": 0}}

        try:
            print(f"\nАНАЛІЗ: '{text[:50]}...'")
            words = re.findall(r"\b[а-яіїєґ]+\b", text.lower())
            try:
                sentences = len(sent_tokenize(text))
            except LookupError:
                sentences = len(text.split('.'))

            stats = {
                'ukr_words': len(words),
                'sentences': sentences,
                'avg_sentence_length': round(len(words) / max(sentences, 1), 1),
                'total_chars': len(text)
            }

            if self.model:
                try:
                    features = self.prepare_features(text)
                    pred = self.model.predict(features)[0]
                    proba = self.model.predict_proba(features)[0]

                    print(f"  Предикт: {pred} (raw)")
                    print(f"  Проби: {proba}")

                    level_map = {0: "легкий", 1: "середній", 2: "складний"}
                    level = level_map.get(pred, "середній")
                    confidence = float(max(proba))

                    print(f"Модель: {level} ({confidence:.2f})")
                    return {
                        "predicted_level": level,
                        "confidence": round(confidence, 2),
                        "stats": stats,
                        "level_probabilities": {"легкий": proba[0], "середній": proba[1], "складний": proba[2]}
                    }
                except Exception as model_error:
                    print(f"  Модель впала: {model_error}")
                    level, confidence = self.rule_based_fallback(text)
            else:
                level, confidence = self.rule_based_fallback(text)
                print(f"Правила: {level} ({confidence:.2f})")

            return {
                "predicted_level": level,
                "confidence": round(confidence, 2),
                "stats": stats,
                "level_probabilities": {
                    "легкий": 0.7 if level == "легкий" else 0.15,
                    "середній": 0.7 if level == "середній" else 0.2,
                    "складний": 0.7 if level == "складний" else 0.15
                }
            }

        except Exception as e:
            print(f"Критична помилка: {e}")
            return {"predicted_level": "середній", "confidence": 0.5, "stats": {"ukr_words": 0}}