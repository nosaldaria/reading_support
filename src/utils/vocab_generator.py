from datasets import load_dataset
import stanza
import json
from collections import Counter
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

ALLOWED_POS = {"NOUN", "VERB", "ADJ"}
MIN_LEN = 5
OUTPUT_PATH = "models/simple_dictionary.json"
SIMILARITY_THRESHOLD = 0.65  # Семантична близькість

print("Loading RoBERTa similarity model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("Loading Stanza Ukrainian model...")
stanza.download('uk', processors='tokenize,pos,lemma,mwt', logging_level='ERROR')
nlp = stanza.Pipeline(lang='uk', processors='tokenize,pos,lemma', use_gpu=False)


def process_text(text):
    """Stanza токенізація + POS/lemma"""
    doc = nlp(text.lower())
    tokens = []

    for sent in doc.sentences:
        for w in sent.words:
            if (w.text.isalpha() and
                    len(w.text) >= MIN_LEN and
                    w.upos in ALLOWED_POS):
                tokens.append({
                    "text": w.text,
                    "lemma": w.lemma,
                    "pos": w.upos,
                    "feats": w.feats  # беремо морфологічні характеристики
                })
    return tokens


def semantic_similarity(complex_word, simple_word):
    """RoBERTa семантична близькість"""
    try:
        complex_sent = f"Слово {complex_word}"
        simple_sent = f"Слово {simple_word}"

        emb1 = model.encode(complex_sent, convert_to_tensor=True)
        emb2 = model.encode(simple_sent, convert_to_tensor=True)

        similarity = util.cos_sim(emb1, emb2).cpu().numpy()[0][0]
        return similarity.item()
    except:
        return 0.0


def collect_pairs(limit=15000):
    print("Loading Spivavtor dataset...")
    dataset = load_dataset("grammarly/spivavtor", split="train")

    pair_counter = Counter()
    word_freq = Counter()

    for i, ex in enumerate(dataset.select(range(limit))):
        src = process_text(ex["src"])
        tgt = process_text(ex["tgt"])

        src_lemmas = [t["lemma"] for t in src]
        tgt_lemmas = [t["lemma"] for t in tgt]

        removed = [lemma for lemma in src_lemmas if lemma not in tgt_lemmas]
        added = [lemma for lemma in tgt_lemmas if lemma not in src_lemmas]

        for complex_lemma in removed:
            for simple_lemma in added:
                src_token = next((t for t in src if t["lemma"] == complex_lemma), None)
                tgt_token = next((t for t in tgt if t["lemma"] == simple_lemma), None)

                if src_token and tgt_token and src_token["pos"] == tgt_token["pos"]:
                    key = (complex_lemma, simple_lemma)
                    pair_counter[key] += 1
                    word_freq[complex_lemma] += 1
                    word_freq[simple_lemma] += 1

        if i % 3000 == 0:
            print(f"Processed {i}/{limit} examples...")

    print(f"Зібрано пар: {len(pair_counter)}")
    return pair_counter, word_freq


def generate_forms(simple_word):
    """
    Генеруємо форми для простого слова з урахуванням типових відмінків.
    Можна розширити через pymorphy2 для української мови або rule-based.
    """
    forms = {}
    doc = nlp(simple_word)
    for sent in doc.sentences:
        for w in sent.words:
            feats = w.feats or ""
            for feat in feats.split("|"):
                if "=" in feat:
                    key, val = feat.split("=")
                    forms.setdefault(key.lower(), []).append(w.text)
    return forms if forms else {"nomn": [simple_word]}  # хочемо хоча б називний


def create_quality_dictionary(pair_counter, word_freq, min_count=2):
    dictionary = {}
    candidates_checked = 0

    print("RoBERTa якість фільтрації...")

    for (complex_lemma, simple_lemma), count in pair_counter.most_common(3000):
        if count < min_count:
            continue

        if len(simple_lemma) >= len(complex_lemma):
            continue

        if word_freq[simple_lemma] < 5:
            continue

        similarity = semantic_similarity(complex_lemma, simple_lemma)
        if similarity < SIMILARITY_THRESHOLD:
            continue

        confidence = round(count / max(word_freq[complex_lemma], 1), 3)
        semantic_score = round(similarity, 3)
        quality_score = round(confidence * semantic_score, 3)

        forms = generate_forms(simple_lemma)

        dictionary[complex_lemma] = {
            "simple": simple_lemma,
            "confidence": confidence,
            "semantic_similarity": semantic_score,
            "quality_score": quality_score,
            "count": count,
            "source": "spivavtor",
            "forms": forms
        }

        candidates_checked += 1
        if candidates_checked % 500 == 0:
            print(f"  Перевірено {candidates_checked} кандидатів...")

    print(f"Залишилось {len(dictionary)} якісних пар")
    return dictionary


def save_dictionary(dictionary, path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"Збережено: {path} ({len(dictionary)} слів)")


if __name__ == "__main__":
    pairs, freq = collect_pairs(limit=15000)
    dictionary = create_quality_dictionary(pairs, freq, min_count=2)
    save_dictionary(dictionary)

    print("\nТОП-20 ВІДФІЛЬТРОВАНИХ СПРОЩЕНЬ:")
    print("   complex_word     → simple_word (conf | sem | quality)")
    print("-" * 70)
    for complex_w, info in sorted(dictionary.items(),
                                  key=lambda x: x[1]["quality_score"],
                                  reverse=True)[:20]:
        print(f"  {complex_w:15} → {info['simple']:12} "
              f"({info['confidence']:.2f} | {info['semantic_similarity']:.2f} | {info['quality_score']:.3f})")
