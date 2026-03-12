import json
import re
import os
import stanza
from collections import Counter

ALLOWED_POS = {"NOUN", "VERB", "ADJ"}
MIN_LEN = 4

print("Loading Stanza Ukrainian model...")
stanza.download("uk", processors="tokenize,pos,lemma", logging_level="ERROR")
nlp = stanza.Pipeline(lang="uk", processors="tokenize,pos,lemma", use_gpu=False)


class TextSimplificationAgent:
    def __init__(
        self,
        # dictionary_path="src/utils/models/simple_dictionary.json",
        fillers_path="src/scripts/fillers.json",
        auto_update=False
    ):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dictionary_path = os.path.join(BASE_DIR, "utils", "models", "simple_dictionary.json")
        self.fillers_path = fillers_path
        self.auto_update = auto_update

        self.word_map = self.load_word_map(self.dictionary_path)
        self.filler_patterns = self.load_fillers(fillers_path)

        self.fallback_stats = Counter()
        self._token_cache = {}

        print(f"Dictionary loaded: {len(self.word_map)} words")
        print(f"Fillers loaded: {len(self.filler_patterns)} patterns")
        print(f"Auto-update: {auto_update}")

    # ------------------ Dictionary ------------------
    def load_word_map(self, path):
        if not os.path.exists(path):
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        normalized = {}
        for k, v in data.items():
            if isinstance(v, dict) and "simple" in v:
                normalized[k] = {
                    "simple": v["simple"],
                    "confidence": v.get("confidence", 0.5),
                    "semantic_similarity": v.get("semantic_similarity", 0.0),
                    "quality_score": v.get("quality_score", 0.5),
                    "source": v.get("source", "unknown"),
                    "forms": v.get("forms", {})
                }
            elif isinstance(v, str):
                normalized[k] = {
                    "simple": v,
                    "confidence": 1.0,
                    "source": "legacy",
                    "forms": {}
                }
        return normalized

    # ------------------ Load fillers ------------------
    def load_fillers(self, path):
        if not os.path.exists(path):
            print(f"Fillers file not found: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        patterns = []
        # data може бути списком або словником
        if isinstance(data, dict):
            phrases = [p for sublist in data.values() for p in sublist]
        else:
            phrases = data

        for phrase in phrases:
            escaped = re.escape(phrase)
            # regex для видалення філерів з пробілами та пунктуацією навколо
            pattern = rf"(?:^|\s){escaped}(?=[\s,.:;!?]|$)"
            patterns.append(pattern)

        return patterns

    # ------------------ Filler removal ------------------
    def remove_fillers(self, text):
        cleaned = text
        for pattern in self.filler_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return self.normalize_punctuation(cleaned)

    # ------------------ Punctuation normalization ------------------
    @staticmethod
    def normalize_punctuation(text: str) -> str:
        # прибрати пробіли перед пунктуацією
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        # пробіли після пунктуації
        text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
        # прибрати коми перед крапками
        text = re.sub(r",\.", ".", text)
        # звести кілька крапок до однієї
        text = re.sub(r"\.{2,}", ".", text)
        # прибрати кілька пробілів
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ------------------ Token processing ------------------
    def _process_tokens(self, text):
        if text in self._token_cache:
            return self._token_cache[text]

        doc = nlp(text.lower())
        tokens = []

        for sent in doc.sentences:
            for w in sent.words:
                if (
                    w.text.isalpha()
                    and len(w.text) >= MIN_LEN
                    and w.upos in ALLOWED_POS
                ):
                    tokens.append({
                        "text": w.text,
                        "lemma": w.lemma,
                        "pos": w.upos,
                        "feats": w.feats
                    })

        self._token_cache[text] = tokens
        return tokens

    # ------------------ Morphology ------------------
    def get_form(self, lemma, feats):
        if lemma not in self.word_map:
            return None

        forms = self.word_map[lemma].get("forms", {})
        if feats:
            for feat in feats.split("|"):
                if "=" in feat:
                    key, _ = feat.split("=")
                    key = key.lower()
                    if key in forms and forms[key]:
                        return forms[key][0]

        return self.word_map[lemma]["simple"]

    # ------------------ Levels ------------------
    def get_simplification_level(self, text_level):
        levels = {
            "легкий": {"min_conf": 0.7, "max_words": 15},
            "середній": {"min_conf": 0.4, "max_words": 12},
            "складний": {"min_conf": 0.2, "max_words": 10}
        }
        return levels.get(text_level, levels["середній"])

    # ------------------ Lexical simplify ------------------
    def lexical_simplify(self, text, text_level):
        config = self.get_simplification_level(text_level)
        min_conf = config["min_conf"]

        tokens = self._process_tokens(text)
        modified = text

        for t in tokens:
            for candidate in {t["lemma"], t["text"]}:
                if candidate in self.word_map:
                    info = self.word_map[candidate]
                    # if info["confidence"] >= min_conf: закоментував
                    replacement = self.get_form(candidate, t["feats"])
                    if replacement:
                        pattern = rf"\b{re.escape(t['text'])}\b"
                        modified = re.sub(
                            pattern, replacement, modified, flags=re.IGNORECASE
                        )
                    break

        return modified, True

    # def lexical_simplify(self, text, text_level):
    #     config = self.get_simplification_level(text_level)
    #     min_conf = config["min_conf"]
    #
    #     tokens = self._process_tokens(text)
    #     modified = text
    #
    #     # Сортуємо токени від найдовших до найкоротших
    #     # Це важливо, щоб спочатку замінити "юридичний відділ", а потім просто "відділ"
    #     tokens.sort(key=lambda x: len(x['text']), reverse=True)
    #
    #     for t in tokens:
    #         original_word = t["text"]
    #         # Перевіряємо і лему, і саме слово в нижньому регістрі
    #         candidates = [t["lemma"].lower(), t["text"].lower()]
    #
    #         for candidate in candidates:
    #             if candidate == "юридичний":
    #                 if candidate in self.word_map:
    #                     info = self.word_map[candidate]
    #
    #                     # if info["confidence"] >= min_conf:
    #                     replacement = self.get_form(candidate, t["feats"])
    #
    #                     if replacement:
    #                         # 1. Підтримуємо регістр (якщо було "Юридичний", стане "Правовий")
    #                         if original_word[0].isupper():
    #                             replacement = replacement.capitalize()
    #
    #                         # 2. Надійний regex для кирилиці замість \b
    #                         # Каже: "Шукай це слово, якщо навколо немає кириличних літер"
    #                         pattern = rf"(?<![а-яіїєґА-ЯІЇЄҐ]){re.escape(original_word)}(?![а-яіїєґА-ЯІЇЄҐ])"
    #
    #                         # Виконуємо заміну лише один раз за прохід для конкретного слова
    #                         modified = re.sub(pattern, replacement, modified, count=1)
    #                     break  # Вихід з циклу кандидатів, якщо заміна знайдена
    #
    #     return modified, True

    # ------------------ Syntactic simplify ------------------
    def syntactic_simplify(self, text, text_level):
        config = self.get_simplification_level(text_level)
        max_words = config["max_words"]

        doc = nlp(text)
        simplified = []

        for sent in doc.sentences:
            buffer = []

            for w in sent.words:
                buffer.append(w.text)

                if (
                    w.upos in {"SCONJ", "CCONJ"}
                    or len(buffer) >= max_words
                ):
                    simplified.append(" ".join(buffer).strip() + ".")
                    buffer = []

            if buffer:
                simplified.append(" ".join(buffer).strip() + ".")

        return " ".join(simplified)

    # ------------------ Pipeline ------------------
    def simplify(self, text, text_level="середній"):
        if not text.strip():
            return text

        # 1️⃣ Видаляємо філери + нормалізація
        text = self.remove_fillers(text)

        # 2️⃣ Лексичне спрощення
        text, _ = self.lexical_simplify(text, text_level)

        # 3️⃣ Синтаксичне спрощення
        text = self.syntactic_simplify(text, text_level)

        # 4️⃣ Фінальна нормалізація пунктуації
        text = self.normalize_punctuation(text)

        return text.strip()

    # ------------------ Stats ------------------
    def get_stats(self):
        return {
            "total_words": len(self.word_map),
            "high_quality_words": sum(
                1 for w in self.word_map.values()
                if w.get("quality_score", 0) > 0.5
            )
        }


# ------------------ Test ------------------
if __name__ == "__main__":
    agent = TextSimplificationAgent(auto_update=True)

    text = (
        "В принципі, слід зазначити, що методологія аналізу даних у певному сенсі потребує доопрацювання. "
        "Як правило, процес обробки інформації фактично відбувається із затримками, що можливо впливає на результати. "
        "Можна сказати, що система в значній мірі реалізується відповідно до початкових вимог. "
        "У цілому, варто сказати, що отримані показники є досить складними для інтерпретації. "
        "Так би мовити, результати аналізу надзвичайно важливі для подальшого прийняття рішень."
    )

    print("ORIGINAL:", text)
    print("SIMPLIFIED:", agent.simplify(text, "складний"))
