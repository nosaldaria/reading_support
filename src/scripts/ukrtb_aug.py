import pandas as pd
import random
import nlpaug.augmenter.word as naw
import nltk
import re
import os

nltk.download('punkt', quiet=True)

PROJECT_ROOT = '/Users/daria/PycharmProjects/reading-support-system'
input_path = os.path.join(PROJECT_ROOT, 'data', 'ukrtb_3levels.csv')
output_path = os.path.join(PROJECT_ROOT, 'data', 'ukrtb.csv')

print("Завантаження датасету...")
df = pd.read_csv(input_path)
print("Поточний розподіл:")
print(df['level'].value_counts().sort_index())


class UkrainianAugmenter:
    def __init__(self):
        self.syn_aug = naw.SynonymAug(aug_p=0.3, aug_max=3)

    def simple_paraphrase(self, text):
        try:
            aug_text = self.syn_aug.augment(text)[0]
            return aug_text
        except:
            words = text.split()
            if len(words) > 5:
                idx = random.randint(1, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
            return ' '.join(words)

    def augment_level(self, texts, n_samples):
        augmented = []
        sample_texts = random.choices(texts, k=n_samples)
        for text in sample_texts:
            aug_text = self.simple_paraphrase(text)
            augmented.append(aug_text)
        return augmented


augmenter = UkrainianAugmenter()

level_counts = df['level'].value_counts().sort_index()
target_size = 9165

print("\nАугментація до розміру Level 2 (9165):")
for level in [1, 3]:
    current = level_counts[level]
    needed = target_size - current
    print(f"Level {level}: {current} -> {target_size} (+{needed})")

level1_texts = df[df['level'] == 1]['text'].tolist()
level1_needed = target_size - level_counts[1]
level1_aug = augmenter.augment_level(level1_texts, level1_needed)
level1_df = pd.DataFrame({
    'text': level1_aug,
    'level': 1,
    'difficulty': 1 / 3.0,
    'source': 'augmented_level1'
})

level3_texts = df[df['level'] == 3]['text'].tolist()
level3_needed = target_size - level_counts[3]
level3_aug = augmenter.augment_level(level3_texts, level3_needed)
level3_df = pd.DataFrame({
    'text': level3_aug,
    'level': 3,
    'difficulty': 3 / 3.0,
    'source': 'augmented_level3'
})

df_balanced = pd.concat([
    df[df['level'] == 1],
    df[df['level'] == 2],
    df[df['level'] == 3],
    level1_df,
    level3_df
], ignore_index=True)

print("\nФінальний розподіл:")
print(df_balanced['level'].value_counts().sort_index())
print(f"Загалом: {len(df_balanced)} речень")

columns_to_save = ['filename', 'lesson_num', 'sentence_num', 'text', 'level', 'difficulty', 'chars', 'words']
df_balanced_subset = df_balanced[columns_to_save].copy()
df_balanced_subset.fillna({
    'filename': 'augmented',
    'lesson_num': 0,
    'sentence_num': 0
}, inplace=True)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_balanced_subset.to_csv(output_path, index=False)
print(f"Збережено: {output_path}")
