import os
import pandas as pd
import re
import glob
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt', quiet=True)

PROJECT_ROOT = '/Users/daria/PycharmProjects/reading-support-system'
UKRTB_PATH = os.path.join(PROJECT_ROOT, 'data', 'ukrtb_raw')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'scripts')

print("Обробка UkrTB з 3 рівнями")
print(f"Корінь проекту: {PROJECT_ROOT}")
print(f"Шлях до даних: {UKRTB_PATH}")


def clean_ukrainian_text(text):
    text = re.sub(r'###\d+###', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\–—»«""»]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_lessons(full_text):
    lessons = re.split(r'###\d+###', full_text)
    return [lesson.strip() for lesson in lessons if lesson.strip()]


def process_single_file(filename, new_level):
    print(f"Обробляю: {os.path.basename(filename)} -> Рівень {new_level}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            full_text = f.read()
        print(f"  Розмір: {len(full_text):,} символів")

        lessons = split_into_lessons(full_text)
        print(f"  Уроків: {len(lessons)}")

        dataset = []
        for lesson_num, lesson_text in enumerate(lessons, 1):
            cleaned_lesson = clean_ukrainian_text(lesson_text)
            sentences = sent_tokenize(cleaned_lesson)

            for sent_num, sentence in enumerate(sentences):
                cleaned_sent = sentence.strip()
                if 25 < len(cleaned_sent) < 400:
                    dataset.append({
                        'filename': os.path.basename(filename),
                        'lesson_num': lesson_num,
                        'sentence_num': sent_num,
                        'text': cleaned_sent,
                        'level': new_level,
                        'difficulty': new_level / 3.0,
                        'chars': len(cleaned_sent),
                        'words': len(cleaned_sent.split())
                    })

        print(f"  Додано {len(dataset):,} речень")
        return dataset

    except Exception as e:
        print(f"  Помилка: {e}")
        return []


file_levels_3 = {
    'f01.txt': 1, 'f02.txt': 1,
    'f03.txt': 2, 'f04.txt': 2,
    'f05.txt': 3
}

print("=" * 60)
print("Діагностика файлів:")
print(f"Папка існує: {os.path.exists(UKRTB_PATH)}")

if os.path.exists(UKRTB_PATH):
    all_files = os.listdir(UKRTB_PATH)
    txt_files = [f for f in all_files if f in file_levels_3]
    print(f"Знайдено .txt файлів: {len(txt_files)}")
    for f in sorted(txt_files):
        print(f"  {f}")
else:
    print("Папка data/ukrtb_raw/ не існує!")
    exit(1)

all_data = []
raw_files_list = [os.path.join(UKRTB_PATH, f) for f in txt_files]
raw_files_sorted = sorted(raw_files_list, key=lambda x: int(re.search(r'f(\d+)\.txt', os.path.basename(x)).group(1)))

print("Обробка файлів з 3 РІВНЯМИ:")
for filename in raw_files_sorted:
    basename = os.path.basename(filename)
    data = process_single_file(filename, file_levels_3[basename])
    all_data.extend(data)

df = pd.DataFrame(all_data)
output_dir = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'ukrtb_3levels.csv')
df.to_csv(output_path, index=False)

print("=" * 60)
print(f"ФІНАЛЬНИЙ ДАТАСЕТ (3 рівні): {len(df):,} речень")
print(f"Збережено: {output_path}")

if len(df) > 0:
    print("Розподіл рівнів:")
    print(df['level'].value_counts().sort_index())

    print("Середня довжина по рівнях:")
    print(df.groupby('level')[['words', 'chars']].mean().round(1))

else:
    print("Датасет порожній!")
