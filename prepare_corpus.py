import os
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from pdfminer.high_level import extract_text
from langdetect import detect, DetectorFactory
import nltk
nltk.data.path.append("C:/Users/parch/AppData/Roaming/nltk_data")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

# Download NLTK resources (only first time)
nltk.download("punkt")
nltk.download("stopwords")

# ==============================
# CONFIG
# ==============================
DATASET_DIR = Path("dataset")
OUTPUT_CORPUS = "iitj_corpus_clean.txt"
STATS_FILE = "dataset_stats.txt"
WORDCLOUD_FILE = "wordcloud.png"

# ==============================
# STEP 1: Extract Text from PDFs
# ==============================

def extract_pdf_text(pdf_path):
    try:
        text = extract_text(str(pdf_path))
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# ==============================
# STEP 2: Keep English Only
# ==============================

def keep_english_text(text):
    lines = text.split("\n")
    english_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue

        try:
            if detect(line) == "en":
                english_lines.append(line)
        except:
            continue

    return " ".join(english_lines)

# ==============================
# STEP 3: Remove Boilerplate / Artifacts
# ==============================

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove page numbers
    text = re.sub(r"\bPage\s*\d+\b", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove non-English characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==============================
# STEP 4: Tokenization & Lowercasing
# ==============================

def tokenize_and_filter(text):
    text = text.lower()
    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    return tokens

# ==============================
# MAIN PIPELINE
# ==============================

all_documents = []
all_tokens = []

pdf_files = list(DATASET_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf in tqdm(pdf_files):
    raw_text = extract_pdf_text(pdf)
    english_text = keep_english_text(raw_text)
    cleaned_text = clean_text(english_text)
    tokens = tokenize_and_filter(cleaned_text)

    if len(tokens) > 0:
        all_documents.append(tokens)
        all_tokens.extend(tokens)

# ==============================
# SAVE CLEAN CORPUS
# ==============================

with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
    for doc in all_documents:
        f.write(" ".join(doc) + "\n")

print(f"\nClean corpus saved to {OUTPUT_CORPUS}")

# ==============================
# DATASET STATISTICS
# ==============================

total_documents = len(all_documents)
total_tokens = len(all_tokens)
vocab_size = len(set(all_tokens))

stats_text = f"""
DATASET STATISTICS
-------------------
Total Documents: {total_documents}
Total Tokens: {total_tokens}
Vocabulary Size: {vocab_size}
"""

print(stats_text)

with open(STATS_FILE, "w") as f:
    f.write(stats_text)

# ==============================
# WORD CLOUD
# ==============================

word_freq = Counter(all_tokens)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate_from_frequencies(word_freq)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in IIT Jodhpur Corpus")
plt.savefig(WORDCLOUD_FILE)
plt.show()

print(f"Wordcloud saved to {WORDCLOUD_FILE}")