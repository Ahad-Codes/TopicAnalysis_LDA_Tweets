import pandas as pd
import glob
import os
import re
from datetime import datetime


# Load the stopwords from the txt file.
def load_stopwords(file_path='./DataSet/stop_words_english.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip().lower() for line in f if line.strip())

# Process each line of code
def clean_text(text, stopwords, keywords=None):
    if not isinstance(text, str):
        return ''

    # 1. delete the link
    text = re.sub(r'http\S+|www\S+', '', text).strip()

    # 2. delete the "RT @\w+"(maybe the forwarding tag)
    text = re.sub(r'^RT @\w+:', '', text).strip()

    # 3. delete"..."(If the original was a emojis, this would"..." here)
    text = text.replace('...', '')

    # 4. delete tag
    text = re.sub(r'#\S+', '', text).strip()

    # 5. delete non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 6. convert all letters to lowercase
    text = text.lower()

    # 7. delete stopwords
    words = [word for word in text.split() if word not in stopwords]

    # delete tweets with less than 2 words
    return ' '.join(words) if len(words) > 2 else ''

# Generating ids
def generate_custom_ids(df):
    df = df.reset_index(drop=True)
    df['id'] = df.index + 1# start from 1
    df['id'] = df['id'].apply(lambda x: f"ID{x:06d}")  # ID format like "ID000001"
    return df

# Grouped by month and saved as a CSV file
def group_by_month_and_save(df, date_column='created_at'):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    grouped = df.groupby(df[date_column].dt.to_period('M'))
    for period, group in grouped:
        period_str = f"{period.year}_{period.month}"  # Format：2020_4
        filename = f"{period_str}_dataset.csv"
        group = generate_custom_ids(group)
        group = group[['id', date_column, 'original_text']]

        group.to_csv(filename, index=False)
        print(f"Saved {filename}")

# Dataset file
file_name = "./DataSet/Covid-19 Twitter Dataset (Aug-Sep 2020).csv"

try:
    # load stopwords
    stopwords = load_stopwords()

    # load csv file
    df = pd.read_csv(file_name, encoding='utf-8')

    if 'original_text' in df.columns and 'created_at' in df.columns:

        # process each line of text
        cleaned_texts = df['original_text'].apply(lambda x: clean_text(x, stopwords))
        df['original_text'] = cleaned_texts[cleaned_texts != '']

        # save to the new csv file
        group_by_month_and_save(df, date_column='created_at')

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")