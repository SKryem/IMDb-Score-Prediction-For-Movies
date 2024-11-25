import string
import pandas as pd
import csv
import re
import os

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    # 'cookie': 'session-id=140-9588448-6326254; session-id-time=2082787201l; ad-oo=0; ci=e30; ubid-main=134-0576476-6044551; session-token=cGDJotr2f01wkkZSjtD1ZhIYssIQsIaYiBWH7OrewBLlLEEAD9xj6fTScK1CaC/Sckx8AHu0ZUqehs7ozijq/vrr15jh87Kz5suE90IkPJPxIILEniH+Yslp/ykA/zwg5QK57ebdg1+ecmISu1czgbhiL2HTAAXCnErGFs7pPVhticj5CLdibXOn5YsWqVXGhu48cs/rrckH+mVBCKWksf4pUr0pniy7P7mzREb/5hKR/mg/c9TcEf/waFuFZUW95k1c4SVUwiAGJvGCDI+pvOA+3P83nTtPMffCLY99Q1PpqoHzSRlo0aUSXvF3m2QyPC+wc/25XJ9qtIOy9pjZmkWaVyo53WgC; csm-hit=tb:TQJAJFZ7HA3A12N1ZA6P+s-N3EJ8C58W4JF1DZ5VTSG|1719934532393&t:1719934532393&adb:adblk_yes',
    'priority': 'u=0, i',
    # 'referer': 'https://www.imdb.com/?ref_=nv_home',
    'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
}


def get_movie_links():
    df = pd.read_csv("./data/movie_metadata.csv").drop_duplicates()
    movie_links = list()
    for index, row in df.iterrows():
        movie_link = row['movie_imdb_link']
        movie_links.append(movie_link)
    return movie_links


def get_movie_id(movie_link):
    return movie_link.split('/title/')[1].split('/')[0]  # ChatGPT generated


def write_rows_to_csv(filename, rows_data, _headers=None):  # ChatGPT
    try:
        # Try to open the file in append mode if it exists, else create a new one
        file_exists = False
        try:
            with open(filename, 'r') as file:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=_headers)
            if not file_exists:
                # Write headers if the file is newly created
                writer.writeheader()

            if isinstance(rows_data, dict):  # case: one row
                rows_data = [rows_data]  # "cast" the dict to a list of dicts

            # Write rows
            for row_data in rows_data:
                writer.writerow(row_data)

    except IOError as e:
        print(f"An error occurred while writing to the CSV file: {e}")


def write_checkpoint_to_file(filename, checkpoint):
    try:
        with open(filename, 'w') as file:
            file.write(str(checkpoint))
        print(f"Checkpoint to : {filename} successful: Last index: {checkpoint}")
    except Exception as e:
        print(f"Error writing number = {checkpoint} to file: {e}")


def read_number_from_file(filename):
    try:
        with open(filename, 'r') as file:
            number = int(file.read().strip())
        return number
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return 0


def load_checkpoint(filename):
    return read_number_from_file(filename)


def save_checkpoint(filename, checkpoint):
    write_checkpoint_to_file(filename, checkpoint)


# Mean number of words per review
def mean_words(reviews: list) -> float:
    total_words = sum(len(review.split()) for review in reviews)
    return round(total_words / len(reviews), 4) if reviews else 0


# Mean number of letters per word (excluding spaces)
def letters_per_word(reviews: list) -> float:
    total_letters = sum(len(word) for review in reviews for word in review.split())
    total_words = sum(len(review.split()) for review in reviews)
    return round(total_letters / total_words, 4) if total_words else 0


# Capital letters frequency
def capital_letter(reviews: list) -> float:
    total_capitals = sum(sum(1 for char in review if char.isupper()) for review in reviews)
    total_letters = sum(sum(1 for char in review if char.isalpha()) for review in reviews)
    return round(total_capitals / total_letters, 6) if total_letters else 0


# Punctuation frequency
def punctuation_freq(reviews: list) -> float:
    total_punctuations = sum(sum(1 for char in review if char in string.punctuation) for review in reviews)
    total_chars = sum(len(review) for review in reviews)
    return round(total_punctuations / total_chars, 6) if total_chars else 0


# Numbers frequency
def numbers_freq(reviews: list) -> float:
    total_numbers = sum(sum(1 for char in review if char.isdigit()) for review in reviews)
    total_letters = sum(len(review.replace(' ', '')) for review in reviews)
    return (total_numbers / total_letters) if total_letters else 0


def sentiment_avg(sentiments: list) -> float:
    score = 0
    for sentiment in sentiments:
        if sentiment == "Positive":
            score += 1
        if sentiment == "Negative":
            score -= 1
        # if sentiment == "Netural" do nothing (score += 0)
    return round(score / len(sentiments), 4) if len(sentiments) else 0


def create_dict_from_data(filename, gruopby_column, feature_column):
    df = pd.read_csv(filename)

    # Group by 'Movie Name' and aggregate the 'Review' column into lists
    movie_review_dict = df.groupby(gruopby_column)[feature_column].apply(list).to_dict()

    return movie_review_dict


def dataset_remove_added_features(data):
    data.drop('Avg_words', axis=1, inplace=True)
    data.drop('Avg_letters', axis=1, inplace=True)
    data.drop('Numbers_freq', axis=1, inplace=True)
    data.drop('Capitals_freq', axis=1, inplace=True)
    data.drop('Punc_freq', axis=1, inplace=True)
    data.drop('Avg_sentiment', axis=1, inplace=True)
    data.drop('Num Of Reviews', axis=1, inplace=True)


def dataset_drop_cols_with_names(data):
    # Drop columns with names
    data.drop('director_name', axis=1, inplace=True, errors='ignore')

    data.drop('actor_1_name', axis=1, inplace=True, errors='ignore')

    data.drop('actor_2_name', axis=1, inplace=True, errors='ignore')

    data.drop('actor_3_name', axis=1, inplace=True, errors='ignore')

    data.drop('movie_title', axis=1, inplace=True, errors='ignore')

    data.drop('plot_keywords', axis=1, inplace=True, errors='ignore')

    data.drop('genres', axis=1, inplace=True, errors='ignore')  # genres are distributed equally


def dataset_handle_categorial_data(data):
    value_counts = data["country"].value_counts()
    vals = value_counts[:2].index
    data['country'] = data.country.where(data.country.isin(vals), 'other') # keep 2 countries that appear the most,
    # convert other countries to "other"

    data['language'] = data.language.where(data.language.isin(["English"]), 'other')

    data = pd.get_dummies(data=data, columns=['country'], prefix=['country'], drop_first=True)
    data = pd.get_dummies(data=data, columns=['language'], prefix=['country'], drop_first=True)
    data = pd.get_dummies(data=data, columns=['content_rating'], prefix=['content_rating'], drop_first=True)
    data = pd.get_dummies(data=data, columns=['color'], prefix=['color'], drop_first=True)
    # data = pd.get_dummies(data=data, columns=['genres'], prefix=['genres'], drop_first=True)
    # data = pd.get_dummies(data=data, columns=['director_name'], prefix=['director_name'], drop_first=True)

    return data


def fill_missing_values_with_median(data, features_to_fill):
    for feature in features_to_fill:
        median_val = data[feature].median()
        data.fillna({feature: median_val}, inplace=True)


def fill_missing_values_with_most_frequent(data, features_to_fill):
    for feature in features_to_fill:
        most_frequent_val = data[feature].value_counts().idxmax()
        data.fillna({feature: most_frequent_val}, inplace=True)


def get_datasets():
    original_dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")
    dataset_remove_added_features(original_dataset)
    dataset_drop_cols_with_names(original_dataset)
    original_dataset = dataset_handle_categorial_data(original_dataset)

    expanded_dataset = pd.read_csv("./data/filtered_expanded_movie_metadata.csv")
    dataset_drop_cols_with_names(expanded_dataset)
    expanded_dataset = dataset_handle_categorial_data(expanded_dataset)

    return original_dataset, expanded_dataset


# we know that the checkpoint dir (for BERT fine-tuned module) is in a dir called "results"
# we use this code (ChatGPT generated) to find the checkpoint directory (its name is not constant)
def find_checkpoint_directory(base_dir):
    # Pattern to match directories like 'checkpoint-XXXXX' where XXXXX is unknown

    pattern = re.compile(r'^checkpoint-\d+$')

    # List all entries in the base directory
    entries = os.listdir(base_dir)

    # Filter and find the directory matching the pattern
    for entry in entries:
        entry_path = os.path.join(base_dir, entry)
        if os.path.isdir(entry_path) and pattern.match(entry):
            return entry_path.replace(os.sep, '/')

    return None
