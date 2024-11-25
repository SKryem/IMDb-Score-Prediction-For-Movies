import pandas as pd
from utils import fill_missing_values_with_most_frequent, fill_missing_values_with_median, \
                  dataset_drop_cols_with_names, dataset_handle_categorial_data


# main movie_matadata.csv file downloaded from:
# https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset


# clean the data and prepare it for the prediction model
# idea credits go to :https://www.kaggle.com/code/saurav9786/imdb-score-prediction-for-movies
def clean_source_data():
    metadata_path = "./data/movie_metadata.csv"
    metadata = pd.read_csv(metadata_path)

    filtered_data = metadata
    # Remove duplicates
    filtered_data.drop_duplicates(subset='movie_title')

    # Remove movie link
    filtered_data.drop('movie_imdb_link', axis=1, inplace=True)

    # Remove NaN values and prepare to fill :
    filtered_data.dropna(axis=0,
                         subset=['director_name', 'num_critic_for_reviews', 'duration', 'director_facebook_likes',
                                 'actor_3_facebook_likes', 'actor_2_name', 'actor_1_facebook_likes', 'actor_1_name',
                                 'actor_3_name', 'facenumber_in_poster', 'num_user_for_reviews', 'language',
                                 'country', 'actor_2_facebook_likes', 'plot_keywords', 'color'], inplace=True)

    # replace NaN values with median values / values that appear the most
    features_to_fill_with_most_frequent = ["color", "content_rating"]
    fill_missing_values_with_most_frequent(filtered_data, features_to_fill_with_most_frequent)

    features_to_fill_with_median = ["aspect_ratio", "budget", "gross"]
    fill_missing_values_with_median(filtered_data, features_to_fill_with_median)

    # Save the filtered data to a new CSV file
    filtered_file_path = "./data/filtered_movie_metadata.csv"
    filtered_data.to_csv(filtered_file_path, index=False)

    print(f"Filtered data saved to {filtered_file_path}")


# add our features as columns to the data
def merge_our_features():
    main_data = pd.read_csv("./data/filtered_movie_metadata.csv")
    main_data['movie_title'] = main_data['movie_title'].str.strip()
    # Load the feature files
    new_features = pd.read_csv('./data/New_features.csv')
    avg_sent_feature = pd.read_csv('data/avg_sentiment_feature.csv')
    critic_reviews_number = pd.read_csv('./data/critic_reviews_number.csv')

    # Merge the feature files with the main data on 'movie_title'
    merged_data = main_data
    merged_data = merged_data.merge(new_features, on='movie_title', how='left')
    merged_data = merged_data.merge(avg_sent_feature, on='movie_title', how='left')
    merged_data = merged_data.merge(critic_reviews_number, on='movie_title', how='left')

    output_file_path = './data/expanded_movie_metadata.csv'
    # Save the merged data to a new file
    merged_data.to_csv(output_file_path, index=False)

    print(f"Features successfully merged and saved to {output_file_path}")


# clean the expanded dataset. Here we had two options:
# option 1- Remove all rows(samples) with no movie reviews. This resulted in the loss of about 600 samples
# option 2- Fill the NaN values with the median of the feature. This maintained the number of samples
# Performance: option 1 yielded better prediction performances, so we are going to default to it.
def clean_expanded_data(fillNaN: bool = False):
    merged_file_path = "./data/expanded_movie_metadata.csv"
    merged_data = pd.read_csv(merged_file_path)

    filtered_data = merged_data

    if fillNaN:
        features_to_fill = ["Avg_sentiment", "Avg_words", "Avg_letters", "Numbers_freq", "Capitals_freq", "Punc_freq",
                            "Num Of Reviews"]

        fill_missing_values_with_median(filtered_data, features_to_fill)

    else:
        # Remove rows where 'Avg_sentiment' is NaN ( Movies with no reviews),
        # this also drop all rows where our other derived features are NaN (except Num Of Reviews)
        filtered_data.dropna(subset=['Avg_sentiment'], inplace=True)
        filtered_data.dropna(subset=['Num Of Reviews'], inplace=True)

    # Save the filtered data to a new CSV file
    filtered_file_path = "./data/filtered_expanded_movie_metadata.csv"
    filtered_data.to_csv(filtered_file_path, index=False)

    print(f"Filtered expanded data saved to {filtered_file_path}")


def create_clean_dataset(fillNaN: bool = True):
    print("****************************")
    print("Creating clean_dataset...")
    clean_source_data()
    merge_our_features()
    clean_expanded_data(fillNaN)
