from scrap_data import scrap_data
from derive_features import derive_features
from create_clean_dataset import create_clean_dataset
import experiments


def main():
    #scrap_data()

    #derive_features()  # fine-tunes a BERT model if no fine-tuned model is found

    #create_clean_dataset(fillNaN=False)
    #experiments.basic_experiment("./experiments_results/basic_experiment_logs")

    #features = ["Avg_sentiment", "Avg_words", "Avg_letters", "Numbers_freq", "Capitals_freq", "Punc_freq"
    #    , 'Num Of Reviews']
    #experiments.feature_impact_multiple_experiments("./experiments_results/multiple_features_logs", features)

    #experiments.derived_features_sentiment_prediction_experiment("./experiments_results/sentiment_prediction_logs")
    #experiments.avg_sentiment_insights_experiment("./experiments_results/insights_logs")

    #experiments.multiple_genre_experiments("./experiments_results/multiple_genres_logs")
    experiments.multiple_year_range_experiment("./experiments_results/multiple_year_range_experiment")
    print("\n\nDone.")
    print("****************************")


main()
