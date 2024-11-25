BERT_fine_tuning_cuda.py: Cleans up an existing dataset and uses it to fine-tunes a pretrained BERT model. This file is used in the derive features process.

scrap_data.py: Responsible for scrapping data from IMDb. This data is used in the derive features process.

derive_features.py: Dervies new features for a movie based on the critic reviews.

create_clean_dataset.py: Creates clean datasets that will be used in the prediction tasks by the models.

experiments.py: Contains the implementation of the experiments code, including models training and results logging.

main.py: Implements the design of the system by calling the appropriate methods for each part (data scraping -> derive features -> create dataset -> run experiments)

plotting.py: Creates plots based on the results of the experiments.

utils.py: Contains various utility functions used by the other files.

interactive_utils.py: Contains the utility functions for the interactive program.
interactive_ui.py: Contains the implementation of the prediction interactive program with the UI.
