import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

import config
import glob

# Download VADER lexicon if you haven't already
#nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to apply VADER sentiment analysis on tweets and save results
def analyze_sentiment(file_name):
    # Load the CSV data
    data = pd.read_csv(file_name)

    # Replace NaN values in 'original_text' with empty strings
    data['original_text'] = data['original_text'].fillna('')

    # Apply VADER sentiment analysis on each tweet
    sentiments = []
    for tweet in data['original_text']:
        sentiment_score = analyzer.polarity_scores(tweet)
        sentiments.append(sentiment_score)

    # Convert the list of sentiment scores into a DataFrame
    sentiment_df = pd.DataFrame(sentiments)

    # Merge the original data with sentiment scores
    data_with_sentiment = pd.concat([data, sentiment_df], axis=1)

    # Save the data with sentiment scores to a new CSV file
    output_file = f"{file_name.split('.')[0]}_with_sentiment.csv"
    data_with_sentiment.to_csv(output_file, index=False)
    print(f"Saved sentiment analysis results for {file_name} as {output_file}")


def create_sentiment():
    path = rf"{config.processed_path}/"

    data_files = glob.glob(f"{path}*.csv")
    for file in data_files:
        analyze_sentiment(file)

    print("All processed data files analyzed for sentiment.")
