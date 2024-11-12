import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if you haven't already
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# List of files to process
file_names = ['data/2020_8_dataset.csv', 'data/2020_9_dataset.csv', 'data/2020_10_dataset.csv']

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

# Run sentiment analysis on each file
for file_name in file_names:
    analyze_sentiment(file_name)
