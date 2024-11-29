import pandas as pd
import glob
import matplotlib.pyplot as plt
import config


def load_sentiment_data():
    """Load all sentiment-analyzed files and combine them into a single DataFrame."""
    processed_path = rf"{config.processed_path}/"
    sentiment_files = glob.glob(f"{processed_path}*_with_sentiment.csv")

    if not sentiment_files:
        raise FileNotFoundError("No sentiment-analyzed files found. Run the sentiment analysis script first.")

    all_data = pd.DataFrame()
    for file in sentiment_files:
        data = pd.read_csv(file)
        all_data = pd.concat([all_data, data], ignore_index=True)

    return all_data


def calculate_average_sentiment_by_month(data):
    """Calculate the average sentiment by month."""
    # Ensure the 'date' column is in datetime format
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
    data = data.dropna(subset=['created_at'])  # Drop rows where date conversion failed

    # Group by year and month
    data['year_month'] = data['created_at'].dt.to_period('M')

    # Calculate average compound sentiment by month
    avg_sentiment = data.groupby('year_month')['compound'].mean().reset_index()
    avg_sentiment['year_month'] = avg_sentiment['year_month'].dt.to_timestamp()

    return avg_sentiment


def save_average_sentiment(avg_sentiment, output_file):
    """Save the average sentiment data to a CSV file."""
    avg_sentiment.to_csv(output_file, index=False)
    print(f"Average sentiment by month saved to {output_file}")


def plot_average_sentiment(avg_sentiment, output_image):
    """Plot and save the average sentiment by month."""
    plt.figure(figsize=(12, 6))
    plt.plot(avg_sentiment['year_month'], avg_sentiment['compound'], marker='o')
    plt.title('Average Sentiment by Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Sentiment (Compound)', fontsize=14)
    plt.grid()
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output_image)
    print(f"Sentiment graph saved to {output_image}")
    plt.show()


def monthly_sentiment_analysis():
    try:
        # Load the combined data
        data = load_sentiment_data()

        # Calculate average sentiment by month
        avg_sentiment = calculate_average_sentiment_by_month(data)

        # Define output file paths
        avg_sentiment_file = rf"{config.processed_path}/average_sentiment_by_month.csv"
        sentiment_plot_file = rf"{config.processed_path}/average_sentiment_by_month.png"

        # Save the results
        save_average_sentiment(avg_sentiment, avg_sentiment_file)

        # Plot and save the graph
        plot_average_sentiment(avg_sentiment, sentiment_plot_file)

        print("Process completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
