from scripts.preprocess import Preprocess
from scripts.sentiment_analysis import sentiment_analysis
from scripts.sentiment_creation import create_sentiment
from scripts.wordCloud import generate_wordclouds
from utils.data_utils import delete_data, open_processed_files
import config
from scripts.LDA_Model import Model

if __name__ == "__main__":

    preprocess = Preprocess(path=rf'{config.raw_path}/')
    filename = preprocess.pipeline()
    all_df = open_processed_files(path = rf"{config.processed_path}/")
    print("print:", all_df[0])
    print(all_df[0].head())

    topics = 4
    model = Model(all_df[0],topics)
    model.CreateLdaModel()
    model.visualize()

    # Generate and save word clouds for each topic
    generate_wordclouds(model.lda_model, topics)

    create_sentiment()

    keywords = ['health', 'vaccine', 'policy', 'lockdown']
    sentiment_analysis(keywords)

    delete_data()