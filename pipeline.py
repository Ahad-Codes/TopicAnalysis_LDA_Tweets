from scripts.preprocess import Preprocess
from scripts.sentiment_analysis import sentiment_analysis
from scripts.sentiment_creation import create_sentiment
from scripts.monthly_sentiment_analysis import monthly_sentiment_analysis
from scripts.wordCloud import generate_wordclouds
from utils.data_utils import delete_data, open_processed_files
import config
from scripts.LDA_Model import Model
import os
import gensim
import click

@click.command()
@click.option('--raw_file_path', default=f'{config.raw_path}', help='Path to the raw files directory.')
@click.option('--num_topics', default=4, help='Number of topics for the LDA model.')
@click.option('--save_results_path', default='./results', help='Path to save the results and models.')
def main(raw_file_path, num_topics, save_results_path):

    config.topics = num_topics
    config.raw_path = raw_file_path

    print(f'Number of topics: {config.topics}')
    print(f'Raw file path: {config.raw_path}')
    print(f'Save results path: {save_results_path}')

    print('\nStarting the pipeline...')

    preprocess = Preprocess(path=rf'{config.raw_path}/')
    filename = preprocess.pipeline()
    all_df = open_processed_files(path = rf"{config.processed_path}/")
    
    if all_df is None:
        print("No files to process")
    else:
        print("print:", all_df[0])
        print(all_df[0].head())


    if os.path.exists(config.model_path):
        print("Loading saved model...")
        loaded_model = gensim.models.LdaMulticore.load(config.model_path)
        model = Model(all_df[0],config.topics,loaded_model)
        lda_model = model.CreateLdaModel()
        print("Loaded")
    else:
        
        model = Model(all_df[0],config.topics)
        lda_model = model.CreateLdaModel()
        lda_model.save(config.model_path)
        print("Model saved successfully!")


    model.visualize()

    #Generate and save word clouds for each topic
    generate_wordclouds(model.lda_model, topics)

    create_sentiment()

    keywords = ['health', 'vaccine', 'policy', 'lockdown', 'economy', 'distancing']
    sentiment_analysis(keywords)

    monthly_sentiment_analysis()

    delete_data()

if __name__ == "__main__":

    main()

    