from scripts.preprocess import Preprocess
from utils.data_utils import delete_data, open_processed_files

preprocess = Preprocess(path=r'data\raw\Covid-19 Twitter Dataset (Aug-Sep 2020).csv')
preprocess.pipeline()

all_df = open_processed_files(path = r"data/processed/")

print(all_df[0].head())

delete_data()


# get data
# preprocess data
# call lda model
# extract results

