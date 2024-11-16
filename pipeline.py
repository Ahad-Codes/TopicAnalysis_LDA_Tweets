from scripts.preprocess import Preprocess
from utils.data_utils import delete_data, open_processed_files
import config
from scripts.LDA_Model import Model

if __name__ == "__main__":
    preprocess = Preprocess(path=rf'{config.raw_path}\Covid-19 Twitter Dataset (Aug-Sep 2020).csv')
    filename = preprocess.pipeline()
    all_df = open_processed_files(path = rf"{config.processed_path}/")
    print(all_df[0].head())

    delete_data()

    topics = 10
    model = Model(all_df[0],topics)
    model.CreateLdaModel()
    model.visualize()