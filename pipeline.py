from scripts.preprocess import Preprocess
from utils.data_utils import delete_data, open_processed_files
import config
from scripts.LDA_Model import Model

if __name__ == "__main__":
    delete_data()

    preprocess = Preprocess(path=rf'{config.raw_path}/')
    filename = preprocess.pipeline()
    all_df = open_processed_files(path = rf"{config.processed_path}/")
    print(all_df[0].head())

    topics = 4
    model = Model(all_df[0],topics)
    model.CreateLdaModel()
    model.visualize()

    