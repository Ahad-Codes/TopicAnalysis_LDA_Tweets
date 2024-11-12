import glob
import os
import pandas as pd

def delete_data(path = r"data/processed/"):

    data_files = glob.glob(f"{path}*.csv")
    for file in data_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
        
    print("All processed data files deleted.")

    return

def group_raw_files(path = r"data/raw/"):

    data_files = glob.glob(os.path.join(path, "*.csv"))
    all_df = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            all_df.append(df)
            print(f"Succesfully opened: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return pd.concat(all_df, ignore_index=True)

def open_processed_files(path = r"data/processed/"):

    data_files = glob.glob(os.path.join(path, "*.csv"))
    all_df = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            all_df.append(df)
            print(f"Succesfully opened: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return all_df
