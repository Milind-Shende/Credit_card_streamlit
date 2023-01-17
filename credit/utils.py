#We Write Helper Function

import pandas as pd
from credit.logger import logging
from credit.exception import CreditException
import os,sys
from credit.constant import mongo_client
import numpy as np

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This Function return Collection as dataFrame

    =========================================================
    
    Params:
    database_name: database name
    collection_name: collection_name

    ==========================================================
    
    return Pandas dataFrame of a collection

    """
    try:
        logging.info(f"Reading data from database:{database_name} and collection:{collection_name}")
        df=pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns:{df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping coulumn:_id")
            df=df.drop("_id",axis=1)
        logging.info(f"Row and columns in df:{df.shape}")
        return df
    except Exception as e:
        raise CreditException(e , sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditException(e, sys) from e