#We Write Helper Function

import pandas as pd
from credit.logger import logging
from credit.exception import CreditException
import os,sys
from credit.constant import mongo_client


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