from credit import utils
from credit.entity import config_entity
from credit.entity import artifact_entity
from credit.logger import logging
from credit.exception import CreditException
import os,sys
import pandas as pd
import numpy as mp
from sklearn.model_selection import train_test_split


class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name, 
            collection_name=self.data_ingestion_config.collection_name)

            # logging.info("Dropping the unique ID column")
            # #Dropping the unique ID column
            # df=df.drop(columns='ID')

            # logging.info("Replacing the column names in the dataset")
            # #Replacing the column names in the dataset
            # df.rename(columns={'PAY_0':'PAY_1'},inplace=True)
            # df.rename(columns={'default.payment.next.month':'Default_Prediction'},inplace=True)

            logging.info("Replacing values in the features with their Actual names")
            #Replacing values in the features with their Actual names
            df['SEX'] = df['SEX'].replace({1:'male', 2:'female'})

            logging.info("Here, we have some other values in Education like {0,4,5,6} which are not in first 3 categories")
            logging.info("So, we are replacing all with section 4")
            #Here, we have some other values in Education like {0,4,5,6} which are not in first 3 categories.
            #So, we are replacing all with section 4
            df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4})
            df['EDUCATION']=df['EDUCATION'].replace({1:'graduate school',2:'university',3:'high school',4:'others'})

            logging.info("Doing the transformation to Marriage columns")
            #Doing the transformation to Marriage columns
            df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
            df['MARRIAGE']=df['MARRIAGE'].replace({1:'married',2:'single',3:'others'})  


            logging.info("Replacing 1 value wit Yes and 0 value with No")
            df['Default']=df['Default'].replace({1:"Yes",0:"No"})  

            # logging.info("We are replacing the values of all PAY_X features -1,-2 with 0.")
            # #We are replacing the values of all PAY_X features -1,-2 with 0.
            # for i in range(1,7):
            #     field='PAY_'+str(i)
            #     df[field]=df[field].replace({-1:0})
            #     df[field]=df[field].replace({-2:0})  

            # #Dropping ['BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'] Because it has high correlation
            # df.drop(['BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis=1,inplace=True)      


            logging.info("Save data in feature store")
            #Save data in feature store
            logging.info("Create feature store folder if not available")
            #Create feature store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)


            logging.info("split dataset into train and test set")
            #split dataset into train and test set
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            
            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            logging.info(train_df.info())
            
            #Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CreditException(error_message=e, error_detail=sys)