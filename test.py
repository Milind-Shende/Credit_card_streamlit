from credit.logger import logging
from credit.exception import CreditException
from credit.utils import get_collection_as_dataframe
from credit.entity import config_entity
from credit.components.data_ingestion import DataIngestion
import os,sys



if __name__=="__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
    except Exception as e:
        print(e)