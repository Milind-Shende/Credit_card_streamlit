from credit.logger import logging
from credit.exception import CreditException
from credit.utils import get_collection_as_dataframe
from credit.entity import config_entity
from credit.components.data_ingestion import DataIngestion
import os,sys
from credit.components.data_transformation import DataTransformation
from credit.components.model_training import ModelTrainer



if __name__=="__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        

        #data transformation
        data_transformation_config = config_entity.DataTrasformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
        data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()


        #model Trainer
        model_trainer_config=config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer =ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

    except Exception as e:
        print(e)