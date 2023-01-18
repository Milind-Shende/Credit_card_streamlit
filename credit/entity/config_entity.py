import os,sys
from credit.exception import CreditException
from credit.logger import logging
from datetime import datetime

FILE_NAME = "defaulter.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
ORDINAL_ENCODER_OBJECT_FILE_NAME = "ordinal_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise CreditException(e,sys)     


class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="CreditCard"
            self.collection_name="Default"
            #Below code will create a file with name "data_ingestion in artifact folder
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception  as e:
            raise CreditException(e,sys)     

    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise CreditException(e,sys)



class DataTrasformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        #Below code will create a file with name "data_transformation in artifact folder
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path = os.path.join(self.data_transformation_dir,TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformed_test_path = os.path.join(self.data_transformation_dir,TEST_FILE_NAME.replace("csv","npz"))
        self.ordinal_encoder_path = os.path.join(self.data_transformation_dir,"ordinal_encoder",ORDINAL_ENCODER_OBJECT_FILE_NAME)





class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir,"model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,'model',MODEL_FILE_NAME)
        self.expected_score=0.5
        self.overfitting_threshold = 0.1


class ModelEvaluationConfig:...
