from credit import utils
from credit.entity import config_entity
from credit.entity import artifact_entity
from credit.logger import logging
from credit.exception import CreditException
import os,sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
import numpy as np


class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTrasformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise CreditException(e, sys)


    

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:

        try:
            logging.info("Reading Train And Test File")
            train_df =pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Spliting Train Into X_train And y_train")
            X_train = train_df.iloc[:,:-1]
            y_train = train_df.iloc[:,-1]

            logging.info("Spliting Test Into X_test And y_test")
            X_test = test_df.iloc[:,:-1]
            y_test = test_df.iloc[:,-1]
     
            logging.info("define numerical Columns")
            # define numerical columns
            logging.info("Train_Numeric")
            train_numeric_columns = X_train.select_dtypes(include=['int64','float64']).columns
            logging.info("Test_Numeric")
            test_numeric_columns = X_test.select_dtypes(include=['int64','float64']).columns
            numeric__train_df=train_df[train_numeric_columns]
            numeric_test_df=test_df[test_numeric_columns]


            logging.info("define categorical columns")
            # define categorical columns
            train_categorical_columns = X_train.select_dtypes(include=['object','category']).columns
            logging.info("Test_Categorical")
            test_categorical_columns = X_test.select_dtypes(include=['object','category']).columns
            logging.info("Train_Categorical")
            categorical_train_df=train_df[train_categorical_columns]

            logging.info("Test_Categorical")
            categorical__test_df=test_df[test_categorical_columns]    
            # print columns
            logging.info('We have {} Train numerical features : {}'.format(len(train_numeric_columns), train_numeric_columns))
            logging.info('We have {} Test numerical features : {}'.format(len(test_numeric_columns), test_numeric_columns))
            logging.info('\nWe have {} Train categorical features : {}'.format(len(train_categorical_columns), train_categorical_columns))
            logging.info('\nWe have {} Test categorical features : {}'.format(len(test_categorical_columns), test_categorical_columns))

            #numerical pipeline
            logging.info("Creating Numerical Pipeline")
            numerical_pipeline=Pipeline([('feature_scaling',StandardScaler())])

            #Categorical pipeline
            logging.info("Creating Categorical Pipeline")
            categorical_pipeline=Pipeline([('categorical_encoder', OrdinalEncoder())])

            logging.info("Combing both numerical and categorical pipeline")
            column_pipeline=ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,train_numeric_columns),
                ("categorical_pipeline",categorical_pipeline,train_categorical_columns)])

            train_df_X=column_pipeline.fit_transform(X_train)
            test_df_X=column_pipeline.transform(X_test)


            logging.info("Re-sampling the dataset using SMOTE method")
            smote = SMOTETomek(sampling_strategy=0.5)
            X_train, y_train = smote.fit_resample(train_df_X,y_train)
            X_test, y_test = smote.fit_resample(test_df_X,y_test)

            logging.info("train and test array concatenate")
            #train and test array
            train_arr = np.c_[X_train, y_train ]
            test_arr = np.c_[X_test, y_test]
            
            logging.info("Saved train and test array to save_numpy_array_data")
            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e, sys) 



    