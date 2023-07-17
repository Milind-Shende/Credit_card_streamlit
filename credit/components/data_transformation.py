from credit import utils
from credit.entity import config_entity
from credit.entity import artifact_entity
from credit.logger import logging
from credit.exception import CreditException
import os,sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import numpy as np
from credit.constant import TARGET_COLUMN


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
            logging.info(train_df.shape)
            logging.info(test_df.shape)

            numeric_features = ['LIMIT_BAL','AGE', 'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
            logging.info(f"Numerical Column{numeric_features}")


            categorical_features = ['SEX','EDUCATION','MARRIAGE'] 
            logging.info(f"CategoricalColumn{categorical_features}")
            

            logging.info("Numerical and Categorical Pipeline Transformation")
            numeric_transformer= Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[ ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))])

            logging.info("Numerical and Categorical Column Transformation")
            transformer = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                            ('cat', categorical_transformer, categorical_features)])

           

            logging.info("Spliting Train Into X_train And y_train")
           
            X_train = train_df.drop(TARGET_COLUMN,axis=1)
            y_train = train_df[TARGET_COLUMN]

            logging.info(X_train.shape)
            logging.info(y_train.shape)

            logging.info("Transforming X_train Column Transform")
            X_train=transformer.fit_transform(X_train)
            

            logging.info("Spliting Test Into X_test And y_test")
            
            X_test = test_df.drop(TARGET_COLUMN,axis=1)
            y_test = test_df[TARGET_COLUMN]

            logging.info(X_test.shape)
            logging.info(y_test.shape)

            logging.info("Transforming X_train Column Transform")
            X_test=transformer.transform(X_test)

            logging.info("Label encoder for Target encoder")
            label_encoder=LabelEncoder()
            label_encoder.fit(y_train)

            y_train=label_encoder.transform(y_train)
            y_test=label_encoder.transform(y_test)

            logging.info("Re-sampling the dataset using SMOTE method")
            smote = SMOTETomek(random_state=42)
            X_train, y_train = smote.fit_resample(X_train,y_train)
            X_test, y_test = smote.fit_resample(X_test,y_test)
            logging.info(X_train.shape)
            logging.info(y_train.shape)
            logging.info(X_test.shape)
            logging.info(y_test.shape)
                

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

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,obj=transformer)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,obj=label_encoder)

            
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path= self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e, sys) 



    