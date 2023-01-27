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

    # @classmethod
    # def get_data_transformer_object(cls)->Pipeline:
    #     try:
    #         Standard_Scaler=StandardScaler()
    #         # Standard_Scaler= ColumnTransformer([
    #         #     ("Numerical_column",StandardScaler(),[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])])

    #         # Ordinal_Encoder=ColumnTransformer([("Categorical_column",Ordinal_Encoder(),[1,2,3])],remainder='passthrough')

    #         pipeline=Pipeline(steps=[('scaler',Standard_Scaler)])

    #         return pipeline
    #     except Exception as e:
    #         raise CreditException(e,sys)
                                    


            # Standard_Scaler=StandardScaler()
            # Ordinal_Encoder=OrdinalEncoder()

            # #numerical pipeline
            # logging.info("Creating Numerical Pipeline")
            # numerical_pipeline=Pipeline([('feature_scaling',StandardScaler())],rema)

            # #Categorical pipeline
            # logging.info("Creating Categorical Pipeline")
            # categorical_pipeline=Pipeline([('categorical_encoder', OrdinalEncoder())])

            # logging.info("Combing both numerical and categorical pipeline")
            # column_pipeline=ColumnTransformer([("numerical_pipeline",numerical_pipeline,train_numeric_columns),("categorical_pipeline",categorical_pipeline,train_categorical_columns)])



    

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:

        try:
            logging.info("Reading Train And Test File")
            train_df =pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(train_df.shape)
            logging.info(test_df.shape)

            logging.info("Spliting Train Into X_train And y_train")
            X_train = train_df.drop(TARGET_COLUMN,axis=1)
            y_train = train_df[TARGET_COLUMN]

            logging.info("Spliting Test Into X_test And y_test")
            X_test = test_df.drop(TARGET_COLUMN,axis=1)
            y_test = test_df[TARGET_COLUMN]
     
            # logging.info("define numerical Columns")
            # # define numerical columns
            # logging.info("Train_Numeric")
            # train_numeric_columns = X_train.select_dtypes(include=['int64','float64']).columns
            # logging.info("Test_Numeric")
            # test_numeric_columns = X_test.select_dtypes(include=['int64','float64']).columns
            # numeric__train_df=train_df[train_numeric_columns]
            # numeric_test_df=test_df[test_numeric_columns]


            # logging.info("define categorical columns")
            # # define categorical columns
            # train_categorical_columns = X_train.select_dtypes(include=['object','category']).columns
            # logging.info("Test_Categorical")
            # test_categorical_columns = X_test.select_dtypes(include=['object','category']).columns
            # logging.info("Train_Categorical")
            # categorical_train_df=train_df[train_categorical_columns]

            # logging.info("Test_Categorical")
            # categorical__test_df=test_df[test_categorical_columns]    
            # # print columns
            # logging.info('We have {} Train numerical features : {}'.format(len(train_numeric_columns), train_numeric_columns))
            # logging.info('We have {} Test numerical features : {}'.format(len(test_numeric_columns), test_numeric_columns))
            # logging.info('\nWe have {} Train categorical features : {}'.format(len(train_categorical_columns), train_categorical_columns))
            # logging.info('\nWe have {} Test categorical features : {}'.format(len(test_categorical_columns), test_categorical_columns))

            # #numerical pipeline
            # logging.info("Creating Numerical Pipeline")
            # numerical_pipeline=Pipeline([('feature_scaling',StandardScaler())])

            # #Categorical pipeline
            # logging.info("Creating Categorical Pipeline")
            # categorical_pipeline=Pipeline([('categorical_encoder', OrdinalEncoder())])

            # logging.info("Combing both numerical and categorical pipeline")
            # column_pipeline=ColumnTransformer([
            #     ("categorical_pipeline",categorical_pipeline,train_categorical_columns),
            #     ("numerical_pipeline",numerical_pipeline,train_numeric_columns)])
            # Ordinal_encoder=OneHotEncoder()

            # X_train=Ordinal_encoder.fit_transform(X_train[['SEX', 'EDUCATION', 'MARRIAGE']])
            # X_test=Ordinal_encoder.fit_transform(X_test[['SEX', 'EDUCATION', 'MARRIAGE']])
            # # Standard_Scaler=StandardScaler()

            # transformation_pipleine = DataTransformation.get_data_transformer_object()
            logging.info("Applying Standard Scaler")
            scaler=StandardScaler()
            scaler.fit(X_train)

            X_train=scaler.transform(X_train)
            X_test=scaler.transform(X_test)




            logging.info("Label encoder for Target encoder")
            label_encoder=LabelEncoder()
            label_encoder.fit(y_train)

            y_train=label_encoder.transform(y_train)
            y_test=label_encoder.transform(y_test)
            

            # logging.info("column_pipeline transformation for Input variable")
            # column_pipeline.fit(X_train)

            # X_train=column_pipeline.transform(X_train)
            # X_test=column_pipeline.transform(X_test)
            


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

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,obj=scaler)

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



    