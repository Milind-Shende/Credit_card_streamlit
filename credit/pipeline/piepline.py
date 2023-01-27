from credit.exception import CreditException
from credit.logger import logging
from credit.predictor import ModelResolver
import pandas as pd
from credit.utils import load_object
import os,sys
from datetime import datetime
import numpy as np
PREDICTION_DIR="prediction"

class CreditcardDefaultPredictor:

    def prediction(input_file):
            logging.info(f"Reading file :{input_file}")
            model_resolver = ModelResolver(model_registry="saved_models")
            df=pd.read_csv(input_file)
            df['SEX'].replace({1:'male', 2:'female'})
            df['EDUCATION'].replace({0:4,5:4,6:4})
            df['EDUCATION'].replace({1:'graduate school',2:'university',3:'high school',4:'others'})
            df['MARRIAGE'].replace({0:3})
            df['MARRIAGE'].replace({1:'married',2:'single',3:'others'})  
            df['Default'].replace({1:"Yes",0:"No"})  
            #Validation
            logging.info(f"Loading transformer to transform dataset")
            transformer=load_object(file_path=model_resolver.get_latest_transformer_path())

            # input_feature_name=list(transformer.feature_name_in_)
            input_arr=transformer.transform(df)
            logging.info(f"Loading model to make prediction")
            model =load_object(file_path=model_resolver.get_latest_model_path())
            prediction=model.predict(input_arr)

            if(prediction==0):
                return "Not a Defaulter"
            else:
                return "Defaulter"
            
