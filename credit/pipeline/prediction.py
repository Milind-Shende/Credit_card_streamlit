from credit.exception import CreditException
from credit.logger import logging
from credit.predictor import ModelResolver
import pandas as pd
from credit.utils import load_object
import os,sys
from datetime import datetime
import numpy as np
PREDICTION_DIR="prediction"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df=pd.read_csv(input_file_path)
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
        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder=load_object(file_path=model_resolver.get_latest_target_encoder_path())

        cat_prediction=target_encoder.inverse_transform(prediction)

        df['prediction']=prediction
        df['cat_pred']=cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path=os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise CreditException(e,sys)