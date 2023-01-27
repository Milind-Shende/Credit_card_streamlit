from flask import Flask,render_template,request,app,jsonify,url_for
import pickle
import pandas as pd
import numpy as np
import os,sys
from credit.exception import CreditException
from credit.logger import logging
import requests

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "saved_models"
SAVED_ZERO_FILE="0"
MODEL_FILE_DIR ="model"
MODEL_FILE_NAME = "model.pkl"
TRANSFORMER_FILE_DIR="transformer"
TRANSFORMER_FILE_NAME="transformer.pkl"
TARGET_ENCODER_FILE_DIR="target_encoder"
TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,MODEL_FILE_DIR,MODEL_FILE_NAME)
# print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TRANSFORMER_FILE_DIR,TRANSFORMER_FILE_NAME)
# print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)
#Load The Model
model=pickle.load(open(MODEL_DIR,"rb"))
transfomer=pickle.load(open(TRANSFORMER_DIR,"rb"))

app=Flask(__name__,template_folder='templates')

@app.route('/',methods=['GET','POST'])
def Home():
    try:
        return render_template("index.html")
    except Exception as e:
        raise CreditException(e,sys)

@app.route("/predict",methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
   
            categorical_value = request.form['SEX','EDUCATION','MARRIAGE']
            numerical_value = request.form['LIMIT_BAL','AGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                                            'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                                            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
        
        # Create a dataframe with the input values
            input_data = pd.DataFrame({
                "categorical_value": [categorical_value],
                "numerical_value": [numerical_value]
            })

            int_features=[input_data.values()]
            final=[np.array(int_features).reshape(1,-1)]
            prediction=model.predict(final)[0][1]

            return render_template('index.html',prediction_text="Probability Of Default Is== {}".format(prediction))

        else:
            return render_template('index.html')
    except Exception as e:
        raise CreditException(e,sys)

if __name__=="__main__":
    app.run(debug=True)
