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
print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TRANSFORMER_FILE_DIR,TRANSFORMER_FILE_NAME)
print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

# TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)
#Load The Model
model=pickle.load(open(MODEL_DIR,"rb"))
print(model)
transfomer=pickle.load(open(TRANSFORMER_DIR,"rb"))
print(transfomer)
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
            
        inputs={
        'LIMIT_BAL': float(request.form['LIMIT_BAL']),
        'SEX': int(request.form['SEX']),
        'EDUCATION': int(request.form['EDUCATION']),
        'MARRIAGE': int(request.form['MARRIAGE']),
        'AGE': int(request.form['AGE']),
        'PAY_1': int(request.form['REPAYMENT_STATUS_SEPT']),
        'PAY_2':int(request.form['REPAYMENT_STATUS_AUGUST']),
        'PAY_3': int(request.form['REPAYMENT_STATUS_JULY']),
        'PAY_4': int(request.form['REPAYMENT_STATUS_JUNE']),
        'PAY_5': int(request.form['REPAYMENT_STATUS_MAY']),
        'PAY_6': int(request.form['REPAYMENT_STATUS_APRIL']),
        'BILL_AMT1':float(request.form['BILL_AMT_SEPT']),
        'BILL_AMT2':float(request.form['BILL_AMT_AUGUST']),
        'BILL_AMT3':float(request.form['BILL_AMT_JULY']),
        'BILL_AMT4':float(request.form['BILL_AMT_JUNE']),
        'BILL_AMT5':float(request.form['BILL_AMT_MAY']),
        'BILL_AMT6':float(request.form['BILL_AMT_APRIL']),
        'PAY_AMT1': float(request.form['PAY_AMT_SEPT']),
        'PAY_AMT2': float(request.form['PAY_AMT_AUGUST']),
        'PAY_AMT3': float(request.form['PAY_AMT_JULY']),
        'PAY_AMT4': float(request.form['PAY_AMT_JUNE']),
        'PAY_AMT5': float(request.form['PAY_AMT_MAY']),
        'PAY_AMT6': float(request.form['PAY_AMT_APRIL'])
        }
        print(inputs)
        # print(np.array(list(inputs.values())).reshape(1,-1))
        # input_arr=np.array(list(inputs.values())).reshape(1,24)
        df=transfomer.transform(np.array(list(inputs.values())).reshape(1,23))
        # [df.np.array().reshape(1,2)]
        prediction=model.predict(df)[0]
        print(prediction)
        

        return render_template('index.html',prediction_text="Probability Of Default Is== {}".format(prediction))

    except Exception as e:
        raise CreditException(e,sys)

if __name__=="__main__":
    app.run(debug=True)