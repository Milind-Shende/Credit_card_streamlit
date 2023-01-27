import os,sys
import pickle
ROOT_DIR = os.getcwd()
MODEL_DIR_NAME = "saved_models"
SAVED_ZERO_FILE="0"
MODEL_FILE_DIR ="model"
MODEL_FILE_NAME = "model.pkl"
MODEL_DIR = os.path.join(ROOT_DIR, MODEL_DIR_NAME,SAVED_ZERO_FILE,MODEL_FILE_DIR,MODEL_FILE_NAME)
print(MODEL_DIR)

model=pickle.load(open("MODEL_DIR","rb"))
print(model)