import os,sys
import pickle
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
TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TRANSFORMER_FILE_DIR,TRANSFORMER_FILE_NAME)
print(MODEL_DIR)

model=pickle.load(open(MODEL_DIR,"rb"))
transfomer=pickle.load(open(TRANSFORMER_DIR,"rb"))
print(model)
print(transfomer)