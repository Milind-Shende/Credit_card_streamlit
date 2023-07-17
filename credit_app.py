import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "saved_models"
SAVED_ZERO_FILE="0"
MODEL_FILE_DIR ="model"
MODEL_FILE_NAME = "model.pkl"
TRANSFORMER_FILE_DIR="transformer"
TRANSFORMER_FILE_NAME="transformer.pkl"
# TARGET_ENCODER_FILE_DIR="target_encoder"
# TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,MODEL_FILE_DIR,MODEL_FILE_NAME)
# print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TRANSFORMER_FILE_DIR,TRANSFORMER_FILE_NAME)
# print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

# TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)

# Load the Model.pkl, Transformer.pkl and Target.pkl
model=pickle.load(open(MODEL_DIR,"rb"))
# print(model)
transfomer=pickle.load(open(TRANSFORMER_DIR,"rb"))
# print(transfomer)


# About page
def about_page():
    

    st.title('Credit Card Default Prediction: A Machine Learning Approach')
    st.write("This is a machine learning model for predicting credit card default. The model uses historical credit card data to determine the likelihood of a credit card holder defaulting on their payments.This is a classification model for a most common dataset, Credit Card defaulter prediction. Prediction of the next month credit card defaulter based on demographic and last six months behavioral data of customers. :notebook:")
    
    st.title("Dataset Source")
    st.write("In our dataset we have 25 columns 30000 Rows which reflect various attributes of the customer. The target column is default.payment.next.month , which reflects whether the customer defaulted or not. Our aim is to predict the probability of default given the payment history of the customer. I have built my model using a public dataset available on kaggle.")    
    st.write(" :link: Kaggle link :- https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset")
    st.write(" :link: UCI Repository :- https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")

def author():
    st.title("About Me")
    st.write("Data Science with Strong knowledge in Data analysis and Machine Learning Skills With 2 Years of experience in Banking Operation And Including PGDM in Finance Management. looking out for the opportunity in the field of Data Science. I am actively seeking an environment where my passion for Data Science and my proficiency in data analysis and machine learning can be nurtured and challenged.")
    st.write(":blond-haired-man:Name:-Milind Shende")
    st.write(":calling: MObile No:-+91 9420699550")
    st.write(":e-mail: E-mail:-milind.shende24rediffmail.com")
    st.write(":link:Github:-https://github.com/Milind-Shende/Credit_card_streamlit.git")
    st.write(":link:Linkedin:-https://www.linkedin.com/in/milind-shende-542314265/")
    

# Main prediction page
def prediction_page():
    # Title and input fields
    st.title('Credit Card Default Predication')
    st.subheader('Customer Information')
    LIMIT_BAL = st.number_input('LIMIT_BAL', min_value=10000.0, max_value=800000.0, step=1000.0)
    SEX = st.selectbox('Gender', ('male', 'female'))
    EDUCATION = st.selectbox('EDUCATION', ('university', 'graduate school','high school','others'))
    MARRIAGE = st.selectbox('MARRIAGE', ('single', 'married'))
    AGE = st.number_input('AGE', min_value=21.0, max_value=79.0, step=1.0)
    PAY_1 = st.number_input('REPAYMENT_STATUS_SEPT', min_value=0.0, max_value=8.0, value=0.0)
    PAY_2 = st.number_input('REPAYMENT_STATUS_AUGUST', min_value=0.0, max_value=8.0, value=0.0)
    PAY_3 = st.number_input('REPAYMENT_STATUS_JULY', min_value=0.0, max_value=8.0, value=0.0)
    PAY_4 = st.number_input('REPAYMENT_STATUS_JUNE', min_value=0.0, max_value=8.0, value=0.0)
    PAY_5 = st.number_input('REPAYMENT_STATUS_MAY', min_value=0.0, max_value=8.0, value=0.0)
    PAY_6 = st.number_input('REPAYMENT_STATUS_APRIL', min_value=0.0, max_value=8.0, value=0.0)
    BILL_AMT1 = st.number_input('BILL_AMT_SEPT', min_value=-165580.0, max_value=964511.0, value=0.0, step=1000.0)
    BILL_AMT2 = st.number_input('BILL_AMT_AUGUST', min_value=-69777.0, max_value=983931.0, value=0.0, step=1000.0)
    BILL_AMT3 = st.number_input('BILL_AMT_JULY', min_value=-157264.0, max_value=1664089.0, value=0.0, step=1000.0)
    BILL_AMT4 = st.number_input('BILL_AMT_JUNE', min_value=-170000.0, max_value=891586.0, value=0.0, step=1000.0)
    BILL_AMT5 = st.number_input('BILL_AMT_MAY', min_value=-81334.0, max_value=927171.0, value=0.0, step=1000.0)
    BILL_AMT6 = st.number_input('BILL_AMT_APRIL', min_value=-339603.0, max_value=961664.0, value=0.0, step=1000.0)
    PAY_AMT1 = st.number_input('PAY_AMT_SEPT', min_value=0.0, max_value=873552.0, value=0.0, step=1000.0)
    PAY_AMT2 = st.number_input('PAY_AMT_AUGUST', min_value=0.0, max_value=1684259.0, value=0.0, step=1000.0)
    PAY_AMT3 = st.number_input('PAY_AMT_JULY', min_value=0.0, max_value=896040.0, value=0.0, step=1000.0)
    PAY_AMT4 = st.number_input('PAY_AMT_JUNE', min_value=0.0, max_value=621000.0, value=0.0, step=1000.0)
    PAY_AMT5 = st.number_input('PAY_AMT_MAY', min_value=0.0, max_value=426529.0, value=0.0, step=1000.0)
    PAY_AMT6 = st.number_input('PAY_AMT_APRIL', min_value=0.0, max_value=528666.0, value=0.0, step=1000.0)

    
    
     
    # Prediction button
    if st.button('Predict'):
        try:
            # Preprocess the input features
            input_data = {
                'LIMIT_BAL':[LIMIT_BAL],
                'SEX':[SEX],
                'EDUCATION':[EDUCATION],
                'MARRIAGE':[MARRIAGE],
                'AGE':[AGE],
                'PAY_1':[PAY_1],
                'PAY_2':[PAY_2],
                'PAY_3':[PAY_3],
                'PAY_4':[PAY_4],
                'PAY_5':[PAY_5],
                'PAY_6':[PAY_6],
                'BILL_AMT1':[BILL_AMT1],
                'BILL_AMT2':[BILL_AMT2],
                'BILL_AMT3':[BILL_AMT3],
                'BILL_AMT4':[BILL_AMT4],
                'BILL_AMT5':[BILL_AMT5],
                'BILL_AMT6':[BILL_AMT6],
                'PAY_AMT1':[PAY_AMT1],
                'PAY_AMT2':[PAY_AMT2],
                'PAY_AMT3':[PAY_AMT3],
                'PAY_AMT4':[PAY_AMT4],
                'PAY_AMT5':[PAY_AMT5],
                'PAY_AMT6':[PAY_AMT6],
                    
            }
        except Exception as e:
            st.error(f"Error occurred: {e}")
        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame(input_data)
        # Perform the transformation using the loaded transformer
        transformed_data = transfomer.transform(input_df)
        # Reshape the transformed data as a NumPy array
        input_arr = np.array(transformed_data)
        

        # Make the prediction using the loaded model
        prediction = model.predict(input_arr)
        st.subheader('Prediction')
        st.write(f'The predicted total charge is: {prediction[0]}')




# Create a dictionary with page names and their corresponding functions
pages = {
    'About': about_page,
    'Prediction': prediction_page,
    'Author':author
}

# Streamlit application
def main():
    # Sidebar navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', list(pages.keys()))

    # Display the selected page content
    pages[selected_page]()

# Run the Streamlit application
if __name__ == '__main__':
    main()