import pymongo
import pandas as pd
import json

#Provide the MongoDB Localhost URL to Connect Python to MongoDB.

client= pymongo.MongoClient("mongodb+srv://Milind2487:mili%232487@milind2487.olvhy.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH="E:\Defaulters\Credit-Card\CreditcardDefaulter.csv"
DATABASE_NAME="CreditCard"
COLLECTION_NAME="Defaulter"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert Dataframe to json format to dump this records into mongodb
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #Insert Converted json Record To MongoDB Database
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
