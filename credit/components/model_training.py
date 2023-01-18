from credit.logger import logging
from credit.exception import CreditException
from credit import utils
from credit.entity import config_entity
from credit.entity import artifact_entity
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix,roc_curve,auc
import os,sys



class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):

        try:
            self.model_trainer_config =model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise CreditException(e,sys)

    def train_model(self,X,y):
        try:
            xgboost_clf=XGBClassifier()
            xgboost_clf.fit(X,y)
            return xgboost_clf
        except Exception as e:
            raise CreditException(e,sys)
    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Train the model")
            model=self.train_model(X=X_train,y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train=model.predict(X_train)
            f1_train_score=f1_score(y_true=y_train,y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test=model.predict(X_test)
            f1_test_score=f1_score(y_true=y_test,y_pred=yhat_test)

            #Check for overfitting or underfitting or expected score
            """Overfitting mean good accuracy on training score but not getting good accuracy on test score

               Underfitting means we are not getting good accuracy on both train and test accuracy

               Expected score means which decided by us.
            
            """
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info(f"Checking if model is overfitting or not")
            diff=abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test Score Difference:{diff} is more than overfitting threshold{self.model_trainer_config.overfitting_threshold}")


            logging.info(f"Saving model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)


            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
            f1_train_score=f1_train_score,f1_test_score=f1_test_score)
            logging.info(f"model_trainer_artifact")
            return model_trainer_artifact
        except Exception as e:
            raise CreditException(e,sys)