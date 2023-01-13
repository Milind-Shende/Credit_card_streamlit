import os,sys
from credit.exception import CreditException
from credit.logger import logging
from datetime import datetime

FILE_NAME = "Credit_Card.csv"


class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.art

        except Exception as e:
            raise CreditException(e,sys)

class DataIngentionConfig:...
class DataTrasformationConfig:...
class ModelTrainerConfig:...
class ModelEvaluationConfig:...
