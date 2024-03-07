import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PROJECTML import logger
from PROJECTML.utils.common import load_bin


class CustomData:
    def __init__(
        self,
        Age: int,
        Gender: object,
        Dependent_count: int,
        Education: object,
        Marital_Status: object,
        Income: object,
        Card_Category: object,
        Months_on_book: int,
        Total_Relationship_Count: int,
        Months_Inactive: int,
        Contacts_Count: int,
        Credit_Limit: float,
        Total_Revolving_Bal: int,
        Total_Amt_Chng_Q4_Q1: float,
        Total_Trans_Amt: int,
        Total_Trans_Ct: int,
        Total_Ct_Chng_Q4_Q1: float,
        Avg_Utilization_Ratio: float,
    ):

        self.Age = Age
        self.Gender = Gender
        self.Dependent_count = Dependent_count
        self.Education = Education
        self.Marital_Status = Marital_Status
        self.Income = Income
        self.Card_Category = Card_Category
        self.Months_on_book = Months_on_book
        self.Total_Relationship_Count = Total_Relationship_Count
        self.Months_Inactive = Months_Inactive
        self.Contacts_Count = Contacts_Count
        self.Credit_Limit = Credit_Limit
        self.Total_Revolving_Bal = Total_Revolving_Bal
        self.Total_Amt_Chng_Q4_Q1 = Total_Amt_Chng_Q4_Q1
        self.Total_Trans_Amt = Total_Trans_Amt
        self.Total_Trans_Ct = Total_Trans_Ct
        self.Total_Ct_Chng_Q4_Q1 = Total_Ct_Chng_Q4_Q1
        self.Avg_Utilization_Ratio = Avg_Utilization_Ratio
        logger.info("Customer data stored inside the CustomerData class")

    def get_data_as_dataframe(self):

        self.preprocessor = joblib.load(Path('artifacts\data_transformation\preprocessor.joblib'))
        logger.info("This is the preprocessor pipeline: %s", self.preprocessor)

        try:
            customer_data_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Dependent_count": [self.Dependent_count],
                "Education": [self.Education],
                "Marital_Status": [self.Marital_Status],
                "Income": [self.Income],
                "Card_Category": [self.Card_Category],
                "Months_on_book": [self.Months_on_book],
                "Total_Relationship_Count": [self.Total_Relationship_Count],
                "Months_Inactive": [self.Months_Inactive],
                "Contacts_Count": [self.Contacts_Count],
                "Credit_Limit": [self.Credit_Limit],
                "Total_Revolving_Bal": [self.Total_Revolving_Bal],
                "Total_Amt_Chng_Q4_Q1": [self.Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [self.Total_Trans_Amt],
                "Total_Trans_Ct": [self.Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [self.Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [self.Avg_Utilization_Ratio]
            }




            self.df = pd.DataFrame(customer_data_dict)
            logger.info("Dataframe created")
            logger.info(f"Dataframe values: {self.df}")

            # Print input data features
            print("Input Data Features:")
            print(self.df.columns.tolist())  # Print feature names
            print("Number of Features:", len(self.df.columns))  # Print number of features

            data_transform=self.preprocessor.transform(self.df)

            logger.info("Done with data_transformation")

            return data_transform

        except Exception as e:
            logger.error("Exception occurred in creating dataframe")
            raise e
        

        


class PredictionPipeline:
        def __init__(self):
            
            self.model = joblib.load(Path('artifacts\model_trainer\model.joblib'))
            logger.info("Model object loaded successfully: %s", self.model)

            print(self.model)
            
            #print("Pipeline Expected Features:")
            #print(self.preprocessor.named_transformers_)
            # Access the ColumnTransformer object within the preprocessor
            #column_transformer = self.preprocessor

            # Retrieve the names of the features after transformation
            #transformed_feature_names = column_transformer.get_feature_names_out()
            #print("Transformed Feature Names:")
            #print(transformed_feature_names)

            logger.info("Model object loaded successfully")

            
            
        
             
             
    
        def predict(self,data_transform):
            #data=self.preprocessor

            #self.df1=self.get_data_as_dataframe()

            
            #data = custom_data_instance.get_data_as_dataframe()
            
            
            #scaled_data=self.preprocessor.transform(data)
            prediction = self.model.predict(data_transform)
            logger.info("Model predicted the Data")
            logger.info(f"Input data: {data_transform}")
            logger.info(f"Predicted output: {prediction}")
            return prediction



    
