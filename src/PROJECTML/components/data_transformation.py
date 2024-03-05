import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PROJECTML.entity.config_entity import DataTransformationConfig



# here i defined the component of DataTransformationConfig below
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def rename_columns(self):
        self.data=self.config.data_path
        self.data = pd.read_csv(self.config.data_path)
        old_names = self.data.columns
        new_names = ['Clientnum', 'Attrition', 'Age', 'Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income', 
             'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive', 'Contacts_Count', 
             'Credit_Limit', 'Total_Revolving_Bal','Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        self.data.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        logger.info("done with renaming the columns ")

    

    def data_transform(self):
        
        
        print("DataFrame columns:", self.data.columns)
        categorical_columns = ['Attrition','Gender','Education','Marital_Status','Income','Card_Category']
            
            # Select only categorical columns
        categorical_data = self.data[categorical_columns]
            
            # Initialize OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
            
            # Fit and transform categorical columns
        categorical_data_encoded = ordinal_encoder.fit_transform(categorical_data)
            
            # Replace categorical columns in the original DataFrame with encoded values
        self.data[categorical_columns] = categorical_data_encoded
        print(self.data.head(5))

        
        return self.data
      


    def create_preprocessing_pipeline(self):
        # Define ordinal encoding transformer for categorical features
        ordinal_encoder = OrdinalEncoder()

        # Define the ColumnTransformer to apply ordinal encoding only to categorical features
        #categorical_features = ['Gender', 'Education', 'Marital_Status', 'Income', 'Card_Category']
        preprocessor = ColumnTransformer(
            transformers=[
                ('ordinal_encoder', ordinal_encoder, [4,6,7,8,9])
            ],
            remainder='passthrough'  # Passthrough numerical features
        )

        #trf1 = ColumnTransformer(
        #[('ordinal_encode', OrdinalEncoder(), [1, 2, 6, 8])],  # Indices of categorical columns
            #remainder='passthrough'
        #)

        # Define the full preprocessing pipeline including the ColumnTransformer
        preprocessing_pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor)
            ]
        )
        logger.info("done creating the preprocessor.joblib")
        #joblib.dump(preprocessing_pipeline)
        joblib.dump(preprocessing_pipeline, os.path.join(self.config.root_dir, self.config.preprocessor_file))

    def correlation(self):
        
        # Find most important features relative to target Price
        print("Find most important features relative to Attrition-target by Correlation matrix ")
        #numeric_columns = self.data.select_dtypes(include=[np.number])
        corr = self.data.corr()
        corr.sort_values(["Attrition"], ascending = False, inplace = True)
        print(corr["Attrition"])

        plt.style.use('ggplot')
        sns.set_style('whitegrid')
        plt.subplots(figsize = (30,30))
        ## Plotting heatmap. Generate a mask for the upper triangle (taken from seaborn example gallery)
        mask = np.zeros_like(self.data.corr(), dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(self.data.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, mask=mask, center = 0, );
        plt.title("Heatmap of all the Features of Train data set", fontsize = 25);


       
                
    def feature_selection(self):
        features = ['Age', 'Gender', 'Dependent_count', 'Education','Marital_Status', 'Income', 'Card_Category','Months_on_book',
            'Total_Relationship_Count','Months_Inactive','Contacts_Count', 'Credit_Limit', 'Total_Revolving_Bal','Total_Amt_Chng_Q4_Q1', 
            'Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio','Attrition']
        self.data = pd.DataFrame(self.data, columns = features)
        logger.info("removed columns with very low correlation: Avg_Open_To_Buy")
        print(self.data.columns)
        logger.info("Done with feature selection")


    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

#here i have defined the tarin_test_split below for performing the train_test_split
    def train_test_spliting(self):
        #data = pd.read_csv(self.config.data_path) # this line helps us to read the data
        transformed_dataset=self.data
        transformed_dataset.to_csv(os.path.join(self.config.root_dir, "transformed_dataset.csv"),index=False)

        # Split the data into training and test sets. (0.75, 0.25) split.
        #train, test = train_test_split(self.data,test_size=0.33,random_state=25) # this line splits the data into train_test_split

        #train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False) # here it saves the train and test data in csv format inisde the artifacts-> transformation folder
        #test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        #logger.info("Splited data into training and test sets")
        #logger.info(train.shape) # this logs the information about that how many training and testing samples i have 
        #logger.info(test.shape)

        #print(train.shape)
        #print(test.shape)
        logger.info("Done with creating the transformed_data.csv file in the artifacts folder")
