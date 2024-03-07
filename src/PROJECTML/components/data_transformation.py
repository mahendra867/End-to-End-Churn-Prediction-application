import os
from PROJECTML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from PROJECTML.entity.config_entity import DataTransformationConfig
from sklearn.compose import make_column_transformer


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        

    def rename_columns(self):
        self.data=self.config.data_path
        self.data = pd.read_csv(self.config.data_path)
        new_names = ['Clientnum', 'Attrition', 'Age', 'Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income',
                     'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive', 'Contacts_Count',
                     'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                     'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        self.data.columns = new_names
        self.data.info()
        logger.info("done with checking the self.data.info")
        #return self.data

    def feature_selection(self):
        columns_to_drop = ['Clientnum', 'Avg_Open_To_Buy']
        self.data.drop(columns=columns_to_drop, inplace=True)

        print(self.data.columns)
        self.data.head()

        y=self.data.isnull().sum()
        print(y)
        self.data.info()

    def create_preprocessing_pipeline(self):

        self.data1=self.data.copy()
        self.data1.drop(columns='Attrition', inplace=True)
        
        print(f"this is self.data1 information {self.data1.info()}")
        # Assuming 'data' is your DataFrame
        categorical_columns = ['Gender', 'Education', 'Marital_Status', 'Income', 'Card_Category']
        numerical_columns = self.data1.select_dtypes(include=[np.number]).columns.tolist()

        # Apply label encoding to the target column


        #X = data.drop('Attrition', axis=1)



        # Create the column transformer
        preprocessor = make_column_transformer(
            (OrdinalEncoder(), categorical_columns),
            remainder='passthrough'  # Numerical columns are passed through
        )

        #print(self.data.info())

        # Apply the transformations to the features
        X_transformed = preprocessor.fit_transform(self.data1)

        # Save the preprocessor object for later use on unseen data
        joblib.dump(preprocessor, os.path.join(self.config.root_dir, self.config.preprocessor_file))

        # Now, 'X_transformed' is your preprocessed feature matrix

        # Apply the transformations to the features
        #X_transformed = preprocessor.fit_transform(data)

        le = LabelEncoder()
        self.data['Attrition'] = le.fit_transform(self.data['Attrition'])

        # Convert the transformed data back into a DataFrame
        X_transformed_df = pd.DataFrame(X_transformed, columns=categorical_columns+numerical_columns)

        # Convert numerical columns back to their original data types
        for col in numerical_columns:
            X_transformed_df[col] = X_transformed_df[col].astype(self.data[col].dtype)

        self.data[categorical_columns+numerical_columns] = X_transformed_df


        # Replace the original columns in 'data' with the transformed columns
        self.data[categorical_columns+numerical_columns] = X_transformed_df

        print(f"this is the new transformed dataset which is self.data {self.data.info()}")
        print(f"this is the shape of the self.data {self.data.shape}")
        print(f"this is the head of the self.data {self.data.head()}")

        # Now, 'data' is your preprocessed DataFrame






        

   



    def correlation(self):
        
        # Find most important features relative to target Price
        print("Find most important features relative to Attrition-target")
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
        #train, test = train_test_split(transformed_dataset,test_size=0.33,random_state=25) # this line splits the data into train_test_split

        #train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False) # here it saves the train and test data in csv format inisde the artifacts-> transformation folder
        #test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        #logger.info("Splited data into training and test sets")
        #logger.info(train.shape) # this logs the information about that how many training and testing samples i have 
        #logger.info(test.shape)

        #print(train.shape)
        #print(test.shape)
        logger.info("Done with data creating the transformed_dataset in the artifacts folder")
