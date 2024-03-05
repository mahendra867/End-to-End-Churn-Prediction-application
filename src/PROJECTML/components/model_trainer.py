# these packages i need in order to create my Model Trainer components 
import pandas as pd
import os
from PROJECTML import logger
import joblib # here iam saving the model because i want to save the data
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
#from lightgbm import LGBMClassifier
from numpy import where
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import  classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PROJECTML.entity.config_entity import ModelTrainerConfig



# now here iam defining a class called model trainer inside it will take ModelTrainerConfig
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    # here iam creating a methode which it will traine the model by using train and test dataset
    #def train(self):
        #self.train_data = pd.read_csv(self.config.train_data_path) # here it is taking the paths of train and test dataset
        #self.test_data = pd.read_csv(self.config.test_data_path)


        

        #lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42) # here i have created my Elastic model which it takes the alpha,l1_ratio, random state values 
        #lr.fit(train_x, train_y) # here i have initiated the model training

        #joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name)) # here are training my model iam just saving inside the folder Model_trainer which it will get create inside the artifacts






    def training_testing_evalution_without_sampling(self,models,graph):
        #def Models_NO(models, graph):
        #X, y = Definedata()
        transformed_dataset=pd.read_csv("artifacts\\data_transformation\\transformed_dataset.csv")
        self.X=transformed_dataset.drop(columns=['Attrition']).values
        self.Y=transformed_dataset['Attrition'].values
        print("this is self.X",self.X)
        print("this is self.Y",self.Y)

        #Split the data into training and test sets. (0.75, 0.25) split.
        self.train,self.test = train_test_split(transformed_dataset,test_size=0.33,random_state=25) # this line splits the data into train_test_split

        print("this is self.train",self.train)
        print("this is self.test",self.test)

        #train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False) # here it saves the train and test data in csv format inisde the artifacts-> transformation folder
        #test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(self.train.shape)  # This logs the information about that how many training and testing samples i have 
        logger.info(self.test.shape)

        self.x_train=self.train.drop(columns=['Attrition'])
        self.y_train=self.train['Attrition']
        self.x_test=self.test.drop(columns=['Attrition'])
        self.y_test=self.test['Attrition']
        logger.info("Done with train test split")
        model = models
        
        print("Columns of X_train:", self.x_train.columns)
        print("Target variable (y_train):", self.y_train.name)
        
        
        model.fit(self.x_train,self.y_train)
        self.y_pred = model.predict(self.x_test)
        self.y_total = model.predict(self.X)
        logger.info("Done with model prediction ")
        
        if graph:
            train_matrix = pd.crosstab(self.y_train, model.predict(self.x_train), rownames=['Actual'], colnames=['Predicted'])    
            test_matrix = pd.crosstab(self.y_test, model.predict(self.x_test), rownames=['Actual'], colnames=['Predicted'])
            matrix = pd.crosstab(self.Y, model.predict(self.X), rownames=['Actual'], colnames=['Predicted'])
        
            f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(15, 2))
        
            g1 = sns.heatmap(train_matrix, annot=True, fmt=".1f", cbar=False,annot_kws={"size": 18},ax=ax1)
            g1.set_title("{}/train set".format(model))
            g1.set_ylabel('Total Churn = {}'.format(1- self.y_train.sum()), fontsize=14, rotation=90)
            g1.set_xlabel('Accuracy for TrainSet: {}'.format(accuracy_score(model.predict(self.x_train), self.y_train)))
            g1.set_xticklabels(['Churn','Not Churn'],fontsize=12)

            g2 = sns.heatmap(test_matrix, annot=True, fmt=".1f",cbar=False,annot_kws={"size": 18},ax=ax2)
            g2.set_title("{}/test set".format(model))
            g2.set_ylabel('Total Churn = {}'.format(1- self.y_test.sum()), fontsize=14, rotation=90)
            g2.set_xlabel('Accuracy for TestSet: {}'.format(accuracy_score(self.y_pred, self.y_test)))
            g2.set_xticklabels(['Churn','Not Churn'],fontsize=12)

            g3 = sns.heatmap(matrix, annot=True, fmt=".1f",cbar=False,annot_kws={"size": 18},ax=ax3)
            g3.set_title("{}/total set".format(model))
            g3.set_ylabel('Total Churn = {}'.format(1- self.Y.sum()), fontsize=14, rotation=90)
            g3.set_xlabel('Accuracy for TotalSet: {}'.format(accuracy_score(self.y_total, self.Y)))
            g3.set_xticklabels(['Churn','Not Churn'],fontsize=12)
        
            plt.show()
            print ("")
            print ("Classification Report: ")
            print (classification_report(self.Y, self.y_total))
        
        else:
            print("\t\tError Table")
            print('Mean Absolute Error      : ', metrics.mean_absolute_error(self.y_test, (self.y_pred)))
            print('Mean Squared  Error      : ', metrics.mean_squared_error(self.y_test, (self.y_pred) ))
            print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(self.y_test, (self.y_pred) )))
            print('Accuracy on Traing set   : ', model.score(self.x_train,self.y_train))
            print('Accuracy on Testing set  : ', model.score(self.x_test,self.y_test))
            print('AUC score                :', roc_auc_score(self.Y, self.y_total)*100,'%')        
        return self.y_total, self.Y
    
    

    def error_table(self): 

        y_predicted, y_actual = self.training_testing_evalution_without_sampling(DecisionTreeRegressor(), False)
        fpr, tpr, thresholds = roc_curve(y_actual, y_predicted)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.0])
        plt.ylim([-0.1,1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


    
    def SMOTE(self):
        # borderline-SMOTE for imbalanced dataset
        from collections import Counter
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification
        from imblearn.over_sampling import SMOTE
        from matplotlib import pyplot
        from numpy import where

        transformed_dataset=pd.read_csv("artifacts\\data_transformation\\transformed_dataset.csv")
        X=transformed_dataset.drop(columns=['Attrition']).values
        Y=transformed_dataset['Attrition'].values
        print("this is self.X",X)
        print("this is self.Y",Y)
    
        #X, y = Definedata()

    # summarize class distribution
        counter = Counter(Y)
        print(counter)
    # transform the dataset
        smt = SMOTE(random_state=0)
        X, Y = smt.fit_resample(X, Y) 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=2)
    # summarize the new class distribution
        counter = Counter(Y)
        print(counter)
    # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(Y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.show()
        
        return X_train, X_test, y_train, y_test

    logger.info("Started the ADASYN sampling")
    def ADASYN(self):
        from collections import Counter
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import ADASYN
        from matplotlib import pyplot
        from numpy import where

        transformed_dataset=pd.read_csv("artifacts\\data_transformation\\transformed_dataset.csv")
        X=transformed_dataset.drop(columns=['Attrition']).values
        Y=transformed_dataset['Attrition'].values
        print("this is self.X",X)
        print("this is self.Y",Y)


    # summarize class distribution
        counter = Counter(Y)
        print(counter)
    # transform the dataset
        X, Y = ADASYN().fit_resample(X, Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=2)
    # summarize the new class distribution
        counter = Counter(Y)
        print(counter)
    # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(Y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.show()

        return X_train, X_test, y_train, y_test
    


    logger.info("started building the model funtion for Training and prediction of sampled data of Both SMOTE and ADASYN techinques")
    def Models(self, X_train, X_test, y_train, y_test, title, graph):
        model_instance = ExtraTreesClassifier()
        model = model_instance
        print(model)
        print(X_train,y_train)
        #print("Columns of X_train:", X_train.columns)
        #print("Target variable (y_train):", y_train.name)

        #X_train.columns
        #y_train.head()
        model.fit(X_train,y_train)

        transformed_dataset=pd.read_csv("artifacts\\data_transformation\\transformed_dataset.csv")
        X=transformed_dataset.drop(columns=['Attrition']).values
        Y=transformed_dataset['Attrition'].values
        print("this is self.X",X)
        print("this is self.Y",Y)
        
        #X, y = Definedata()
        train_matrix = pd.crosstab(y_train, model.predict(X_train), rownames=['Actual'], colnames=['Predicted'])    
        test_matrix = pd.crosstab(y_test, model.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
        matrix = pd.crosstab(Y, model.predict(X), rownames=['Actual'], colnames=['Predicted'])
        
        if graph:
            f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True, figsize=(15, 2))
        
            g1 = sns.heatmap(train_matrix, annot=True, fmt=".1f", cbar=False,annot_kws={"size": 18},ax=ax1)
            g1.set_title(title)
            g1.set_ylabel('Total Churn = {}'.format(y_train.sum()), fontsize=14, rotation=90)
            g1.set_xlabel('Accuracy score (TrainSet): {}'.format(accuracy_score(model.predict(X_train), y_train)))
            g1.set_xticklabels(['Churn','Not Churn'],fontsize=12)

            g2 = sns.heatmap(test_matrix, annot=True, fmt=".1f",cbar=False,annot_kws={"size": 18},ax=ax2)
            g2.set_title(title)
            g2.set_ylabel('Total Churn = {}'.format(y_test.sum()), fontsize=14, rotation=90)
            g2.set_xlabel('Accuracy score (TestSet): {}'.format(accuracy_score(model.predict(X_test), y_test)))
            g2.set_xticklabels(['Churn','Not Churn'],fontsize=12)

            g3 = sns.heatmap(matrix, annot=True, fmt=".1f",cbar=False,annot_kws={"size": 18},ax=ax3)
            g3.set_title(title)
            g3.set_ylabel('Total Churn = {}'.format(Y.sum()), fontsize=14, rotation=90)
            g3.set_xlabel('Accuracy score (Total): {}'.format(accuracy_score(model.predict(X), Y)))
            g3.set_xticklabels(['Churn','Not Churn'],fontsize=12)

            plt.show()

        print("\t\tError Table")
        print('Accuracy on Traing set   : ', model.score(X_train,y_train))
        print('Accuracy on Testing set  : ', model.score(X_test,y_test))
        print('Overall Accuracy_Score   :',accuracy_score(Y, model.predict(X))*100,'%')
        print('Recall ratio             :',metrics.recall_score(Y, model.predict(X))*100,'%')
        print('AUC score                :', roc_auc_score(Y, model.predict(X))*100,'%')

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        #joblib.dump(model_instance, "best_model.joblib")

        return Y, model.predict(X)
    




