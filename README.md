# End to End Churn Prediction application with MLops

#### Working of an Application
https://github.com/mahendra867/End-to-End-Churn-Prediction-application/assets/95703197/a73aac65-d14e-4751-8355-40c77805cc5f






## Problem Statement
Description:
A churn model is a mathematical representation of how churn impacts your business. Churn calculations are built on existing data (the number of customers who left your service during a given time period). A predictive churn model extrapolates on this data to show future potential churn rates.

Churn (aka customer attrition) is a scourge on subscription businesses. When your revenue is based on recurring monthly or annual contracts, every customer who leaves puts a dent in your cash flow. High retention rates are vital for your survival. So what if we told you there was a way to predict, at least to some degree, how and when your customers will cancel?

Building a predictive churn model helps you make proactive changes to your retention efforts that drive down churn rates. Understanding how churn impacts your current revenue goals and making predictions about how to manage those issues in the future also helps you stem the flow of churned customers. If you don’t take action against your churn now, any company growth you experience simply won’t be sustainable.

Comprehensive customer profiles help you see what types of customers are canceling their accounts. Now it’s time to figure out how and why they’re churning. Ask yourself the following questions to learn more about the pain points in your product and customer experience that lead to a customer deciding to churn.

## What is customer churn?
Customer churn (or customer attrition) is a tendency of customers to abandon a brand and stop being a paying client of a particular business. The percentage of customers that discontinue using a company’s products or services during a particular time period is called a customer churn (attrition) rate. One of the ways to calculate a churn rate is to divide the number of customers lost during a given time interval by the number of acquired customers, and then multiply that number by 100 percent. For example, if you got 150 customers and lost three last month, then your monthly churn rate is 2 percent.

Churn rate is a health indicator for businesses whose customers are subscribers and paying for services on a recurring basis, thus, a customer stays open for more interesting or advantageous offers. Plus, each time their current commitment ends, customers have a chance to reconsider and choose not to continue with the company. Of course, some natural churn is inevitable, and the figure differs from industry to industry. But having a higher churn figure than that is a definite sign that a business is doing something wrong.”

There are many things brands may do wrong, from complicated onboarding when customers aren’t given easy-to-understand information about product usage and its capabilities to poor communication, e.g. the lack of feedback or delayed answers to queries. Another situation: Longtime clients may feel unappreciated because they don’t get as many bonuses as the new ones.

### This data frame contains the following columns:

- 'Clientium' : Unique identifier for the customer holding the account: we remove this column
- 'Attrition': Internal event (customer activity) variable - if the account is closed then 1 else 0: this is our target
- 'Age': Customer's Age in Years.
- 'Gender': M=Male, F=Female.
- 'Dependent_count': Number of dependents.
- 'Education': Educational Qualification of the account holder.
- 'Marital_Status': Married, Single, Divorced, Unknown.
- 'Income': Annual Income Category of the account holder
- 'Card_Category': Type of Card (Blue, Silver, Gold, Platinum).
- 'Months_on_book': Period of relationship with bank.
- 'Total_Relationship_Count': Total no. of products held by the customer.
- 'Months_Inactive': No. of Months in the last 12 months.
- 'Contacts_Count': No. of Contacts in the last 12 months.
- 'Credit_Limit': Credit Limit on the Credit Card.
- 'Total_Revolving_Bal': Total Revolving Balance on the Credit Card.
- 'Avg_Open_To_Buy': Open to Buy Credit Line (Average of last 12 months
- 'Total_Amt_Chng_Q4_Q1': Change in Transaction Amount (Q4 over Q1).
- 'Total_Trans_Amt': Total Transaction Amount (Last 12 months).
- 'Total_Trans_Ct': Total Transaction Count (Last 12 months).
- 'Total_Ct_Chng_Q4_Q1': Change in Transaction Count (Q4 over Q1).
- 'Avg_Utilization_Ratio': Average Card Utilization Ratio.

## Approach 

### Data Ingestion
Description:
In the data ingestion stage, I implemented the process of fetching and preparing the dataset for further analysis. Here's a breakdown of the steps involved:

- Downloading Data: Utilizing the urllib library, I downloaded the data from the provided URL. The Zipfile module was then employed to handle the zipped format of the data.

- Extracting Zip File: The downloaded zip file was extracted using the Zipfile module to access the dataset for subsequent processing.

### Data Validation
Description:
The data validation stage involves ensuring the integrity and consistency of the dataset. Here's how I approached this stage:

- Validating All Columns: Using the pandas library, I read the extracted dataset and compared its columns with the schema defined in the schema.yaml file. This process verified whether all expected columns were present in the dataset.

### Data Transformation
Description:
Data transformation is crucial for preparing the dataset in a format suitable for modeling. Here's an overview of the transformation steps:

- Renaming Columns: The column names of the dataset were updated to improve readability and consistency.

- Feature Selection: Certain columns deemed irrelevant or redundant were dropped from the dataset to streamline the feature space.

- Creating Preprocessing Pipeline: A preprocessing pipeline was established to handle categorical variables through ordinal encoding and preserve numerical features.

- Correlation Analysis: Analyzing the correlation between features and the target variable provided insights into feature importance and potential model performance.

- Train-Test Split: The dataset was split into training and testing sets to facilitate model training and evaluation.

### Model Trainer and Model Evaluation
Description:
The model trainer stage involves training various machine learning models and evaluating their performance. Here's how I proceeded with this stage:

- Training and Testing: I trained the model using the training dataset and evaluated its performance on both training and testing sets.

- Error Analysis: Metrics such as Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, Accuracy, and AUC score were computed to assess model performance.

- SMOTE and ADASYN Sampling: To address class imbalance, I experimented with Synthetic Minority Over-sampling Technique (SMOTE) and Adaptive Synthetic Sampling (ADASYN) techniques to generate synthetic samples for the minority class.

- Model Selection and Evaluation: Various models were trained and evaluated, including Decision Tree Regressor and Extra Trees Classifier. Evaluation metrics such as Accuracy, Recall Ratio, and AUC score were computed to assess model performance.

- Visualization: Heatmaps and Receiver Operating Characteristic (ROC) curves were plotted to visually analyze the model's performance.

By following this workflow, I effectively addressed the problem statement of predicting customer churn and provided actionable insights for retention efforts.







## Modular coding WorkFlows

1. Update config.yam1
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the [main.py](http://main.py/)
9. Update the [app.py](http://app.py/)




# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/mahendra867/End-to-End-Churn-Prediction-application.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.9 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


## Azure Deployment Workflow
This guide provides step-by-step instructions for deploying your application on Azure Cloud using Docker containers and Azure services.

### 1.Creating Azure Account
- Open Azure cloud application and sign up for a free account.
- Choose "Pay as you go" while creating the account.
### 2. Searching Azure Services
- Search for Web App for Containers in Azure.
- Search for Container Registry in Azure.
### 3. Creating Container Registry in Azure
- 1)Search for Container Registry in Azure.
- 2)Tap on create and fill in the details:
    -  Registry Name: churnrepositoryimagename
    - Location: South India
    - Price Plan: Standard
- 3)Click on create.
### 4. Pushing Docker Image to Azure Container Registry
- Build a new Docker image:

```bash
docker build -t churnrepositoryimagename.azurecr.io/churn_app:latest .
```
- Login to Azure Container Registry:

```bash
docker login churnrepositoryimagename.azurecr.io
```
- Push the Docker image to Azure Container Registry:
```bash
docker push churnrepositoryimagename.azurecr.io/churn_app:latest
```
### 5. Creating Web App for Container
- Search for Web App for Containers.
- Choose Docker container in the public section and select Linux as the operating system.
- Fill in the details and click on Review+Create.
### 6. Configuring Web App
- In the Docker section, choose the repository name and image.
- Click on Review+Create and then create.
### 7. Final Configuration
- Navigate to the deployment center of the Web App.
- Configure SCM Basic Auth Publishing Credentials to ON under Platform settings.
- Save the general settings configuration.
### 8. Connecting Azure with GitHub
- Go to settings and choose GitHub Actions.
- Register GitHub with Azure.
- Choose the GitHub organization and repository.
- Click on save.
### 9. Viewing the Running Website
- Click on Browse in the Web App.
- Share the public link to access the application.

Follow these steps to successfully deploy your application on Azure Cloud using Docker containers and Azure services.



