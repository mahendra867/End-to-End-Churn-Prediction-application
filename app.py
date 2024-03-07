from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from PROJECTML.pipeline.prediction import PredictionPipeline,CustomData
from PROJECTML import logger

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET']) 
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # Create CustomData object using form data and the schema
        data = CustomData(
            Age=int(request.form.get("Age")),
            Gender=request.form.get("Gender"),
            Dependent_count=int(request.form.get('Dependent_count')),
            Education=request.form.get("Education"),
            Marital_Status=request.form.get("Marital_Status"),
            Income=request.form.get("Income"),
            Card_Category=request.form.get("Card_Category"),
            Months_on_book=int(request.form.get("Months_on_book")),
            Total_Relationship_Count=int(request.form.get("Total_Relationship_Count")),
            Months_Inactive=int(request.form.get("Months_Inactive")),
            Contacts_Count=int(request.form.get("Contacts_Count")),
            Credit_Limit=float(request.form.get("Credit_Limit")),
            Total_Revolving_Bal=int(request.form.get("Total_Revolving_Bal")),
            Total_Amt_Chng_Q4_Q1=float(request.form.get("Total_Amt_Chng_Q4_Q1")),
            Total_Trans_Amt=int(request.form.get("Total_Trans_Amt")),
            Total_Trans_Ct=int(request.form.get("Total_Trans_Ct")),
            Total_Ct_Chng_Q4_Q1=float(request.form.get("Total_Ct_Chng_Q4_Q1")),
            Avg_Utilization_Ratio=float(request.form.get("Avg_Utilization_Ratio"))
        )



        datatransformed = data.get_data_as_dataframe()
        logger.info("This is the data frame we passed to the model: %s", datatransformed)
        logger.info('Initiated prediction')
        predict_pipeline = PredictionPipeline()
        logger.info("Done with initilizing the object to the predictionnpipeline() class")

        data = np.array(datatransformed).reshape(1,18)  # Exclude target column
        #logger.info(" this is array of the user data: %s",data)
        #logger.info("Done with dataframe converted to array now iam passing that to my preprocessor and model object")
        prediction = predict_pipeline.predict(data)
        logger.info("done with prediction ")
        final_result = str(int(prediction[0]))  # Convert numpy array to integer first, then to string
        logger.info('Made prediction and returning to results.html')
        return render_template("results.html", final_result=final_result)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9095)