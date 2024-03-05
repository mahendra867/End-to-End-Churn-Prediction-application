from PROJECTML.config.configuration import ConfigurationManager
from PROJECTML.components.model_trainer import ModelTrainer
from PROJECTML import logger
from sklearn.tree import DecisionTreeRegressor


 # here i have named a stage name w.r.t below 1 class created 
STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline: # here i have created a class
    def __init__(self):
        pass

    def main(self): # here i have created a main methode 
       config = ConfigurationManager() # here iam initlizing my ConfigurationManager()
       model_trainer_config = config.get_model_trainer_config() # here iam getting my get_model_trainer_config()
       model_trainer_config = ModelTrainer(config=model_trainer_config) # here iam  passing my  model_trainer_config to the ModelTrainer function
        #model_trainer_config.train() # here iam training the model
       y_predict, y_test = model_trainer_config.training_testing_evalution_without_sampling(DecisionTreeRegressor(), True)
        
       error_table_result=model_trainer_config.error_table()
       logger.info("We could confirm that the data is IMBALANCED and the regressor can not handle this data set. Let start the second step.")
       logger.info("<<<<<<<------------------------Started sampling the data for balancing the target feature by SMOTE and ADASYN Techinques------------------------>>>>>>>")
       logger.info("Started the SMOTE sampling")
       X_train1, X_test1, y_train1, y_test1=model_trainer_config.SMOTE()
       logger.info("Started the ADASYN sampling")
       X_train4, X_test4, y_train4, y_test4 =model_trainer_config.ADASYN()



       title = 'ExtraTreesClassifier/SMOTE'
        #model_instance = ExtraTreesClassifier()
       model_trainer_config.Models(X_train1, X_test1, y_train1, y_test1, title,True)

       title = 'ExtraTreesClassifier/ADASYN'
       model_trainer_config.Models(X_train4, X_test4, y_train4, y_test4, title,True)





# Now here i will initilized the pipeline of model trainer inside my main methode

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e