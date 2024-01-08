import json
import re
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import pickle
# NLTK
import nltk
from nltk.corpus import stopwords


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
import xgboost as xgb
def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    mean_squared_error=np.sqrt(metrics.mean_squared_error(y_test, predictions))
    r2_score=metrics.r2_score(y_test, predictions)
    accuracy= metrics.accuracy_score(y_test,predictions)
    print("Metrics")
    print("---------------------","\n")
    print("MSE: ", mean_squared_error)
    print("r2 ", r2_score)
    print("accuracy ", accuracy)

    return mean_squared_error, r2_score, accuracy



def get_feat_and_target():
    df= pd.read_csv('issues.csv')

    print("DF columns: ", df.columns)
    data2= pd.DataFrame(df, columns=['complaint', 'Issue'])
    print("Data2 columns", data2.columns)
    # Encoder la variable cible "category"
    data2["issue_id"]= data2["Issue"].factorize()[0]
    category_id_df_issue = data2[['Issue', 'issue_id']].drop_duplicates()
    category_to_id_issue = dict(category_id_df_issue.values)
    id_to_category_issue = dict(category_id_df_issue[['issue_id', 'Issue']].values)
    X_issue= data2.loc[:, 'complaint']
    y_issue= data2.loc[:, 'issue_id']
    
    
    

    
    return X_issue,y_issue    

def train_and_evaluate(config_path):
    cv = pickle.load(open("vectorizer.pkl", 'rb'))

    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["issue_target"]
    
    #alpha=config["multiNB"]["alpha"]
    #fit_prior=config["multiNB"]["fit_prior"]
    learning_rate=config["xgboost"]["learning_rate"]
    n_estimators=config["xgboost"]["n_estimators"]
    max_depth=config["xgboost"]["max_depth"]
    n_jobs=config["xgboost"]["n_jobs"]
    
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target()
    test_x,test_y=get_feat_and_target()
    

    xtrain_cv_issue = cv.transform(train_x)
    xtest_cv_issue = cv.transform(test_x)
    mlflow.set_experiment("banking_demo")
    with mlflow.start_run():

        model_dir = config["model_dir"]
        model_webapp_dir= config["model_webapp_dir"]
        #model = MultinomialNB(alpha=alpha,fit_prior=fit_prior)
        model_issue = xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs )
        model_issue.fit(xtrain_cv_issue, train_y.ravel())
        y_pred = model_issue.predict(xtest_cv_issue)
        mean_squared_error, r2_score, accuracy = accuracymeasures(test_y,y_pred,'weighted')
        joblib.dump(model_issue, 'issue_model.pkl')
        #joblib.dump(cv, "vectorizer.pkl")

        mlflow.log_param("learning_rate_issue", learning_rate)
        mlflow.log_param("n_estimators_issue", n_estimators)
        mlflow.log_param("max_depth_issue", max_depth)
        mlflow.log_param("n_jobs_issue", n_jobs)
        mlflow.log_metric("mean_squared_error_issue", mean_squared_error)
        mlflow.log_metric("r2_score_issue", r2_score)
        mlflow.log_metric("accuracy_issue", accuracy)


        # # For remote server only (Dagshub)
        remote_server_uri = "https://dagshub.com/osaaoui/banking_model.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)


        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model_issue, "model_issue", registered_model_name="xgb_model_issue")
        else:
            mlflow.sklearn.log_model(model_issue, "model_issue")


################### MLFLOW ###############################
    #mlflow_config = config["mlflow_config"]
    #remote_server_uri = mlflow_config["remote_server_uri"]
    #config = read_params(config_path)
    #mlflow_config = config["mlflow_config"] 
    #model_dir = config["model_dir"]
    #mlflow.set_tracking_uri(remote_server_uri)
    #mlflow.set_experiment(mlflow_config["experiment_name"])

    #with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
     #   model = MultinomialNB(alpha=alpha,fit_prior=fit_prior)
     #   model.fit(train_x, train_y.ravel())
     #   y_pred = model.predict(test_x)
     #   mean_squared_error, r2_score = accuracymeasures(test_y,y_pred,'weighted')
     #   joblib.dump(model, model_dir)
     #   mlflow.log_param("alpha",alpha)
     #   mlflow.log_param("fit_prior", fit_prior)

     #   mlflow.log_metric("mean_squared_error", mean_squared_error)
     #   mlflow.log_metric("r2_score", r2_score)
       
      #  tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

       # if tracking_url_type_store != "file":
        #    mlflow.sklearn.log_model(
         #       model, 
          #      "model", 
           #     registered_model_name=mlflow_config["registered_model_name"])
        #else:
         #   mlflow.sklearn.load_model(model, "model")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)



