import json
import re
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
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

def clean_text(text):
    stopword=set(stopwords.words('english'))
    text = re.sub("[^a-zA-Z]", " ", text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = ' '.join(text)
    text = text.lower()

    return text


def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    df_grouped= df.groupby(['Product'])
    df_balanced =df_grouped.apply(lambda x: x.sample(df_grouped.size().min()).reset_index(drop=True))
    df_balanced = df_balanced.droplevel(['Product'])
    
    df_balanced['clean_text'] = df_balanced["Consumer complaint narrative"].apply(clean_text)
    # Encoder la variable cible
    df_balanced["category_id"]= df_balanced["Product"].factorize()[0]
    #category_id_df = df_balanced[['Product', 'category_id']].drop_duplicates()
    #category_to_id = dict(category_id_df.values)
    #id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    #print("Categories mapping: ", id_to_category.items())
    #x=df.loc[:, 0:7]
    #y=df.iloc[:, 7].values.reshape(-1,1)
    

    #bagofword_vec = count_vec.fit_transform(data['complaint'])
    #labels =data.category_id
    x=df_balanced.loc[:, 'clean_text']
    y=df_balanced.loc[:, 'category_id']
    print("X. Head: ", x.head(2))

    
    return x,y    

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    
    #alpha=config["multiNB"]["alpha"]
    #fit_prior=config["multiNB"]["fit_prior"]
    learning_rate=config["xgboost"]["learning_rate"]
    n_estimators=config["xgboost"]["n_estimators"]
    max_depth=config["xgboost"]["max_depth"]
    n_jobs=config["xgboost"]["n_jobs"]
    
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    count_vec = CountVectorizer(max_df=0.90,min_df=2,
                           max_features=1000,stop_words='english')

    cv= count_vec.fit(train_x)
    #xtrain_cv = count_vec.fit_transform(train_x)
    xtrain_cv = cv.transform(train_x)
    xtest_cv = count_vec.transform(test_x)
    test_size= 0.3
    mlflow.set_experiment("banking_demo")
    with mlflow.start_run():

        model_dir = config["model_dir"]
        model_webapp_dir= config["model_webapp_dir"]
        #model = MultinomialNB(alpha=alpha,fit_prior=fit_prior)
        model = xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs )
        model.fit(xtrain_cv, train_y.ravel())
        y_pred = model.predict(xtest_cv)
        mean_squared_error, r2_score, accuracy = accuracymeasures(test_y,y_pred,'weighted')
        joblib.dump(model, model_dir)
        joblib.dump(cv, "vectorizer.pkl")

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_metric("mean_squared_error", mean_squared_error)
        mlflow.log_metric("r2_score", r2_score)
        mlflow.log_metric("accuracy", accuracy)


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



