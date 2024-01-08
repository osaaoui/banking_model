import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer




cv = pickle.load(open("vectorizer.pkl", 'rb'))
clf = joblib.load(open("models/model.joblib", 'rb'))
issue_model= joblib.load('issue_model.pkl', 'rb')
def model_predict(email):
    if email == "":
        return ""
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction_issue= issue_model.predict(tokenized_email)
    print("Prediction: ", prediction)
    print("Prediction_issue: ", prediction_issue)
    result= (prediction, prediction_issue)
    return result
