from fastapi import FastAPI
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
model_path = os.path.join(os.path.dirname(__file__), "student_academic_performance_model.pkl")
model = joblib.load(model_path)
df=pd.read_csv("C:/Users/Ritam Choudhury/STUDENTS ACADEMIC PERFORMANCE DATASET/xAPI-Edu-Data.csv")
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from fastapi import FastAPI
from fastapi.responses import JSONResponse
app = FastAPI()

def preprocess_data(df):
    encoder=OrdinalEncoder(categories=[['L','M','H']])
    df['Class_encoded']=encoder.fit_transform(df[['Class']])
    df.drop(columns=['StageID','GradeID','SectionID','ParentschoolSatisfaction','Class'],inplace=True)
    enc2=OrdinalEncoder(categories=[['M','F']])
    df['gender']=enc2.fit_transform(df[['gender']])
    df['NationalITy'] = OrdinalEncoder().fit_transform(df[['NationalITy']])
    df['PlaceofBirth']=OrdinalEncoder().fit_transform(df[['PlaceofBirth']])
    df['Topic']=OrdinalEncoder().fit_transform(df[['Topic']])
    df['Semester']=OrdinalEncoder().fit_transform(df[['Semester']])
    df['Relation']=OrdinalEncoder().fit_transform(df[['Relation']])
    enc=OrdinalEncoder(categories=[['Yes','No']])
    df['ParentAnsweringSurvey']=enc.fit_transform(df[['ParentAnsweringSurvey']])
    enc1=OrdinalEncoder(categories=[['Under-7','Above-7']])
    df['StudentAbsenceDays']=enc1.fit_transform(df[['StudentAbsenceDays']])
    return df

@app.get("/")
def root():
    return {"message": "Welcome to the Student Academic Performance API"}

@app.get("/originaldata")
def show():
    return JSONResponse(content=df.to_dict(orient="records"))
@app.get("/data")
def read_data():
    processed_df = preprocess_data(df.copy())
    return JSONResponse(content=processed_df.to_dict(orient="records"))
    
sc=StandardScaler()
processed_df=preprocess_data(df.copy())
X=processed_df.drop(columns=['Class_encoded'])
y=processed_df['Class_encoded']
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=42)
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

@app.post("/predict/model1")
def predict1():
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    accuracy=model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=15, scoring='precision_macro')
    return {
    "Model": "RandomForestClassifier",
    "Accuracy" : float(accuracy),
    "cross_validation_scores": [float(score) for score in scores],
    "mean_cross_validation_accuracy": float(scores.mean())
    }

@app.post("/predict/model2")
def predict2():
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(X_train, y_train)
    accuracy=et_model.score(X_test,y_test)
    y_pred = et_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores = cross_val_score(et_model, X, y, cv=15, scoring='precision_macro')
    return {
    "Model": "ExtraTreesClassifier",
    "Accuracy" : float(accuracy),
    "cross_validation_scores": [float(score) for score in scores],
    "mean_cross_validation_accuracy": float(scores.mean())
    }

@app.post("/predict/model3")
def predict3():
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    accuracy=xgb_model.score(X_test,y_test)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores = cross_val_score(xgb_model, X, y, cv=15, scoring='precision_macro')
    return {
    "Model": "XGBClassifier",
    "Accuracy" : float(accuracy),
    "cross_validation_scores": [float(score) for score in scores],
    "mean_cross_validation_accuracy": float(scores.mean())
    }

@app.post("/predict/compare")
def compare_models():
    return {
        "RandomForest": predict1(),
        "ExtraTrees": predict2(),
        "XGBoost": predict3()
    }



