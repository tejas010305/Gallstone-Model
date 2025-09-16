import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import zipfile
import os

zip_path = '/content/dataset-uci.zip'
extract_path = '/content/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


df = pd.read_csv('/content/dataset-uci.csv')


df.columns
# present(0), and absent(1)

df['Gallstone Status'] = df['Gallstone Status'].map({0:'Yes',1:'No'})

x = df[['Age', 'Gender', 'Comorbidity',
       'Coronary Artery Disease (CAD)', 'Hypothyroidism', 'Hyperlipidemia',
       'Diabetes Mellitus (DM)', 'Height', 'Weight', 'Body Mass Index (BMI)',
       'Total Body Water (TBW)', 'Extracellular Water (ECW)',
       'Intracellular Water (ICW)',
       'Extracellular Fluid/Total Body Water (ECF/TBW)',
       'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (LM) (%)',
       'Body Protein Content (Protein) (%)', 'Visceral Fat Rating (VFR)',
       'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)',
       'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)',
       'Visceral Muscle Area (VMA) (Kg)', 'Hepatic Fat Accumulation (HFA)',
       'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
       'High Density Lipoprotein (HDL)', 'Triglyceride',
       'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)',
       'Alkaline Phosphatase (ALP)', 'Creatinine',
       'Glomerular Filtration Rate (GFR)', 'C-Reactive Protein (CRP)',
       'Hemoglobin (HGB)', 'Vitamin D']]
y = df['Gallstone Status']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12)

knn = KNeighborsClassifier()
naive = GaussianNB()
log = LogisticRegression()
svm = SVC()

knn.fit(x_train,y_train)
naive.fit(x_train,y_train)
log.fit(x_train,y_train)
svm.fit(x_train,y_train)

knn_pred = knn.predict(x_test)
naive_pred = naive.predict(x_test)
log_pred = log.predict(x_test)
svm_pred = svm.predict(x_test)

def evaluate_models(name,y_test,y_pred):
       print(f"{name}")
  print(f"Accuracy : {accuracy_score(y_test,y_pred)}")
  print("-" * 40)


evaluate_models("KNN",y_test,knn_pred)
evaluate_models("Naive Bayes",y_test,naive_pred)
evaluate_models("Logistic Regression",y_test,log_pred)
evaluate_models("SVM ",y_test,svm_pred)

largest = 0
if accuracy_score(y_test,knn_pred) > 0:
  largest = accuracy_score(y_test,knn_pred)
if accuracy_score(y_test,naive_pred) > largest:
  largest = accuracy_score(y_test,naive_pred)
if accuracy_score(y_test,log_pred) > largest:
  largest = accuracy_score(y_test,log_pred)
if accuracy_score(y_test,svm_pred) > largest:
  largest = accuracy_score(y_test,svm_pred)
else:
  print(largest)

import pickle

with open('Gallstone_Model.pkl','wb') as file:
  pickle.dump(log,file)
