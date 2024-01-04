import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.header("Diabetes detection Application")

image = Image.open("C:\\Users\\ACER\\Downloads\\diabetes.png")
st.image(image)

data = pd.read_csv("E:\\ML projects\\Diabetes with st\\diabetes.csv")

st.subheader("Data of People")
st.dataframe(data)
st.subheader("Description of Data")
st.write(data.iloc[:,:8].describe())

x = data.iloc[:,:8].values
y=data.iloc[:,8].values
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.8)

model = RandomForestClassifier(n_estimators=500)

model.fit(X_train,y_train)
yPred = model.predict(X_test)

st.subheader("Accuracy of the trained Model")
st.write(accuracy_score(y_test,yPred))

st.subheader("Enter your Input data")
preg = st.slider("Pregnancy",0,20,0)
gluc = st.slider("Glucose",0,200,0)
bp = st.slider("Blood Pressure",0,130,0)
sThick = st.slider("Skin Thickness",0,100,0)
ins = st.slider("Insulin",0.0,1000.0,0.0)
bmi = st.slider("BMI",0.0,70.0,0.0)
dpf = st.slider("DPF",0.000,3.000,0.000)
age = st.slider("Age",0,100,0)


inputDic = {"Pregnancies":preg, "Glucose":gluc, "Blood Pressure":bp, "Skin Thickness":sThick, "Insulin":ins, "BMI":bmi, "DPF":dpf, "Age":age }
ui = pd.DataFrame(inputDic, index=["User Input Values"])

st.subheader("Entered Data")
st.write(ui)

st.subheader("Predictions (0-Non Diabetes, 1-Diabetes)")
st.write(model.predict(ui))