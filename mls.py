import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  ,LogisticRegression
from sklearn.metrics import r2_score ,mean_absolute_error
import numpy 
from sklearn.metrics import accuracy_score
import gradio 
data= pd.read_csv("studentPerformance.csv")
x=data[['Study_Hours','Attendance','Practice_Tests']]
y_lin=data['Final_Score']
y_log=data['Pass_Fail']
x_lin_train,x_lin_test,y_lin_train,y_lin_test=train_test_split(x,y_lin,test_size=0.2,random_state=42)
x_log_train ,x_log_test,y_log_train,y_log_test=train_test_split(x,y_log,test_size=0.2,random_state=42)
lin_model=LinearRegression()
log_model = LogisticRegression(C=0.5, max_iter=1000, solver='lbfgs')
lin_model.fit(x_lin_train,y_lin_train)
log_model.fit(x_log_train ,y_log_train)
y_lin_pred=lin_model.predict(x_lin_test)
y_log_pred=log_model.predict(x_log_test)
mse_lin=mean_absolute_error(y_lin_test,y_lin_pred)
print('le mse pour le modele lineaire =',mse_lin)
r2=r2_score(y_lin_test,y_lin_pred)
print('le r2 score=' ,r2)
acc=accuracy_score(y_log_test,y_log_pred)
print('accuracy du modele logistique=' ,acc)
def prediction (Study_Hours,Attendance,Practice_Tests):
    score=lin_model.predict([[Study_Hours,Attendance,Practice_Tests]])[0]
    resultat=log_model.predict([[Study_Hours,Attendance,Practice_Tests]])[0]
    resultat_text="pass"if resultat==1 else "fail" 
    return f"votre score :{score}",f"RESULT :{resultat_text}"
interf=gradio.Interface(
fn=prediction,
inputs = [
        gradio.Number(label="Study_hours"),
        gradio.Number(label="Attendance"),
        gradio.Number(label="Practice_Tests")
    ],
outputs=["text","text"],
title="Le score et la resultat final d'un etudiant",
description= "Entrez vos informations pour predire votre resultat "
)
interf.launch()












