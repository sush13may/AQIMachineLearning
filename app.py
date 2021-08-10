# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:40:23 2021

@author: sush1
"""

import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template,request


app=Flask(__name__)
model = pickle.load(open('XGBoost_Regrsspr_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    print('In function')
    df = pd.read_csv('real_2018.csv')
    df.replace(to_replace='-', value=0,inplace=True)
    
    prediction = model.predict(df.values)
    prediction = prediction.tolist()
    prediction = [round(num,2) for num in prediction]
    return render_template('predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)