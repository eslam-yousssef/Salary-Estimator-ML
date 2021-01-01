# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 23:19:37 2020

@author: Eslam Youssef
"""

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
#    age=request.form('age')
    data = request.form.to_dict()
    
    yrs=float(data['yrs'])
    arr=np.array([[yrs]])
    
    pred=model.predict(arr)
    
    return render_template('index.html', result=pred)

if __name__ == "__main__":
    app.run(debug=True)