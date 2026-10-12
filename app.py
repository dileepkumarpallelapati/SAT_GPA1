import flask
from flask import Flask, render_template, request, jsonify

import numpy as np

import pickle
import  sklearn.linear_model as LinearRegression

with open('slr_model.pkl','rb') as f:
    model = pickle.load(f)



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def prediction():
    sat=request.json['sat_score']


    if not 1600 <= sat <= 2100:
        return jsonify({'error':'enter correct val'})


    #gpa = (sat-400)/300
    #gpa = (sat-1600)/125

    gpa=model.predict([[sat]])[0]


    return jsonify({'gpa':round(gpa,2)})


    #return render_template('index.html',gpa=gpa)







if  __name__ == '__main__':
    app.run(debug=True)