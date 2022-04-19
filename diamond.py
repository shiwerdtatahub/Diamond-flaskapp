# coding:utf-8
from flask import Flask, Response, jsonify , render_template, request, flash
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
app.secret_key = "adetunji"

#laod the model
model = pickle.load(open('knnmodel.sav','rb'))
en = pickle.load(open('en.sav','rb'))
en1 = pickle.load(open('en1.sav','rb'))
en2 = pickle.load(open('en2.sav','rb'))

@app.route('/')
def home():
    result = ''
    return render_template('diamond.html', **locals())


@app.route('/predict', methods=['POST',  'GET'])
def predict():
    flash(" Price of the diamonds is:")
    carat = float(request.form['carat'])
    cut = request.form['cut']
    color = request.form['color']
    clarity = request.form['clarity']
    depth = float(request.form['depth'])
    table = float(request.form['table'])
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])
   
    #to transform the encoded input
    cutT = en.transform([cut])[0]
    colorT = en1.transform([color])[0]
    clarityT = en2.transform([clarity])[0]


    new_diamond = OrderedDict([('caret',carat),('cut',cutT),('color',colorT),('clarity',clarityT),('depth',depth),('table',table),('x', x),('y',y),('z',z)])
    new_diamond = pd.Series(new_diamond).values.reshape(1,-1)
    result = int(model.predict(new_diamond)[0])
    return render_template('diamond.html', **locals())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0' ,port=9096)