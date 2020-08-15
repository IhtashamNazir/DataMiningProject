from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.externals import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
app = Flask(__name__)

# model= pickle.load(open('model.pkl','rb'))
with open(r'E:\Study\7th Semester\Data Minig\Project\ForestFireWeb\pickle_model', "rb") as f:
    pm = pickle.load(f)

# pm = joblib.load("joblib_model")


@app.route('/')
def hello_world():
    return render_template("Index.html")


@app.route('/home')
def home():
    return render_template("Index.html")


@app.route('/about')
def about():
    return render_template("About.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    # print(int_features)
    # print(final)
    prediction = pm.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('Index.html',
                               dpredicts='Your Forest is in Danger.'
                                         '\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('Index.html',
                               predicts='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
