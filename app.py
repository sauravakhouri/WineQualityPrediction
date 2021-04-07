import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=joblib.load('finalized_model.pkl')

StandardScaler=StandardScaler()
X_train=pd.read_csv('X_train.csv')
X_train=StandardScaler.fit_transform(X_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # for rendering HTML GUI
    features=[float(x) for x in request.form.values()]
    final_features=[np.array(features)]

    #handing the input as  model is trained using standardised dataset
    final_features=StandardScaler.transform(final_features)

    prediction=model.predict(final_features)
    print(final_features)

    output=prediction[0]
    print(output)

    return render_template('index.html',prediction_text="The Quality Rating of the Wine is: {}".format(output))


if __name__=="__main__":
    app.run(debug=True)
