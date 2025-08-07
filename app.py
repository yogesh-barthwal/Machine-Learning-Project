from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
application= Flask(__name__)

app= application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data= CustomData(
            fixed_acidity=float(request.form['fixed_acidity']),
            volatile_acidity=float(request.form['volatile_acidity']),
            citric_acid=float(request.form['citric_acid']),
            residual_sugar=float(request.form['residual_sugar']),
            chlorides=float(request.form['chlorides']),
            free_sulfur_dioxide=float(request.form['free_sulfur_dioxide']),
            total_sulfur_dioxide=float(request.form['total_sulfur_dioxide']),
            density=float(request.form['density']),
            pH=float(request.form['pH']),
            sulphates=float(request.form['sulphates']),
            alcohol=float(request.form['alcohol'])
        )

        pred_df= data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline= PredictPipeline()
        results= predict_pipeline.predict(pred_df)
        return render_template('home.html', results= results[0])
    

if __name__== "__main__":
    app.run(host="0.0.0.0", debug=True)