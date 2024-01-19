from flask import Flask, request, render_template
import numpy as np
import pandas as pandas
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        features = [np.array(input_data)]
        prediction = model.predict(features)
        prediction_text = "Waspada, Anda Beresiko Terkena STROKE" if prediction == 1 else "Selamat, Anda Tidak Beresiko Terkena Stroke"
        return render_template("index.html", prediction=prediction)
    
    except ValueError as e:
        return render_template("index.html", prediction_text="Terjadi kesalahan: {}".format(str(e)))

if __name__=='__main__':
    app.run(debug=True)