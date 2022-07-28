# from crypt import methods
from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__)

modelutil = joblib.load('Utilization')
modelth = joblib.load('Throughput')

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/prediksi")
def util():
    return render_template('pred.html', util = 0, th = 0, rssi=0, rrc_connected_users=0, dl_uu_latency=0, sinr_pusch=0)

@app.route("/predict", methods = ["POST"])
def predict():
    rssi, rrc_connected_users, dl_uu_latency , sinr_pusch = [x for x in request.form.values()]
    data = np.array([[float(rssi), float(rrc_connected_users), float(dl_uu_latency), float(sinr_pusch)]])

    predicted1 = modelutil.predict(data)
    hasil1 = np.round(predicted1, decimals=3)
    ka = np.array(float(hasil1))

    predicted2 = modelth.predict(data)
    hasil2 = np.round(predicted2, decimals=3)
    ku = np.array(float(hasil2))

    return render_template("pred.html", util = ka, th = ku, rssi=rssi, rrc_connected_users=rrc_connected_users, dl_uu_latency=dl_uu_latency,  sinr_pusch=sinr_pusch)


if __name__ == "__main__":
    app.run(debug=True)