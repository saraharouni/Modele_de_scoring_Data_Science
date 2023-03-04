from flask import Flask, jsonify, request
import streamlit as st
import pandas as pd
import pickle
import os 

app = Flask(__name__)

modele_path = os.path.join(os.getcwd(), "best_lgb_model.pkl")

with open(modele_path, 'rb') as f:
    modele = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame.from_dict(data)
    predictions = modele.predict(data_df)
    return jsonify(predictions.tolist())

@app.route('/')
def index():
    st.set_page_config(page_title="Mon Dashboard")
    st.write("Hello World!")

if __name__ == '__main__':
    app.run(port='5080')