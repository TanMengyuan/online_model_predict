"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@time: 2024/4/28 11:42
"""

import pandas as pd
from flask import Flask, request, jsonify, render_template
from catboost import CatBoostRegressor
from pydantic import BaseModel, ValidationError, Field
from typing import Optional

app = Flask(__name__)

film_model_catboost = CatBoostRegressor()
film_model_save_path = '../predict_model/film/catboost_model.cbm'
film_model_catboost.load_model(film_model_save_path)

powder_model_catboost = CatBoostRegressor()
powder_model_save_path = '../predict_model/powder/catboost_model.cbm'
powder_model_catboost.load_model(powder_model_save_path)


class FilmInputData(BaseModel):
    copolymer: Optional[float] = Field(default=0.0, alias="copolymer")
    A3431531268: Optional[float] = Field(default=0.0, alias="A3431531268")
    A864662311: Optional[float] = Field(default=0.0, alias="A864662311")
    A3975295864: Optional[float] = Field(default=0.0, alias="A3975295864")
    A4216335232: Optional[float] = Field(default=0.0, alias="A4216335232")
    A3217380708: Optional[float] = Field(default=0.0, alias="A3217380708")
    A951226070: Optional[float] = Field(default=0.0, alias="A951226070")
    G994485099: Optional[float] = Field(default=0.0, alias="G994485099")
    G2976033787: Optional[float] = Field(default=0.0, alias="G2976033787")


class PowderInputData(BaseModel):
    A2720313463: Optional[float] = Field(default=0.0, alias="A2720313463")
    A2041434490: Optional[float] = Field(default=0.0, alias="A2041434490")
    A2084364935: Optional[float] = Field(default=0.0, alias="A2084364935")
    A951226070: Optional[float] = Field(default=0.0, alias="A951226070")
    G3692055567: Optional[float] = Field(default=0.0, alias="G3692055567")
    G3276511768: Optional[float] = Field(default=0.0, alias="G3276511768")
    A4216335232: Optional[float] = Field(default=0.0, alias="A4216335232")

    def transform_fields(self):
        data = self.dict()
        transformed_data = {}
        for key, value in data.items():
            if key == "A4216335232":
                transformed_data["A4216335232-1"] = value
            else:
                transformed_data[key + "-2"] = value
        return transformed_data


@app.route('/')
def index():
    return render_template('index.html', message="Welcome to the Prediction Service")


@app.route('/predict/film', methods=['POST'])
def predict_film():
    try:
        inputData = FilmInputData.parse_raw(request.data)
    except ValidationError as e:
        print(e)
        return jsonify({'error': "invalid arguments."}), 400

    input_data_df = pd.DataFrame([inputData.dict()])

    result = film_model_catboost.predict(input_data_df)
    return jsonify({'result': result[0]})


@app.route('/predict/powder', methods=['POST'])
def predict_powder():
    try:
        inputData = PowderInputData.parse_raw(request.data)
    except ValidationError as e:
        print(e)
        return jsonify({'error': "invalid arguments."}), 400

    input_data_df = pd.DataFrame([inputData.transform_fields()])

    result = powder_model_catboost.predict(input_data_df)
    return jsonify({'result': result[0]})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
