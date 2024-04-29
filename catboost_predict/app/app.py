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
catboost = CatBoostRegressor()
model_save_path = '../predict_model/catboost_model.cbm'
catboost.load_model(model_save_path)


class InputData(BaseModel):
    copolymer: Optional[float] = Field(default=0.0, alias="copolymer")
    A3431531268: Optional[float] = Field(default=0.0, alias="A3431531268")
    A864662311: Optional[float] = Field(default=0.0, alias="A864662311")
    A3975295864: Optional[float] = Field(default=0.0, alias="A3975295864")
    A4216335232: Optional[float] = Field(default=0.0, alias="A4216335232")
    A3217380708: Optional[float] = Field(default=0.0, alias="A3217380708")
    A951226070: Optional[float] = Field(default=0.0, alias="A951226070")
    G994485099: Optional[float] = Field(default=0.0, alias="G994485099")
    G2976033787: Optional[float] = Field(default=0.0, alias="G2976033787")


def catboost_predict(input_data):
    return catboost.predict(input_data)


@app.route('/')
def index():
    return render_template('index.html', message="Hello, World!")


@app.route('/test', methods=['POST'])
def process():
    try:
        inputData = InputData.parse_raw(request.data)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400

    # 获取请求体中的数据
    data = str(inputData.A2245273601)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # 调用Python方法
    result = data[::-1]

    # 返回处理结果
    return jsonify({'result': result})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputData = InputData.parse_raw(request.data)
    except ValidationError as e:
        print(e)
        return jsonify({'error': "invalid arguments."}), 400

    input_data_df = pd.DataFrame([inputData.dict()])

    result = catboost_predict(input_data_df)
    return jsonify({'result': result[0]})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
