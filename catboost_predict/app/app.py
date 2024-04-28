"""
@version: python3.7
@author: ‘mengyuantan‘
@contact: tanmy1016@126.com
@time: 2024/4/28 11:42
"""

from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
from pydantic import BaseModel, ValidationError

app = Flask(__name__)
catboost = CatBoostRegressor()
model_save_path = '../predict_model/catboost_model.cbm'
catboost.load_model(model_save_path)


class InputData(BaseModel):
    copolymer: float
    A3431531268: float
    A864662311: float
    A3975295864: float
    A4216335232: float
    A3217380708: float
    A951226070: float
    G994485099: float
    G2976033787: float


def catboost_predict(input_data):
    return catboost.predict(input_data)


def process_data(data):
    # 这里可以放置具体的处理逻辑
    return data[::-1]  # 假设我们只是简单地反转字符串


@app.route('/process', methods=['POST'])
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
    result = process_data(data)

    # 返回处理结果
    return jsonify({'result': result})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputData = InputData.parse_raw(request.data)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400

    result = catboost_predict(inputData)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
