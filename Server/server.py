from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request, json
from PIL import Image
from cv2 import cv2
from flask_cors import CORS

from test import predict_single, predict_multiple

app = Flask(__name__)
CORS(app)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/single_predict', methods=['POST'])
def single_predict():
    filesimage = request.files.get('image')
    if filesimage is None:
        return jsonify(results=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    else:
        # filesimage.save("image/text.png")
        # im = cv2.imread('image/text.png')  # 读取图片rgb 格式<class 'numpy.ndarray'>
        # image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # 格式转换，bgr转rgb
        # image.save('image/text.png', dpi=(300.0, 300.0))  # 调整图像的分辨率为300,dpi可以更改
        img = cv2.imdecode(np.frombuffer(filesimage.read(), np.uint8), cv2.IMREAD_COLOR)
        # result_list = predict_multiple(img)
        # plt.imshow(img)
        # plt.show()
        # cv2.imwrite("5.jpg", img)
        result_list, result = predict_single(img)
        return jsonify(result=result, list=result_list)


@app.route('/multiple_predict', methods=['POST'])
def multiple_predict():
    filesimage = request.files.get('image')
    if filesimage is None:
        return jsonify(0)
    else:
        # filesimage.save("image/text.png")
        # im = cv2.imread('image/text.png')  # 读取图片rgb 格式<class 'numpy.ndarray'>
        # image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # 格式转换，bgr转rgb
        # image.save('image/text.png', dpi=(300.0, 300.0))  # 调整图像的分辨率为300,dpi可以更改
        img = cv2.imdecode(np.frombuffer(filesimage.read(), np.uint8), cv2.IMREAD_COLOR)
        result_list, b64 = predict_multiple(img)
        return jsonify(result=result_list, img=b64)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        img = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        result_list, b64 = predict_multiple(img)
        return jsonify(result=result_list, img=b64)
        # f.save('2.png')
    # return jsonify(result="success")


if __name__ == '__main__':
    app.run(debug=True, port='8000', host='127.0.0.1')
