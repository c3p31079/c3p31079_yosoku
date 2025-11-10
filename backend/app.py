from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
from degradation_predictor import predict_replacement_time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model("swing_degradation_model.h5")
classes = ["chain_early","chain_mid","chain_late","seat_early","seat_mid","seat_late"]

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # 画像前処理
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # モデル推論
    pred = model.predict(img)[0]
    cls = classes[np.argmax(pred)]
    score = float(np.max(pred))

    # 劣化スコアから交換時期を推定
    months_left, suggestion = predict_replacement_time(cls, score)

    return jsonify({
        "class": cls,
        "score": round(score, 3),
        "predicted_months_left": months_left,
        "suggestion": suggestion
    })

@app.route("/")
def home():
    return "劣化予測のAPIが実行中です！"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
