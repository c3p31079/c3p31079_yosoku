import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from degradation_predictor import predict_replacement_time

# Flask 初期化
app = Flask(__name__)
CORS(app)

# モデルロード（起動時に一度だけ）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "swing_degradation_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

classes = ["chain_early","chain_mid","chain_late","seat_early","seat_mid","seat_late"]

# 画像から劣化分類＋交換時期推定
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(file).convert("RGB").resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 推論
        pred = model.predict(img_array)[0]
        cls = classes[np.argmax(pred)]
        score = float(np.max(pred))

        # 交換時期推定
        months_left, suggestion = predict_replacement_time(cls, score)

        return jsonify({
            "class": cls,
            "score": round(score, 3),
            "predicted_months_left": months_left,
            "suggestion": suggestion
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "劣化予測のAPIが実行中です！"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
