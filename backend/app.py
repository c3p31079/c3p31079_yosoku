from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import traceback

app = Flask(__name__)
CORS(app, origins="*")  # GitHub Pages からアクセス可能に

# モデルロード
model = tf.keras.models.load_model("swing_degradation_model.h5")
classes = ["chain_early","chain_mid","chain_late","seat_early","seat_mid","seat_late"]

# 仮の交換時期関数
def predict_replacement_time(cls, score):
    # スコアに応じて残り月数と提案を返す（例）
    months_left = max(1, int((1-score)*36))  # 最大3年を想定
    suggestion = "交換推奨" if months_left < 6 else "経過観察"
    return months_left, suggestion

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']

        # 画像前処理
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).resize((224,224))
        img = np.array(img)/255.0
        if img.shape[-1] == 4:  # RGBAならRGBに変換
            img = img[..., :3]
        img = np.expand_dims(img, axis=0)

        # モデル推論
        pred = model.predict(img)[0]
        cls = classes[np.argmax(pred)]
        score = float(np.max(pred))

        # 交換時期予測
        months_left, suggestion = predict_replacement_time(cls, score)

        return jsonify({
            "class": cls,
            "score": round(score, 3),
            "predicted_months_left": months_left,
            "suggestion": suggestion
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Swing degradation API is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
