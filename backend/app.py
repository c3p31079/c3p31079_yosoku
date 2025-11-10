from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

#モデル読み込み
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "swing_degradation_model.h5")
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ モデル読み込み成功: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {MODEL_PATH}")

#クラス名
CLASS_NAMES = ["chain_early", "chain_mid", "chain_late"]

#交換推奨時期（例）
RECOMMENDATION_MONTHS = {
    "chain_early": 6,
    "chain_mid": 3,
    "chain_late": 0
}


@app.route("/")
def home():
    return "Swing degradation prediction API is running."


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "画像ファイルが送信されていません。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ファイル名が空です。"}), 400

    try:
        #画像前処理
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        #予測
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]))

        #推奨メッセージ
        months = RECOMMENDATION_MONTHS.get(predicted_label, None)
        if months is None:
            recommendation = "推奨時期を特定できません。"
        elif months == 0:
            recommendation = "すぐに交換を推奨します。"
        else:
            recommendation = f"{months}か月後の交換を推奨します。"

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": round(confidence, 3),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
