from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# モデル読み込み
MODEL_PATH = "swing_degradation_model.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ モデル読み込み成功:", MODEL_PATH)
else:
    raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {MODEL_PATH}")

# クラス名
CLASS_NAMES = ["chain_early", "chain_mid", "chain_late"]

# 月数設定（必要に応じて調整可）
RECOMMENDATION_MONTHS = {
    "chain_early": 3,
    "chain_mid": 6,
    "chain_late": 0
}

@app.route("/")
def home():
    return "Swing degradation prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # 画像の読み込み
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 推論
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_NAMES[predicted_index]

        # 推奨月数の取得
        months = RECOMMENDATION_MONTHS.get(predicted_label, None)
        if months is None:
            recommendation = "推奨時期を特定できません。"
        elif months == 0:
            recommendation = "すぐに交換を推奨します。"
        else:
            recommendation = f"{months}か月後の交換を推奨します。"

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": float(np.max(predictions[0])),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
