from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from degradation_predictor import predict_replacement_time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model("swing_degradation_model.h5")
classes = ["chain_early","chain_mid","chain_late","seat_early","seat_mid","seat_late"]

@app.route("/predict", methods=["POST"])
def predict():
    img_bytes = request.files["file"].read()
    img = Image.open(io.BytesIO(img_bytes)).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    cls = classes[np.argmax(pred)]
    score = float(np.max(pred))

    #劣化スコアから交換時期を推定
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
    app.run(host='0.0.0.0', port=10000)  #Renderのポートは自動で環境変数PORTになる場合もあるため注意！が必要らしい
