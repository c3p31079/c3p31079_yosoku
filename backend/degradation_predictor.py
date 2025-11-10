# degradation_predictor.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

MODEL_PATH = "swing_degradation_model.h5"
DATA_LOG = "data/degradation_log.csv"
PLOT_PATH = "static/graph.png"

model = load_model(MODEL_PATH)

def predict_degradation(img_path):
    #1. 画像をCNNに入力できる形に整形
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    #2. 劣化スコア（確率ベース）を予測
    pred = model.predict(img_array)[0]
    degradation_score = round(float(pred[0] * 100), 2)

    #3. 予測結果をCSVへ記録
    log_prediction(degradation_score)

    #4. スコア推移を可視化
    plot_degradation_trend()

    #5. 劣化速度を基に交換時期を推定
    replacement_month = estimate_replacement()

    return degradation_score, replacement_month

def log_prediction(score):
    now = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame([[now, score]], columns=["date", "score"])
    
    if os.path.exists(DATA_LOG):
        df_old = pd.read_csv(DATA_LOG)
        df = pd.concat([df_old, df])
    
    df.to_csv(DATA_LOG, index=False)

#折れ線グラフ作成
def plot_degradation_trend():
    df = pd.read_csv(DATA_LOG)
    plt.figure(figsize=(6,4))
    plt.plot(df["date"], df["score"], marker='o')
    plt.title("Degradation Score Trend")
    plt.xlabel("Date")
    plt.ylabel("Degradation Score (0–100)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

def estimate_replacement(threshold=80):
    df = pd.read_csv(DATA_LOG)
    if len(df) < 2:
        return "データ不足"

    x = np.arange(len(df))
    y = df["score"].values
    slope = np.polyfit(x, y, 1)[0]  #劣化速度（スコア上昇率）

    if slope <= 0:
        return "改善傾向あり"

    current = y[-1]
    remaining = (threshold - current) / slope
    months_left = round(remaining)
    
    return f"{months_left} ヶ月後（推定）"
