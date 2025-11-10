import numpy as np
import datetime

# 仮の進行率モデル（線形近似）
# 劣化スコア: 0.0=初期, 1.0=限界
def predict_replacement_time(cls, score):
    # 部位ごとの平均劣化速度（1ヶ月あたりスコア上昇）
    rates = {
        "chain": 0.05,  # 20ヶ月で劣化完了
        "seat": 0.04,   # 25ヶ月で劣化完了
    }

    # 部位を抽出
    part = "chain" if "chain" in cls else "seat"

    # 残存寿命（ヶ月）
    remaining = (1.0 - score) / rates[part]
    remaining = max(0, round(remaining, 1))

    # 推奨メッセージ
    if remaining < 3:
        suggestion = "⚠ 早急に交換推奨"
    elif remaining < 6:
        suggestion = "◇ 交換を計画してください"
    else:
        suggestion = "◎ 現在良好"

    return remaining, suggestion
