async function send() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) { alert("画像を選択してください"); return; }

  const formData = new FormData();
  formData.append("file", file);

  const API_URL = "https://あなたのRenderURL/predict"; // 実際のURLに置き換え

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    if(data.error){
      alert("Error: " + data.error);
      return;
    }

    document.getElementById("result").innerHTML = `
      <h3>結果</h3>
      <p>分類: ${data.class}</p>
      <p>スコア: ${(data.score*100).toFixed(1)}%</p>
      <p>予測残寿命: ${data.predicted_months_left}ヶ月</p>
      <p>${data.suggestion}</p>
    `;
  } catch (err) {
    alert("通信エラー: " + err);
  }
}
