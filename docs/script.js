async function send() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) {
    alert("画像を選択してください。");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const API_URL = "https://your-render-app-name.onrender.com/predict"; // ← RenderのURLに置き換える

  document.getElementById("loading").style.display = "block";
  document.getElementById("result").innerHTML = "";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error("サーバーエラー: " + res.status);
    }

    const data = await res.json();
    document.getElementById("loading").style.display = "none";

    if (data.error) {
      document.getElementById("result").innerHTML = `<p style="color:red;">${data.error}</p>`;
    } else {
      document.getElementById("result").innerHTML = `
        <h3>判定結果</h3>
        <p>分類: <b>${data.predicted_label}</b></p>
        <p>確信度: ${(data.confidence * 100).toFixed(1)}%</p>
        <p>${data.recommendation}</p>
      `;
    }

  } catch (err) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("result").innerHTML = `<p style="color:red;">通信エラー: ${err.message}</p>`;
  }
}
