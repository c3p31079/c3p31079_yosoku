async function send() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) { alert("画像を選択してください"); return; }

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("https://c3p31079-yosoku.onrender.com", {
    method: "POST",
    body: formData
  });
  const data = await res.json();

  document.getElementById("result").innerHTML = `
    <h3>結果</h3>
    <p>分類: ${data.class}</p>
    <p>スコア: ${(data.score * 100).toFixed(1)}%</p>
    <p>予測残寿命: ${data.predicted_months_left}ヶ月</p>
    <p>${data.suggestion}</p>
  `;
}
