async function send() {
    const file = document.getElementById("fileInput").files[0];
    if (!file) { alert("画像を選択してください"); return; }

    const formData = new FormData();
    formData.append("file", file);

    const API_URL = "https://c3p31079-yosoku.onrender.com/predict"; // RenderのURL

    try {
        const res = await fetch(API_URL, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();

        if(data.error){
            alert("APIエラー: " + data.error);
            return;
        }

        document.getElementById("result").innerHTML = `
            <h3>結果</h3>
            <p>分類: ${data.class}</p>
            <p>スコア: ${(data.score * 100).toFixed(1)}%</p>
            <p>予測残寿命: ${data.predicted_months_left}ヶ月</p>
            <p>${data.suggestion}</p>
        `;
    } catch (err) {
        console.error(err);
        alert("API通信でエラーが発生しました。コンソールを確認してください。");
    }
}
