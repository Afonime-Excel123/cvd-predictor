// ============================================================
// script.js — CONNECTS THE FORM TO THE FLASK API
// UCI Heart Disease Dataset version
// ============================================================

const form        = document.getElementById('predictionForm');
const resultBox   = document.getElementById('resultBox');
const riskLevel   = document.getElementById('riskLevel');
const probability = document.getElementById('probability');
const resultMsg   = document.getElementById('resultMessage');
const probBar     = document.getElementById('probBar');
const resultEmoji = document.getElementById('resultEmoji');

const EMOJIS = { low: '💚', moderate: '🟡', high: '❤️' };

form.addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => { data[key] = value; });

    const submitBtn = form.querySelector('.submit-btn');
    submitBtn.querySelector('.btn-text').textContent = 'Analysing...';
    submitBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        const riskKey = result.risk_level.toLowerCase();

        riskLevel.textContent   = `${result.risk_level} Risk`;
        probability.textContent = `${result.probability}% probability of heart disease`;
        resultMsg.textContent   = result.message;
        resultEmoji.textContent  = EMOJIS[riskKey] || '❤️';

        probBar.style.width = '0%';
        resultBox.className = `result-card ${riskKey}`;
        resultBox.style.display = 'block';

        setTimeout(() => {
            probBar.style.width = `${result.probability}%`;
        }, 100);

        resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        resultBox.style.display = 'block';
        resultBox.className = 'result-card high';
        riskLevel.textContent = 'Error';
        resultMsg.textContent = 'Something went wrong. Make sure the server is running and try again.';
    }

    submitBtn.querySelector('.btn-text').textContent = 'Run Risk Assessment';
    submitBtn.disabled = false;
});