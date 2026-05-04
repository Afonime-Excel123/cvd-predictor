// ============================================================
// script.js — CONNECTS THE FORM TO THE FLASK API
// ============================================================

const form        = document.getElementById('predictionForm');
const resultBox   = document.getElementById('resultBox');
const riskLevel   = document.getElementById('riskLevel');
const probability = document.getElementById('probability');
const resultMsg   = document.getElementById('resultMessage');
const probBar     = document.getElementById('probBar');
const resultEmoji = document.getElementById('resultEmoji');

// Emoji per risk level — makes results feel human not clinical
const EMOJIS = { low: '💚', moderate: '🟡', high: '❤️' };

form.addEventListener('submit', async function(e) {
    e.preventDefault();

    // Collect form values into a plain object
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => { data[key] = value; });

    // Loading state
    const submitBtn = form.querySelector('.submit-btn');
    submitBtn.querySelector('.btn-text').textContent = 'Analysing...';
    submitBtn.disabled = true;

    try {
        // POST to Flask API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        const riskKey = result.risk_level.toLowerCase();

        // Populate result card
        riskLevel.textContent  = `${result.risk_level} Risk`;
        probability.textContent = `${result.probability}% probability of developing CVD within 10 years`;
        resultMsg.textContent  = result.message;
        resultEmoji.textContent = EMOJIS[riskKey] || '❤️';

        // Animated probability bar
        probBar.style.width = '0%';
        resultBox.className = `result-card ${riskKey}`;
        resultBox.style.display = 'block';

        // Small delay so the CSS transition actually plays
        setTimeout(() => {
            probBar.style.width = `${result.probability}%`;
        }, 100);

        // Smooth scroll to result
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        resultBox.style.display = 'block';
        resultBox.className = 'result-card high';
        riskLevel.textContent = 'Error';
        resultMsg.textContent = 'Something went wrong. Make sure the server is running and try again.';
    }

    submitBtn.querySelector('.btn-text').textContent = 'Analyse My Risk';
    submitBtn.disabled = false;
});