const input = document.getElementById('message-input');
const checkBtn = document.getElementById('check-btn');
const result = document.getElementById('result');
const resultLabel = document.getElementById('result-label');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');

async function checkMessage() {
    const message = input.value.trim();
    if (!message) {
        alert('Please enter a message to check.');
        return;
    }

    checkBtn.disabled = true;
    checkBtn.textContent = 'Checking...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        result.classList.remove('hidden');
        resultLabel.textContent = data.prediction === 'spam' ? '🚨 Spam Detected' : '✅ Not Spam (Ham)';
        resultLabel.className = `result-label ${data.prediction}`;
        confidenceBar.style.width = `${data.confidence}%`;
        confidenceBar.style.background = data.prediction === 'spam' ? '#dc2626' : '#16a34a';
        confidenceText.textContent = `Model confidence: ${data.confidence}%`;
    } catch (err) {
        alert('Something went wrong. Please try again.');
    } finally {
        checkBtn.disabled = false;
        checkBtn.textContent = 'Check Message';
    }
}

checkBtn.addEventListener('click', checkMessage);

document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        input.value = chip.dataset.text;
        checkMessage();
    });
});
