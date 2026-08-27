document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const sourceLang = document.getElementById('sourceLang');
    const targetLang = document.getElementById('targetLang');
    const translateBtn = document.getElementById('translateBtn');
    const translateSpinner = document.getElementById('translateSpinner');
    const charCount = document.getElementById('charCount');
    const presetSelect = document.getElementById('presetSelect');
    const swapBtn = document.getElementById('swapBtn');
    const copyBtn = document.getElementById('copyBtn');
    const speakBtn = document.getElementById('speakBtn');
    const adapterSelect = document.getElementById('adapterSelect');
    const switchModelBtn = document.getElementById('switchModelBtn');
    const activeModelName = document.getElementById('activeModelName');
    const outputModelTag = document.getElementById('outputModelTag');
    const latencyBadge = document.getElementById('latencyBadge');
    const latencyTime = document.getElementById('latencyTime');
    const tempRange = document.getElementById('tempRange');
    const tempValue = document.getElementById('tempValue');
    const maxTokensRange = document.getElementById('maxTokensRange');
    const tokensValue = document.getElementById('tokensValue');
    const repetitionPenalty = document.getElementById('repetitionPenalty');
    const repValue = document.getElementById('repValue');
    const toast = document.getElementById('toast');

    // 1. Character Counter
    sourceText.addEventListener('input', () => {
        const length = sourceText.value.length;
        charCount.textContent = `${length} / 1000`;
    });

    // 2. Preset Selection
    presetSelect.addEventListener('change', () => {
        if (presetSelect.value) {
            sourceText.value = presetSelect.value;
            charCount.textContent = `${sourceText.value.length} / 1000`;
        }
    });

    // 3. Swap Languages
    swapBtn.addEventListener('click', () => {
        const temp = sourceLang.value;
        sourceLang.value = targetLang.value;
        targetLang.value = temp;
        
        if (targetText.textContent && !targetText.classList.contains('placeholder')) {
            sourceText.value = targetText.textContent.trim();
            targetText.textContent = 'Translation output will appear here...';
            targetText.classList.add('placeholder');
            charCount.textContent = `${sourceText.value.length} / 1000`;
        }
    });

    // 4. Parameter Sliders Update
    tempRange.addEventListener('input', (e) => tempValue.textContent = e.target.value);
    maxTokensRange.addEventListener('input', (e) => tokensValue.textContent = e.target.value);
    repetitionPenalty.addEventListener('input', (e) => repValue.textContent = e.target.value);

    // 5. Toast Notification Helper
    function showToast(message, duration = 3000) {
        toast.textContent = message;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), duration);
    }

    // 6. Copy Translation
    copyBtn.addEventListener('click', () => {
        const textToCopy = targetText.textContent;
        if (!textToCopy || targetText.classList.contains('placeholder')) {
            showToast('⚠️ No translation text to copy!');
            return;
        }
        navigator.clipboard.writeText(textToCopy.trim()).then(() => {
            showToast('✅ Translation copied to clipboard!');
        }).catch(() => {
            showToast('❌ Copy failed. Please select text manually.');
        });
    });

    // 7. Text-to-Speech Pronunciation (Web Speech API)
    speakBtn.addEventListener('click', () => {
        const textToSpeak = targetText.textContent;
        if (!textToSpeak || targetText.classList.contains('placeholder')) {
            showToast('⚠️ No translation text to read!');
            return;
        }
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(textToSpeak.trim());
            utterance.rate = 0.9;
            window.speechSynthesis.speak(utterance);
            showToast('🔊 Playing audio...');
        } else {
            showToast('⚠️ Text-to-speech not supported in browser.');
        }
    });

    // 8. Execute Translation API Call
    async function performTranslation() {
        const text = sourceText.value.trim();
        if (!text) {
            showToast('⚠️ Please enter text to translate.');
            return;
        }

        translateBtn.disabled = true;
        translateSpinner.classList.remove('hidden');
        latencyBadge.classList.add('hidden');
        targetText.textContent = 'Translating using Hugging Face model...';
        targetText.classList.add('placeholder');

        const startTime = performance.now();

        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    source_lang: sourceLang.value,
                    target_lang: targetLang.value,
                    temperature: parseFloat(tempRange.value),
                    max_new_tokens: parseInt(maxTokensRange.value, 10),
                    repetition_penalty: parseFloat(repetitionPenalty.value)
                })
            });

            const endTime = performance.now();
            const elapsed = Math.round(endTime - startTime);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: 'Server error' }));
                throw new Error(errData.detail || 'Translation failed.');
            }

            const data = await response.json();
            targetText.textContent = data.translation || 'No output generated.';
            targetText.classList.remove('placeholder');

            latencyTime.textContent = elapsed;
            latencyBadge.classList.remove('hidden');

            if (data.model_adapter) {
                outputModelTag.textContent = `Adapter: ${data.model_adapter}`;
            }

        } catch (error) {
            targetText.textContent = `Error: ${error.message}`;
            targetText.classList.remove('placeholder');
            showToast(`❌ Translation Error: ${error.message}`);
        } finally {
            translateBtn.disabled = false;
            translateSpinner.classList.add('hidden');
        }
    }

    translateBtn.addEventListener('click', performTranslation);

    // Ctrl+Enter or Cmd+Enter trigger translate
    sourceText.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            performTranslation();
        }
    });

    // 9. Dynamic Adapter Switching API Call
    switchModelBtn.addEventListener('click', async () => {
        const selectedAdapter = adapterSelect.value;
        switchModelBtn.disabled = true;
        switchModelBtn.innerHTML = '<span>⏳ Loading...</span>';

        try {
            const response = await fetch('/api/switch_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ adapter_subfolder: selectedAdapter })
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Failed to switch model' }));
                throw new Error(err.detail);
            }

            const data = await response.json();
            activeModelName.textContent = selectedAdapter;
            outputModelTag.textContent = `Adapter: ${selectedAdapter}`;
            showToast(`✅ Successfully switched to: ${selectedAdapter}`);
        } catch (err) {
            showToast(`❌ Switch Error: ${err.message}`);
        } finally {
            switchModelBtn.disabled = false;
            switchModelBtn.innerHTML = '<span>🔄 Switch Adapter</span>';
        }
    });
});
