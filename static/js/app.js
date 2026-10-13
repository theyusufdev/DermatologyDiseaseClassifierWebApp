/**
 * Dermatology Disease Classifier — Frontend Logic
 * Handles form interaction, prediction requests, and result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── State ──
    const featureValues = {};
    const featureOrder = [
        'erythema', 'scaling', 'definite_borders', 'itching',
        'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
        'oral_mucosal_involvement', 'knee_and_elbow_involvement',
        'scalp_involvement', 'family_history', 'melanin_incontinence',
        'eosinophils_infiltrate', 'PNL_infiltrate', 'fibrosis_papillary_dermis',
        'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
        'clubbing_rete_ridges', 'elongation_rete_ridges',
        'thinning_suprapapillary_epidermis', 'spongiform_pustule',
        'munro_microabcess', 'focal_hypergranulosis',
        'disappearance_granular_layer', 'vacuolisation_damage_basal_layer',
        'spongiosis', 'saw_tooth_appearance_retes', 'follicular_horn_plug',
        'perifollicular_parakeratosis', 'inflammatory_mononuclear_infiltrate',
        'band_like_infiltrate', 'age'
    ];

    // Initialize all features to 0
    featureOrder.forEach(f => featureValues[f] = 0);

    // ── Scale & Binary Button Handlers ──
    document.querySelectorAll('.scale-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const feature = btn.dataset.feature;
            const value = parseInt(btn.dataset.value);
            featureValues[feature] = value;

            // Update active state within the same group
            const parent = btn.closest('.scale-selector');
            parent.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Ripple effect
            addRipple(btn);
        });
    });

    document.querySelectorAll('.binary-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const feature = btn.dataset.feature;
            const value = parseInt(btn.dataset.value);
            featureValues[feature] = value;

            const parent = btn.closest('.binary-toggle');
            parent.querySelectorAll('.binary-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            addRipple(btn);
        });
    });

    // ── Age Input ──
    const ageInput = document.getElementById('input-age');
    if (ageInput) {
        ageInput.addEventListener('input', (e) => {
            let val = parseInt(e.target.value);
            if (isNaN(val)) val = 0;
            if (val < 0) val = 0;
            if (val > 120) val = 120;
            featureValues['age'] = val;
        });
    }

    // ── Predict Button ──
    const predictBtn = document.getElementById('btn-predict');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePredict);
    }

    // ── Reset Button ──
    const resetBtn = document.getElementById('btn-reset');
    if (resetBtn) {
        resetBtn.addEventListener('click', handleReset);
    }

    // ── Ambient mouse glow ──
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            card.style.setProperty('--mouse-x', x + '%');
            card.style.setProperty('--mouse-y', y + '%');
        });
    });

    // ── Functions ──
    async function handlePredict() {
        // Validate age
        if (featureValues['age'] <= 0) {
            showToast('Lütfen geçerli bir yaş değeri girin.');
            return;
        }

        // Build feature array in correct order
        const features = featureOrder.map(f => featureValues[f]);

        // UI feedback
        predictBtn.classList.add('loading');
        const originalHTML = predictBtn.innerHTML;
        predictBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
            </svg>
            Analiz Ediliyor...
        `;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Prediction failed');
            }

            const data = await response.json();
            renderResults(data);

            // Scroll to results
            setTimeout(() => {
                document.getElementById('result-panel').scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }, 200);

        } catch (error) {
            showToast('Tahmin yapılamadı: ' + error.message);
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.innerHTML = originalHTML;
        }
    }

    function renderResults(data) {
        const panel = document.getElementById('result-panel');
        const diseaseName = document.getElementById('result-disease-name');
        const classLabel = document.getElementById('result-class-label');
        const probList = document.getElementById('probability-list');

        diseaseName.textContent = data.predicted_label;
        classLabel.textContent = `Sınıf ${data.predicted_class}`;

        // Sort probabilities by value desc
        const sorted = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]);

        probList.innerHTML = '';
        sorted.forEach(([name, prob], idx) => {
            const item = document.createElement('div');
            item.className = 'prob-item' + (idx === 0 ? ' is-top' : '');
            item.innerHTML = `
                <div class="prob-item__header">
                    <span class="prob-item__name">${name}</span>
                    <span class="prob-item__value">${prob.toFixed(2)}%</span>
                </div>
                <div class="prob-item__bar-bg">
                    <div class="prob-item__bar-fill" style="width: 0%"></div>
                </div>
            `;
            probList.appendChild(item);

            // Animate bar fill
            requestAnimationFrame(() => {
                setTimeout(() => {
                    item.querySelector('.prob-item__bar-fill').style.width = prob + '%';
                }, 50 + idx * 100);
            });
        });

        panel.classList.add('visible');
    }

    function handleReset() {
        // Reset state
        featureOrder.forEach(f => featureValues[f] = 0);

        // Reset scale buttons
        document.querySelectorAll('.scale-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.value === '0') btn.classList.add('active');
        });

        // Reset binary buttons
        document.querySelectorAll('.binary-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.value === '0') btn.classList.add('active');
        });

        // Reset age
        if (ageInput) ageInput.value = '';

        // Hide results
        document.getElementById('result-panel').classList.remove('visible');

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function showToast(message) {
        const toast = document.getElementById('toast');
        const msg = document.getElementById('toast-message');
        msg.textContent = message;
        toast.classList.add('visible');
        setTimeout(() => toast.classList.remove('visible'), 4000);
    }

    function addRipple(element) {
        element.style.transform = 'scale(0.95)';
        setTimeout(() => {
            element.style.transform = '';
        }, 150);
    }

    // ── Initialize default states (all at 0) ──
    document.querySelectorAll('.scale-btn[data-value="0"]').forEach(btn => {
        btn.classList.add('active');
    });
    document.querySelectorAll('.binary-btn[data-value="0"]').forEach(btn => {
        btn.classList.add('active');
    });
});
