// Global variables
let isTraining = false;
let isGeneratingReport = false;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Iris Classification System loaded');
    
    // Add event listeners
    const trainBtn = document.getElementById('trainBtn');
    if (trainBtn) {
        trainBtn.addEventListener('click', trainModels);
    }
    
    // Check system status on load
    checkSystemStatus();
});

// Modal functions
function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'block';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

// Training function
async function trainModels() {
    if (isTraining) return;
    
    isTraining = true;
    showLoadingModal('Training Models...', 'This may take a few minutes. Please wait.');
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                remove_outliers: true,
                use_grid_search: true
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Training failed');
        }
        
        const result = await response.json();
        closeModal('loadingModal');
        
        showSuccessModal(
            'Training Completed!',
            `Successfully trained ${result.models_trained.length} models. ` +
            `Best model: ${result.best_model}. Training time: ${result.training_time.toFixed(1)}s`
        );
        
        // Refresh page after 3 seconds
        setTimeout(() => {
            location.reload();
        }, 3000);
        
    } catch (error) {
        closeModal('loadingModal');
        showErrorModal('Training Failed', error.message);
    } finally {
        isTraining = false;
    }
}

// Report generation function
async function generateReport() {
    if (isGeneratingReport) return;
    
    isGeneratingReport = true;
    showLoadingModal('Generating Report...', 'Creating comprehensive analysis report with visualizations.');
    
    try {
        const response = await fetch('/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Report generation failed');
        }
        
        const result = await response.json();
        closeModal('loadingModal');
        
        showSuccessModal(
            'Report Generated!',
            `Report created successfully. <br><br>
            <a href="${result.download_url}" target="_blank" class="btn-primary">Download Report</a>`
        );
        
    } catch (error) {
        closeModal('loadingModal');
        showErrorModal('Report Generation Failed', error.message);
    } finally {
        isGeneratingReport = false;
    }
}

// System status check
async function checkSystemStatus() {
    try {
        const response = await fetch('/health');
        const status = await response.json();
        
        console.log('System Status:', status);
        
        // Update UI based on status
        updateSystemStatusUI(status);
        
    } catch (error) {
        console.error('Failed to check system status:', error);
    }
}

// Update system status UI
function updateSystemStatusUI(status) {
    const statusCards = document.querySelectorAll('.status-card');
    const actionCards = document.querySelectorAll('.action-card');
    
    if (status.models_trained) {
        // Models are trained - enable features
        statusCards.forEach(card => {
            if (card.classList.contains('warning')) {
                card.classList.remove('warning');
                card.classList.add('success');
                card.innerHTML = `
                    <h3>✅ System Ready</h3>
                    <p>Models are trained and ready for predictions</p>
                    <p><strong>Available Models:</strong> ${status.available_models.length}</p>
                `;
            }
        });
        
        // Enable disabled action cards
        actionCards.forEach(card => {
            if (card.classList.contains('disabled')) {
                card.classList.remove('disabled');
            }
        });
    }
}

// Prediction functions (for predict page)
function fillExample(species) {
    const examples = {
        'setosa': [5.1, 3.5, 1.4, 0.2],
        'versicolor': [7.0, 3.2, 4.7, 1.4],
        'virginica': [6.3, 3.3, 6.0, 2.5]
    };
    
    const values = examples[species];
    if (values) {
        document.getElementById('sepal_length').value = values[0];
        document.getElementById('sepal_width').value = values[1];
        document.getElementById('petal_length').value = values[2];
        document.getElementById('petal_width').value = values[3];
    }
}

// Reset prediction form
function resetForm() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.reset();
        
        // Hide result section if it exists
        const resultSection = document.querySelector('.result-section');
        if (resultSection) {
            resultSection.style.display = 'none';
        }
        
        // Clear any error highlights
        const inputs = form.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.style.borderColor = '#ddd';
        });
    }
}

// API prediction function (alternative to form submission)
async function makePrediction(features, modelName = null) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sepal_length: parseFloat(features.sepal_length),
                sepal_width: parseFloat(features.sepal_width),
                petal_length: parseFloat(features.petal_length),
                petal_width: parseFloat(features.petal_width),
                model_name: modelName
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

// Feature importance visualization (if needed)
async function getFeatureImportance(modelName = null) {
    try {
        const url = modelName ? `/feature-importance?model_name=${modelName}` : '/feature-importance';
        const response = await fetch(url);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get feature importance');
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('Feature importance error:', error);
        return null;
    }
}

// Modal helper functions
function showLoadingModal(title, subtitle) {
    document.getElementById('loadingText').textContent = title;
    document.getElementById('loadingSubtext').textContent = subtitle;
    showModal('loadingModal');
}

function showSuccessModal(title, message) {
    document.getElementById('successTitle').textContent = title;
    document.getElementById('successMessage').innerHTML = message;
    showModal('successModal');
}

function showErrorModal(title, message) {
    document.getElementById('errorMessage').textContent = message;
    showModal('errorModal');
}

// Form validation helper
function validatePredictionForm(form) {
    const inputs = form.querySelectorAll('input[type="number"]');
    let isValid = true;
    const errors = [];
    
    inputs.forEach(input => {
        const value = parseFloat(input.value);
        const fieldName = input.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        // Reset border color
        input.style.borderColor = '#ddd';
        
        if (isNaN(value)) {
            isValid = false;
            input.style.borderColor = '#ff4444';
            errors.push(`${fieldName} is required`);
        } else if (value < 0 || value > 10) {
            isValid = false;
            input.style.borderColor = '#ff4444';
            errors.push(`${fieldName} must be between 0 and 10 cm`);
        }
    });
    
    if (!isValid) {
        showErrorModal('Validation Error', errors.join('<br>'));
    }
    
    return isValid;
}

// Utility functions
function formatModelName(modelName) {
    const nameMap = {
        'random_forest': 'Random Forest',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'logistic_regression': 'Logistic Regression'
    };
    
    return nameMap[modelName] || modelName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatConfidence(confidence) {
    return `${(confidence * 100).toFixed(1)}%`;
}

function formatProbabilities(probabilities) {
    const formatted = {};
    for (const [species, prob] of Object.entries(probabilities)) {
        formatted[species] = `${(prob * 100).toFixed(1)}%`;
    }
    return formatted;
}

// Export functions for use in other scripts
window.trainModels = trainModels;
window.generateReport = generateReport;
window.fillExample = fillExample;
window.resetForm = resetForm;
window.makePrediction = makePrediction;
window.getFeatureImportance = getFeatureImportance;
window.closeModal = closeModal;
window.validatePredictionForm = validatePredictionForm;

// Close modals when clicking outside
window.addEventListener('click', function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Escape key closes modals
    if (event.key === 'Escape') {
        const visibleModal = document.querySelector('.modal[style*="block"]');
        if (visibleModal) {
            visibleModal.style.display = 'none';
        }
    }
    
    // Ctrl+Enter submits prediction form
    if (event.ctrlKey && event.key === 'Enter') {
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm && document.activeElement.form === predictionForm) {
            predictionForm.submit();
        }
    }
});
