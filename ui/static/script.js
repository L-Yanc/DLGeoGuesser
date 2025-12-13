// GeoGuesser Game Show - Frontend Logic

let stream = null;
let currentImage = null;

// DOM Elements
const fileUpload = document.getElementById('file-upload');
const cameraBtn = document.getElementById('camera-btn');
const cameraPreview = document.getElementById('camera-preview');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const analyzeBtn = document.getElementById('analyze-btn');
const statusLight = document.getElementById('status-light');
const statusText = document.getElementById('status-text');
const buzzer = document.getElementById('buzzer');

// Chat Elements
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send');
const chatMessages = document.getElementById('chat-messages');

// States
const idleState = document.getElementById('idle-state');
const processingState = document.getElementById('processing-state');
const resultsState = document.getElementById('results-state');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkPipelineStatus();
    setupEventListeners();
});

function setupEventListeners() {
    fileUpload.addEventListener('change', handleFileUpload);
    cameraBtn.addEventListener('click', startCamera);
    captureBtn.addEventListener('click', captureImage);
    closeCameraBtn.addEventListener('click', stopCamera);
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Chat listeners
    chatSendBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !chatInput.disabled) {
            sendChatMessage();
        }
    });
    
    const clearChatBtn = document.getElementById('clear-chat-btn');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearConversation);
    }
}

function checkPipelineStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.ready) {
                setStatus('ready', 'READY');
                checkChatModelsStatus();
            } else {
                setStatus('processing', 'LOADING...');
                setTimeout(checkPipelineStatus, 2000);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            setStatus('processing', 'ERROR');
        });
}

function checkChatModelsStatus() {
    fetch('/api/chat/status')
        .then(response => response.json())
        .then(data => {
            const modelSelect = document.getElementById('llm-model-select');
            
            // Enable/disable options based on availability
            for (let option of modelSelect.options) {
                const modelType = option.value;
                if (data.models && data.models[modelType] !== undefined) {
                    option.disabled = !data.models[modelType];
                    
                    // Add status indicator
                    if (data.models[modelType]) {
                        option.text = option.text.replace(' (unavailable)', '');
                    } else {
                        if (!option.text.includes('(unavailable)')) {
                            option.text += ' (unavailable)';
                        }
                    }
                }
            }
            
            console.log('Chat models status:', data.models);
        })
        .catch(error => {
            console.error('Error checking chat models:', error);
        });
}

function setStatus(state, text) {
    statusLight.className = `status-light ${state}`;
    statusText.textContent = text;
}

// File Upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = e.target.result;
            showImagePreview(currentImage);
        };
        reader.readAsDataURL(file);
    }
}

// Camera Functions
async function startCamera() {
    // Toggle camera if already open
    if (cameraPreview.style.display === 'block') {
        stopCamera();
        return;
    }
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
        cameraPreview.style.display = 'block';
        imagePreview.style.display = 'none';
        
        // Update button text
        cameraBtn.innerHTML = '<span class="btn-icon">✖</span><span class="btn-text">CLOSE CAMERA</span>';
    } catch (error) {
        alert('Could not access camera: ' + error.message);
    }
}

function captureImage() {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    currentImage = canvas.toDataURL('image/jpeg');
    stopCamera();
    showImagePreview(currentImage);
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraPreview.style.display = 'none';
    
    // Reset button text
    cameraBtn.innerHTML = '<span class="btn-icon">📷</span><span class="btn-text">USE CAMERA</span>';
}

function showImagePreview(imageSrc) {
    previewImg.src = imageSrc;
    imagePreview.style.display = 'block';
    cameraPreview.style.display = 'none';
}

// Analysis
async function analyzeImage() {
    if (!currentImage) {
        alert('Please upload or capture an image first!');
        return;
    }

    // Get selected model
    const selectedModel = document.getElementById('llm-model-select').value;
    
    // Check if VLA is selected but not available
    if (selectedModel === 'vla_finetuned') {
        const modelSelect = document.getElementById('llm-model-select');
        const selectedOption = modelSelect.options[modelSelect.selectedIndex];
        if (selectedOption.disabled || selectedOption.text.includes('(unavailable)')) {
            const proceed = confirm(
                'VLA server is not available. The analysis will run without LLM explanations.\n\n' +
                'To use VLA:\n' +
                '1. Start the VLA server: python scripts/start_vla_server.py\n' +
                '2. Or set VLA_SERVER_URL to a remote server\n\n' +
                'Continue without VLA?'
            );
            if (!proceed) {
                return;
            }
        }
    }
    
    // Show processing state
    showState('processing');
    setStatus('processing', 'ANALYZING...');
    buzzer.classList.remove('active');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                image: currentImage,
                model: selectedModel,
                session_id: sessionId
            })
        });

        const data = await response.json();

        if (data.success) {
            // Activate buzzer immediately when results arrive
            buzzer.classList.add('active');
            playBuzzerSound();
            setStatus('ready', 'COMPLETE!');
            
            // Then display results
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
            showState('idle');
            setStatus('ready', 'READY');
        }
    } catch (error) {
        console.error('Error analyzing image:', error);
        alert('Error analyzing image: ' + error.message);
        showState('idle');
        setStatus('ready', 'READY');
    }
}

function showState(state) {
    idleState.style.display = state === 'idle' ? 'block' : 'none';
    processingState.style.display = state === 'processing' ? 'block' : 'none';
    resultsState.style.display = state === 'results' ? 'block' : 'none';
}

function displayResults(data) {
    console.log('📊 Displaying results:', data);
    showState('results');
    
    // Store results for chat context
    currentResults = data;
    
    // Update chat model info
    const modelSelect = document.getElementById('llm-model-select');
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    document.getElementById('current-chat-model').textContent = selectedOption.text;
    
    // Enable chat after results are displayed
    enableChat();

    // Visualization
    const vizImg = document.getElementById('result-viz');
    const vizContainer = document.getElementById('visualization-container');
    const vizLoading = document.getElementById('viz-loading');
    
    console.log('🖼️ Visualization data:', data.visualizations);
    
    if (data.visualizations && data.visualizations.grid) {
        // Show loading state
        vizContainer.style.display = 'block';
        vizContainer.style.opacity = '0.5';
        vizImg.style.display = 'none';
        if (vizLoading) {
            vizLoading.style.display = 'block';
        }
        
        // Load image
        const img = new Image();
        img.onload = function() {
            vizImg.src = img.src;
            vizImg.style.display = 'block';
            vizContainer.style.opacity = '1';
            if (vizLoading) {
                vizLoading.style.display = 'none';
            }
            console.log('✅ Visualization loaded:', data.visualizations.grid);
        };
        img.onerror = function() {
            console.error('❌ Failed to load visualization:', data.visualizations.grid);
            vizContainer.style.display = 'none';
            if (vizLoading) {
                vizLoading.textContent = 'Visualization unavailable';
                vizLoading.style.color = 'var(--bright-coral)';
            }
        };
        img.src = data.visualizations.grid + '?t=' + Date.now();
    } else {
        console.warn('⚠️ No visualization data available');
        vizContainer.style.display = 'none';
    }

    // Country
    document.getElementById('country-name').textContent = data.country.top || '---';
    document.getElementById('country-confidence').textContent = data.country.confidence + '%';
    
    const countryTop5 = document.getElementById('country-top5');
    countryTop5.innerHTML = data.country.top_5.map(item => `
        <div class="top5-item">
            <span>${item.name}</span>
            <span>${item.confidence}%</span>
        </div>
    `).join('');

    // Vibe
    document.getElementById('vibe-name').textContent = data.vibe.top || '---';
    document.getElementById('vibe-confidence').textContent = data.vibe.confidence + '%';
    
    const vibeTop5 = document.getElementById('vibe-top5');
    vibeTop5.innerHTML = data.vibe.top_5.map(item => `
        <div class="top5-item">
            <span>${item.name}</span>
            <span>${item.confidence}%</span>
        </div>
    `).join('');

    // Detections
    const detectionsList = document.getElementById('detections-list');
    if (data.detections.count > 0) {
        detectionsList.innerHTML = data.detections.objects.map(obj => 
            `<span class="detection-tag">${obj}</span>`
        ).join('');
    } else {
        detectionsList.innerHTML = '<p>No objects detected</p>';
    }

    // LLM Analysis
    console.log('🤖 LLM analysis:', data.llm_analysis);
    const llmBox = document.getElementById('llm-analysis-box');
    const llmContent = document.getElementById('llm-analysis-content');
    const llmBadge = document.getElementById('llm-model-badge');
    
    if (data.llm_analysis && data.llm_analysis.analysis) {
        llmBox.style.display = 'block';
        llmContent.textContent = data.llm_analysis.analysis;
        
        // Set model badge
        const modelName = data.llm_analysis.model.replace('self_trained_', '').toUpperCase();
        llmBadge.textContent = `${modelName} Model`;
    } else {
        llmBox.style.display = 'none';
    }

    // OCR Results
    console.log('📝 OCR data:', data.ocr);
    const ocrBox = document.getElementById('ocr-box');
    const ocrText = document.getElementById('ocr-text');
    const ocrLanguages = document.getElementById('ocr-languages');
    
    if (data.ocr && data.ocr.has_text) {
        console.log('✅ Showing OCR box with text:', data.ocr.text);
        ocrBox.style.display = 'block';
        
        // Display extracted text
        if (data.ocr.text) {
            // Handle both string and array formats
            const textArray = Array.isArray(data.ocr.text) ? data.ocr.text : [data.ocr.text];
            
            if (textArray.length > 0 && textArray[0]) {
                ocrText.innerHTML = '<div class="ocr-label">Extracted Text:</div>' +
                    textArray.map(text => 
                        `<div class="ocr-text-item">"${text}"</div>`
                    ).join('');
            } else {
                ocrText.innerHTML = '<div class="ocr-text-item">No text extracted</div>';
            }
        } else {
            ocrText.innerHTML = '<div class="ocr-text-item">No text extracted</div>';
        }
        
        // Display detected languages
        if (data.ocr.languages && data.ocr.languages.length > 0) {
            ocrLanguages.innerHTML = '<div class="ocr-label">Languages:</div>' +
                '<div class="ocr-lang-tags">' +
                data.ocr.languages.map(lang => 
                    `<span class="ocr-lang-tag">${lang}</span>`
                ).join('') +
                '</div>';
        } else {
            ocrLanguages.innerHTML = '';
        }
    } else {
        ocrBox.style.display = 'none';
    }

    // Processing time
    document.getElementById('processing-time').textContent = data.processing_time;
}

function playBuzzerSound() {
    // Create a simple beep sound using Web Audio API
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}

// Chat Functions
let currentResults = null;
let sessionId = 'session_' + Date.now(); // Unique session ID

function enableChat() {
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatInput.placeholder = "Ask about the location...";
}

function disableChat() {
    chatInput.disabled = true;
    chatSendBtn.disabled = true;
    chatInput.placeholder = "Analyze an image first...";
}

function addChatMessage(text, isUser = false, model = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user' : 'assistant'}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'chat-bubble';
    bubbleDiv.textContent = text;
    
    messageDiv.appendChild(bubbleDiv);
    
    if (!isUser && model) {
        const badgeDiv = document.createElement('div');
        badgeDiv.className = 'chat-model-badge';
        // Format model name nicely
        const modelName = model.replace('self_trained_', '').toUpperCase();
        badgeDiv.textContent = `${modelName}`;
        messageDiv.appendChild(badgeDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearChatWelcome() {
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) {
        welcome.remove();
    }
}

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Clear welcome message on first chat
    clearChatWelcome();
    
    // Add user message
    addChatMessage(message, true);
    chatInput.value = '';
    
    // Get selected model from dropdown
    const selectedModel = document.getElementById('llm-model-select').value;
    
    // Show loading
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-loading';
    loadingDiv.textContent = 'Thinking...';
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
        const response = await fetch('/api/chat/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                model: selectedModel,
                session_id: sessionId,
                context: currentResults,  // Send full analysis results as context
                max_tokens: 150,
                temperature: 0.8
            })
        });
        
        const data = await response.json();
        
        // Remove loading
        loadingDiv.remove();
        
        if (data.success) {
            addChatMessage(data.response, false, selectedModel);
        } else {
            const errorMsg = data.error || 'Unknown error';
            if (data.available_models) {
                addChatMessage(`${errorMsg}. Available: ${data.available_models.join(', ')}`, false);
            } else {
                addChatMessage(`Error: ${errorMsg}`, false);
            }
        }
    } catch (error) {
        loadingDiv.remove();
        addChatMessage(`Error: ${error.message}`, false);
        console.error('Chat error:', error);
    }
}

function clearConversation() {
    // Clear UI
    chatMessages.innerHTML = '<div class="chat-welcome">Ask me about the location! 🌍</div>';
    
    // Clear server-side history
    fetch('/api/chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
    }).catch(err => console.error('Failed to clear conversation:', err));
}
