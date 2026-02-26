document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded, initializing functionality");
    
    // Initialize upload form
    initUploadForm();
    
    // Initialize image upload functionality
    initImageUpload();
    
    // Initialize webcam functionality
    initWebcam();
    
    // Initialize restart button
    initRestartButton();
    
    // Setup charts
    initDeepfakeChart();
    
    // Check for previous analysis to restore
    checkForPreviousAnalysis();
});

// Global variables
const MAX_ANALYSIS_TIME = 30000; // 30 seconds max before forcing completion
const MAX_STALLED_TIME = 5000; // 5 seconds of no progress before advancing

// Global variables for managing video analysis state
let uploadedFileId = null;
let analysisInProgress = false;
let lastAnalysisTimestamp = 0;
let resultRefreshAttempts = 0;
const MAX_REFRESH_ATTEMPTS = 10;

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const videoUploadInput = document.getElementById('video-upload');
const uploadVideoPreview = document.getElementById('upload-video-preview');
const analyzeVideoBtn = document.getElementById('upload-btn');
const uploadStatusElement = document.getElementById('upload-progress');
const uploadProgressBar = uploadStatusElement ? uploadStatusElement.querySelector('.progress-bar') : null;
const uploadStatusText = uploadProgressBar;
const resultsSection = document.getElementById('results-section');
const timeline = document.getElementById('timeline');
const timelineMarker = document.getElementById('timeline-marker');
const resultsTable = document.getElementById('results-table') ? document.getElementById('results-table').querySelector('tbody') : null;
const resultSummary = document.getElementById('result-summary');
const modelUsed = document.getElementById('model-used');

// Initialize upload form
function initUploadForm() {
    console.log("Initializing upload form");
    
    // The video loading is now handled by video-fix.js
    // Just handle the form submission
    if (analyzeVideoBtn) {
        analyzeVideoBtn.addEventListener('click', async function() {
            if (videoUploadInput && videoUploadInput.files.length > 0) {
                await handleVideoUpload();
            } else {
                showMessage('Please select a video file first', 'warning');
            }
        });
    }
}

// Show messages to the user
function showMessage(message, type = 'info') {
    const alertElement = document.getElementById('upload-alert');
    if (alertElement) {
        alertElement.textContent = message;
        alertElement.className = `alert alert-${type} mt-3`;
        alertElement.classList.remove('d-none');
        
        // Auto-hide success and info messages after 5 seconds
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                alertElement.classList.add('d-none');
            }, 5000);
        }
    }
}

// Get color for confidence value
function getColorForConfidence(confidence) {
    // Red for high confidence (likely fake)
    if (confidence > 0.7) {
        return `rgba(255, 0, 0, ${confidence})`;
    }
    // Yellow for uncertain
    else if (confidence > 0.3) {
        return `rgba(255, 165, 0, 0.7)`;
    }
    // Green for low confidence (likely real)
    else {
        return `rgba(0, 128, 0, ${1 - confidence})`;
    }
}

// Initialize deepfake chart
function initDeepfakeChart() {
    const uploadChartCanvas = document.getElementById('upload-deepfake-chart');
    if (uploadChartCanvas) {
        window.uploadDeepfakeChartInstance = new Chart(uploadChartCanvas, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Deepfake Confidence',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'category',
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Confidence (0-1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Confidence: ${(context.raw.value * 100).toFixed(1)}%`;
                            },
                            title: function(context) {
                                return `Time: ${context[0].label}`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Update deepfake chart with new data
function updateDeepfakeChart(chart, data) {
    if (!chart) return;
    
    chart.data.labels = data.map(item => item.label);
    chart.data.datasets[0].data = data.map(item => {
        return {
            x: item.label,
            y: item.value,
            value: item.value,
            timestamp: item.timestamp
        };
    });
    
    // Update colors based on confidence
    chart.data.datasets[0].backgroundColor = data.map(item => {
        return getColorForConfidence(item.value).replace(')', ', 0.2)');
    });
    
    chart.data.datasets[0].borderColor = data.map(item => {
        return getColorForConfidence(item.value);
    });
    
    chart.update();
}

// Handle video upload process
async function handleVideoUpload() {
    try {
        if (!videoUploadInput.files.length) {
            showMessage('Please select a video file first', 'warning');
            return;
        }
        
        const file = videoUploadInput.files[0];
        if (!file.type.startsWith('video/')) {
            showMessage('Please upload a valid video file', 'danger');
            return;
        }
        
        // Reset analysis state for new upload
        resetAnalysisState();
        
        // Show upload UI state
        uploadStatusElement.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        uploadProgressBar.style.width = '0%';
        uploadProgressBar.setAttribute('aria-valuenow', 0);
        uploadProgressBar.textContent = 'Uploading video...';
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', document.getElementById('model-select')?.value || 'mesonet');
        
        // Show message about fresh analysis
        showMessage('Starting fresh analysis with clean environment...', 'info');
        
        // Upload the file
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        uploadedFileId = data.file_id;
        
        // Update progress bar to show upload complete
        uploadProgressBar.style.width = '100%';
        uploadProgressBar.setAttribute('aria-valuenow', 100);
        uploadProgressBar.textContent = 'Upload complete, processing video...';
        
        showMessage('Video uploaded successfully. Processing with fresh analysis environment...', 'success');
        
        // Start polling for status
        startProgressPolling(uploadedFileId);
        
    } catch (error) {
        console.error('Error uploading video:', error);
        showMessage(`Error: ${error.message}`, 'danger');
        uploadStatusElement.classList.add('d-none');
    }
}

// Reset all analysis state to ensure clean results
function resetAnalysisState() {
    uploadedFileId = null;
    analysisInProgress = false;
    resultRefreshAttempts = 0;
    lastAnalysisTimestamp = Date.now();
    
    // Clear any previous results
    if (timeline) timeline.innerHTML = '';
    if (resultsTable) resultsTable.innerHTML = '';
    if (window.videoResults) window.videoResults = null;
    
    // Reset any progress indicators
    if (uploadProgressBar) {
        uploadProgressBar.style.width = '0%';
        uploadProgressBar.setAttribute('aria-valuenow', 0);
    }
    
    console.log('Analysis state reset');
}

// Request video analysis with cache-busting
async function requestAnalysis(fileId) {
    try {
        const response = await fetch(`/result/${fileId}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                // Results not ready yet, schedule another attempt
                resultRefreshAttempts++;
                if (resultRefreshAttempts < MAX_REFRESH_ATTEMPTS) {
                    console.log(`Results not ready, retrying in 2s (attempt ${resultRefreshAttempts}/${MAX_REFRESH_ATTEMPTS})`);
                    setTimeout(() => requestAnalysis(fileId), 2000);
                } else {
                    showMessage('Could not retrieve results after multiple attempts', 'danger');
                }
            } else {
                throw new Error(`HTTP error: ${response.status}`);
            }
            return;
        }
        
        // Reset attempts counter
        resultRefreshAttempts = 0;
        
        // Process results
        const data = await response.json();
        console.log('Analysis results received:', data);
        
        // Show session information
        const processingTime = data.processing_time ? data.processing_time.toFixed(1) : "?";
        showMessage(`Analysis completed in session ${data.session_id || 'unknown'} (${processingTime}s)`, 'success');
        
        // Process and display results
        processResults(data);
        
        // Hide progress bar after a delay
        setTimeout(() => {
            uploadStatusElement.classList.add('d-none');
        }, 2000);
        
    } catch (error) {
        console.error('Error requesting analysis:', error);
        showMessage(`Error retrieving results: ${error.message}`, 'danger');
        
        // Stop loading state
        if (uploadStatusElement) uploadStatusElement.classList.add('d-none');
        analysisInProgress = false;
    }
}

// Schedule a result refresh attempt to ensure fresh results
function scheduleResultRefresh(fileId) {
    if (resultRefreshAttempts >= MAX_REFRESH_ATTEMPTS) {
        console.error('Max refresh attempts reached, giving up');
        showMessage('Could not get fresh analysis results', 'danger');
        uploadStatusElement.classList.add('d-none');
        analysisInProgress = false;
        return;
    }
    
    resultRefreshAttempts++;
    console.log(`Scheduling result refresh attempt ${resultRefreshAttempts}...`);
    
    // Add randomized delay to avoid synchronization issues
    const delay = 1000 + (Math.random() * 500);
    
    setTimeout(() => {
        console.log('Refreshing results...');
        requestAnalysis(fileId);
    }, delay);
}

// Monitor progress of video processing
function startProgressPolling(fileId) {
    console.log(`Starting progress polling for file ${fileId}`);
    let pollCount = 0;
    const maxPolls = 20; // Maximum 20 polls (40 seconds at 2-second intervals)
    
    // Show polling status
    uploadProgressBar.textContent = 'Processing video...';
    
    // Force completion after a certain time
    const forceCompletionTimeout = setTimeout(() => {
        console.log("Force completing analysis due to timeout");
        requestAnalysis(fileId);
    }, 20000); // 20 seconds max wait time (reduced from 45)
    
    // Show faster progress updates to improve perceived performance
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 5;
        if (progress > 90) {
            clearInterval(progressInterval);
        }
        uploadProgressBar.style.width = `${progress}%`;
        uploadProgressBar.setAttribute('aria-valuenow', progress);
        uploadProgressBar.textContent = `Processing video... ${progress}%`;
    }, 1000); // Update every second
    
    const pollInterval = setInterval(async () => {
        try {
            pollCount++;
            
            if (pollCount > maxPolls) {
                clearInterval(pollInterval);
                clearInterval(progressInterval);
                clearTimeout(forceCompletionTimeout);
                showMessage('Processing timed out. Attempting to display available results.', 'warning');
                // Try to get whatever results are available
                requestAnalysis(fileId);
                return;
            }
            
            const response = await fetch(`/status/${fileId}`);
            const data = await response.json();
            
            // Update progress based on status
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                clearInterval(progressInterval);
                clearTimeout(forceCompletionTimeout);
                
                // Show completion info
                const processingTime = data.processing_time ? data.processing_time.toFixed(1) : "?";
                const frameCount = data.frame_count || 0;
                
                uploadProgressBar.style.width = '100%';
                uploadProgressBar.setAttribute('aria-valuenow', 100);
                uploadProgressBar.textContent = `Completed! Processed ${frameCount} frames in ${processingTime}s`;
                
                // Show session info
                console.log(`Analysis completed in session: ${data.session_id}`);
                
                // Fetch and display results
                await requestAnalysis(fileId);
                
            } else if (data.status === 'processing') {
                // Progress updates are handled by the progressInterval
            } else {
                clearInterval(pollInterval);
                clearInterval(progressInterval);
                clearTimeout(forceCompletionTimeout);
                showMessage(`Error: ${data.message || 'Unknown error'}`, 'danger');
                uploadStatusElement.classList.add('d-none');
            }
            
        } catch (error) {
            console.error('Error polling status:', error);
            // Don't clear interval, keep trying
        }
    }, 2000); // Poll every 2 seconds (reduced from 5)
}

// Generate a unique session ID for this analysis
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 10);
}

// Process and display results
function processResults(data) {
    // Validate data structure
    if (!data || !data.results || !Array.isArray(data.results) || data.results.length === 0) {
        showMessage('Invalid or empty results data', 'warning');
        return;
    }
    
    // Store file ID for persistence
    if (data.file_id) {
        localStorage.setItem('last_analysis_file_id', data.file_id);
        
        // Update URL with file ID for sharing/bookmarking
        const url = new URL(window.location.href);
        url.searchParams.set('file_id', data.file_id);
        window.history.replaceState({}, '', url);
    }
    
    // Show results section
    resultsSection.classList.remove('d-none');
    
    // Update model information
    if (data.model && modelUsed) {
        modelUsed.textContent = `${data.model.name} - ${data.model.description}`;
    }
    
    // Extract video information
    const videoInfo = data.video_info || {
        total_frames: 1,
        fps: 30,
        width: 640,
        height: 480
    };
    
    const fps = videoInfo.fps || 30;
    const totalFrames = videoInfo.total_frames || 1;
    const videoDuration = totalFrames / fps;
    
    // Process results
    const results = data.results || [];
    
    // Calculate average confidence
    let totalConfidence = 0;
    results.forEach(result => {
        totalConfidence += (result.confidence_fake || 0);
    });
    const avgConfidence = results.length > 0 ? totalConfidence / results.length : 0;
    const confidencePercent = Math.round(avgConfidence * 100);
    
    // Update result summary
    if (resultSummary) {
        if (avgConfidence > 0.7) {
            resultSummary.className = 'alert alert-danger';
            resultSummary.textContent = `This video is likely a deepfake (${confidencePercent}% confidence)`;
        } else if (avgConfidence > 0.3) {
            resultSummary.className = 'alert alert-warning';
            resultSummary.textContent = `This video may contain manipulated elements (${confidencePercent}% confidence)`;
        } else {
            resultSummary.className = 'alert alert-success';
            resultSummary.textContent = `This video appears to be authentic (${(100-confidencePercent)}% confidence)`;
        }
    }
    
    // Generate timeline
    generateTimeline(results, totalFrames, fps, videoDuration);
    
    // Create chart data
    const chartData = results.map(result => {
        return {
            label: ((result.frame || 0) / fps).toFixed(2) + 's',
            value: result.confidence_fake || 0,
            timestamp: (result.frame || 0) / fps
        };
    });
    
    // Update chart
    if (window.uploadDeepfakeChartInstance) {
        updateDeepfakeChart(window.uploadDeepfakeChartInstance, chartData);
    }
    
    // Populate results table
    populateResultsTable(results, fps);
}

// Generate timeline visualization
function generateTimeline(results, totalFrames, fps, videoDuration) {
    if (!timeline) return;
    if (!results || !Array.isArray(results) || results.length === 0) return;
    
    // Ensure we have valid parameters
    totalFrames = totalFrames || 1;
    fps = fps || 30;
    videoDuration = videoDuration || (totalFrames / fps);
    
    // Clear existing timeline
    timeline.innerHTML = '';
    
    // Add timeline points
    results.forEach(result => {
        const frame = result.frame || 0;
        const framePos = frame / totalFrames;
        const leftPos = framePos * 100;
        
        // Create timeline point
        const point = document.createElement('div');
        point.className = 'timeline-point';
        point.style.left = `${leftPos}%`;
        
        // Set color based on confidence
        const confidence = result.confidence_fake || 0;
        const color = getColorForConfidence(confidence);
        point.style.backgroundColor = color;
        
        // Add data attributes
        point.setAttribute('data-frame', frame);
        point.setAttribute('data-confidence', confidence);
        point.setAttribute('data-time', (frame / fps).toFixed(2));
        
        // Add click event
        point.addEventListener('click', function() {
            // Highlight this point
            const allPoints = timeline.querySelectorAll('.timeline-point');
            allPoints.forEach(p => p.classList.remove('highlighted'));
            point.classList.add('highlighted');
            
            // Update marker position
            if (timelineMarker) {
                timelineMarker.style.left = `${leftPos}%`;
            }
            
            // Seek video to this position if available
            if (uploadVideoPreview) {
                uploadVideoPreview.currentTime = frame / fps;
            }
            
            // Show frame details (if you have a function for this)
            if (typeof showFrameDetails === 'function') {
                showFrameDetails(result);
            }
        });
        
        timeline.appendChild(point);
    });
    
    // Add time markers
    addTimelineSeparator(0, '0:00');
    
    // Add markers every 30 seconds
    for (let i = 30; i < videoDuration; i += 30) {
        const pos = i / videoDuration * 100;
        const minutes = Math.floor(i / 60);
        const seconds = i % 60;
        const label = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        addTimelineSeparator(pos, label);
    }
    
    // Add end marker
    const totalMinutes = Math.floor(videoDuration / 60);
    const totalSeconds = Math.floor(videoDuration % 60);
    const endLabel = `${totalMinutes}:${totalSeconds.toString().padStart(2, '0')}`;
    addTimelineSeparator(100, endLabel);
}

// Helper function to add timeline separators
function addTimelineSeparator(position, label) {
    if (!timeline) return;
    
    const separator = document.createElement('div');
    separator.className = 'timeline-separator';
    separator.style.left = `${position}%`;
    
    const labelElem = document.createElement('div');
    labelElem.className = 'timeline-label';
    labelElem.textContent = label;
    
    separator.appendChild(labelElem);
    timeline.appendChild(separator);
}

// Populate results table
function populateResultsTable(results, fps) {
    if (!resultsTable) return;
    if (!results || !Array.isArray(results) || results.length === 0) return;
    
    // Ensure we have a valid fps
    fps = fps || 30;
    
    // Clear existing table
    resultsTable.innerHTML = '';
    
    // Show only a subset of results to avoid overwhelming the table
    const maxTableRows = 20;
    const step = Math.max(1, Math.floor(results.length / maxTableRows));
    
    for (let i = 0; i < results.length; i += step) {
        const result = results[i];
        if (!result) continue;
        
        const frame = result.frame || 0;
        const confidence = result.confidence_fake || 0;
        const timeInSeconds = (frame / fps).toFixed(2);
        const confidencePercent = (confidence * 100).toFixed(1);
        
        const row = document.createElement('tr');
        
        // Highlight high confidence rows
        if (confidence > 0.7) {
            row.className = 'high-confidence';
        } else if (confidence < 0.3) {
            row.className = 'low-confidence';
        }
        
        // Create a visual indicator for confidence level
        const confidenceBar = `
            <div class="confidence-indicator" style="
                width: ${confidencePercent}%; 
                height: 10px; 
                background-color: ${getConfidenceColor(confidence)};
                border-radius: 2px;
            "></div>
        `;
        
        row.innerHTML = `
            <td>${frame}</td>
            <td>${timeInSeconds}s</td>
            <td>
                ${confidencePercent}%
                ${confidenceBar}
            </td>
        `;
        
        // Make the entire row clickable to highlight the corresponding point in the timeline
        row.style.cursor = 'pointer';
        row.addEventListener('click', function() {
            highlightTimelinePoint(frame, fps);
        });
        
        resultsTable.appendChild(row);
    }
}

// Get color for confidence indicator based on value
function getConfidenceColor(confidence) {
    if (confidence > 0.7) {
        return '#dc3545'; // Red for high confidence (likely fake)
    } else if (confidence > 0.3) {
        return '#ffc107'; // Yellow for uncertain
    } else {
        return '#28a745'; // Green for low confidence (likely real)
    }
}

// Highlight a point in the timeline
function highlightTimelinePoint(frame, fps) {
    if (!timeline) return;
    
    // Find the closest point in the timeline
    const points = timeline.querySelectorAll('.timeline-point');
    let closestPoint = null;
    let minDistance = Infinity;
    
    points.forEach(point => {
        const pointFrame = parseInt(point.getAttribute('data-frame'));
        const distance = Math.abs(pointFrame - frame);
        
        if (distance < minDistance) {
            minDistance = distance;
            closestPoint = point;
        }
    });
    
    if (closestPoint) {
        // Remove highlighting from all points
        points.forEach(p => p.classList.remove('highlighted'));
        
        // Highlight the closest point
        closestPoint.classList.add('highlighted');
        
        // Update timeline marker position
        if (timelineMarker) {
            const framePos = parseInt(closestPoint.getAttribute('data-frame')) / parseInt(timeline.getAttribute('data-total-frames') || 1);
            const leftPos = framePos * 100;
            timelineMarker.style.left = `${leftPos}%`;
        }
        
        // Seek video to this position if available
        if (uploadVideoPreview) {
            uploadVideoPreview.currentTime = frame / fps;
        }
        
        // Show frame details
        const confidence = parseFloat(closestPoint.getAttribute('data-confidence'));
        showFrameDetails({
            frame: frame,
            confidence_fake: confidence
        });
    }
}

// Initialize webcam functionality
function initWebcam() {
    console.log("Initializing webcam functionality");
    
    const webcamVideo = document.getElementById('webcam-video');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const startWebcamBtn = document.getElementById('start-webcam-btn');
    const stopWebcamBtn = document.getElementById('stop-webcam-btn');
    const webcamStatus = document.getElementById('webcam-status');
    const fakeProbability = document.getElementById('fake-probability');
    const connectionIndicator = document.getElementById('connection-indicator');
    const connectionText = document.getElementById('connection-text');
    const cameraResolution = document.getElementById('camera-resolution');
    const webcamModelSelect = document.getElementById('webcam-model-select');
    
    let stream = null;
    let websocket = null;
    let webcamChart = null;
    let isAnalyzing = false;
    let dataPoints = [];
    
    // Initialize webcam chart
    const webcamChartCanvas = document.getElementById('deepfake-chart');
    if (webcamChartCanvas) {
        webcamChart = new Chart(webcamChartCanvas, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Deepfake Confidence',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'second',
                            displayFormats: {
                                second: 'HH:mm:ss'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Confidence (0-1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Confidence: ${(context.raw.y * 100).toFixed(1)}%`;
                            }
                        }
                    }
                },
                animation: {
                    duration: 0 // Disable animations for performance
                }
            }
        });
    }
    
    // Clear chart button
    const clearChartBtn = document.getElementById('clear-chart');
    if (clearChartBtn) {
        clearChartBtn.addEventListener('click', function() {
            if (webcamChart) {
                dataPoints = [];
                updateWebcamChart();
            }
        });
    }
    
    // Start webcam button
    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', startWebcamAnalysis);
    }
    
    // Stop webcam button
    if (stopWebcamBtn) {
        stopWebcamBtn.addEventListener('click', stopWebcamAnalysis);
    }
    
    // Handle tab change to ensure webcam stops when switching tabs
    const tabs = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            // If switching away from webcam tab
            if (event.relatedTarget && event.relatedTarget.id === 'webcam-tab') {
                stopWebcamAnalysis();
            }
        });
    });
    
    async function startWebcamAnalysis() {
        if (isAnalyzing) return;
        
        try {
            // Request camera access
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user"
                }, 
                audio: false 
            });
            
            // Connect webcam stream to video element
            webcamVideo.srcObject = stream;
            
            // Wait for video to be ready
            await new Promise(resolve => {
                webcamVideo.onloadedmetadata = resolve;
            });
            
            // Start playing video
            await webcamVideo.play();
            
            // Set up overlay canvas
            if (overlayCanvas) {
                overlayCanvas.width = webcamVideo.videoWidth;
                overlayCanvas.height = webcamVideo.videoHeight;
            }
            
            // Update camera resolution info
            if (cameraResolution) {
                cameraResolution.textContent = `${webcamVideo.videoWidth}x${webcamVideo.videoHeight}`;
            }
            
            // Connect to WebSocket for analysis
            const model = webcamModelSelect ? webcamModelSelect.value : 'mesonet';
            connectWebSocket(model);
            
            // Update UI
            startWebcamBtn.classList.add('d-none');
            stopWebcamBtn.classList.remove('d-none');
            webcamStatus.className = 'alert alert-info';
            webcamStatus.innerHTML = '<div class="recording-indicator"></div> Analyzing webcam feed...';
            
            isAnalyzing = true;
            
        } catch (error) {
            console.error('Error starting webcam:', error);
            webcamStatus.className = 'alert alert-danger';
            webcamStatus.textContent = `Error: ${error.message || 'Could not access camera'}`;
        }
    }
    
    function stopWebcamAnalysis() {
        // Stop webcam stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        
        // Close WebSocket connection
        if (websocket) {
            websocket.close();
            websocket = null;
        }
        
        // Update UI
        if (startWebcamBtn) startWebcamBtn.classList.remove('d-none');
        if (stopWebcamBtn) stopWebcamBtn.classList.add('d-none');
        if (webcamStatus) {
            webcamStatus.className = 'alert alert-info';
            webcamStatus.textContent = 'Click "Start Detection" to begin analysis';
        }
        
        // Reset connection status
        if (connectionIndicator) connectionIndicator.style.backgroundColor = '#ccc';
        if (connectionText) connectionText.textContent = 'Not connected';
        
        isAnalyzing = false;
    }
    
    function connectWebSocket(model) {
        // Close existing connection
        if (websocket) {
            websocket.close();
        }
        
        // Create new WebSocket connection
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/webcam?model=${model}`;
        websocket = new WebSocket(wsUrl);
        
        // Connection opened
        websocket.onopen = function(event) {
            console.log('WebSocket connection established');
            connectionIndicator.style.backgroundColor = '#28a745';
            connectionText.textContent = 'Connected';
        };
        
        // Connection closed
        websocket.onclose = function(event) {
            console.log('WebSocket connection closed');
            connectionIndicator.style.backgroundColor = '#dc3545';
            connectionText.textContent = 'Disconnected';
            
            // Try to reconnect if still analyzing
            if (isAnalyzing) {
                setTimeout(() => {
                    if (isAnalyzing) {
                        connectWebSocket(model);
                    }
                }, 2000);
            }
        };
        
        // Connection error
        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
            connectionIndicator.style.backgroundColor = '#dc3545';
            connectionText.textContent = 'Connection error';
        };
        
        // Receive message
        websocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                // Handle error
                if (data.error) {
                    console.error('WebSocket error:', data.error);
                    webcamStatus.className = 'alert alert-warning';
                    webcamStatus.textContent = `Warning: ${data.error}`;
                    return;
                }
                
                // Update confidence display
                if (data.confidence_fake !== undefined) {
                    const confidencePercent = Math.round(data.confidence_fake * 100);
                    
                    // Update progress bar
                    if (fakeProbability) {
                        fakeProbability.style.width = `${confidencePercent}%`;
                        fakeProbability.setAttribute('aria-valuenow', confidencePercent);
                        fakeProbability.textContent = `${confidencePercent}%`;
                        
                        // Update color based on confidence
                        if (data.confidence_fake > 0.7) {
                            fakeProbability.className = 'progress-bar bg-danger';
                        } else if (data.confidence_fake > 0.3) {
                            fakeProbability.className = 'progress-bar bg-warning';
                        } else {
                            fakeProbability.className = 'progress-bar bg-success';
                        }
                    }
                    
                    // Add data point to chart
                    dataPoints.push({
                        x: new Date(),
                        y: data.confidence_fake
                    });
                    
                    // Keep only last 30 seconds of data
                    const thirtySecondsAgo = new Date(Date.now() - 30000);
                    dataPoints = dataPoints.filter(point => point.x > thirtySecondsAgo);
                    
                    // Update chart
                    updateWebcamChart();
                }
                
                // Draw overlay if available
                if (data.overlay_frame && overlayCanvas) {
                    const ctx = overlayCanvas.getContext('2d');
                    const img = new Image();
                    img.onload = function() {
                        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
                        ctx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
                    };
                    img.src = 'data:image/jpeg;base64,' + data.overlay_frame;
                }
                
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };
    }
    
    function updateWebcamChart() {
        if (!webcamChart) return;
        
        webcamChart.data.datasets[0].data = dataPoints;
        
        // Update colors based on confidence
        webcamChart.data.datasets[0].backgroundColor = dataPoints.map(point => {
            return getColorForConfidence(point.y).replace(')', ', 0.2)');
        });
        
        webcamChart.data.datasets[0].borderColor = dataPoints.map(point => {
            return getColorForConfidence(point.y);
        });
        
        webcamChart.update();
    }
}

// Initialize image upload functionality
function initImageUpload() {
    console.log("Initializing image upload functionality");
    
    const imageUploadInput = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const imageAnalyzeBtn = document.getElementById('image-analyze-btn');
    const imageUploadProgress = document.getElementById('image-upload-progress');
    const imageUploadAlert = document.getElementById('image-upload-alert');
    
    // Handle image file selection
    if (imageUploadInput) {
        imageUploadInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                
                // Validate file is an image
                if (!file.type.startsWith('image/')) {
                    showImageMessage('Please select a valid image file', 'warning');
                    imageAnalyzeBtn.disabled = true;
                    return;
                }
                
                // Create object URL for preview
                const objectUrl = URL.createObjectURL(file);
                imagePreview.src = objectUrl;
                imagePreview.onload = function() {
                    URL.revokeObjectURL(objectUrl);
                };
                
                // Enable analyze button
                imageAnalyzeBtn.disabled = false;
            }
        });
    }
    
    // Handle analyze button click
    if (imageAnalyzeBtn) {
        imageAnalyzeBtn.addEventListener('click', function() {
            if (imageUploadInput && imageUploadInput.files.length > 0) {
                handleImageAnalysis();
            } else {
                showImageMessage('Please select an image file first', 'warning');
            }
        });
    }
    
    // Function to show messages in the image tab
    function showImageMessage(message, type = 'info') {
        if (imageUploadAlert) {
            imageUploadAlert.textContent = message;
            imageUploadAlert.className = `alert alert-${type} mt-3`;
            imageUploadAlert.classList.remove('d-none');
            
            // Auto-hide success and info messages after 5 seconds
            if (type === 'success' || type === 'info') {
                setTimeout(() => {
                    imageUploadAlert.classList.add('d-none');
                }, 5000);
            }
        }
    }
    
    // Handle image analysis
    async function handleImageAnalysis() {
        try {
            // Show progress
            imageUploadProgress.classList.remove('d-none');
            const progressBar = imageUploadProgress.querySelector('.progress-bar');
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            progressBar.textContent = 'Uploading image...';
            
            // Hide previous results
            document.getElementById('image-results-section').classList.add('d-none');
            
            // Create form data
            const formData = new FormData();
            formData.append('file', imageUploadInput.files[0]);
            formData.append('model', document.getElementById('image-model-select').value || 'mesonet');
            
            // Show upload progress animation
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progress > 90) {
                    clearInterval(progressInterval);
                }
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
                progressBar.textContent = `${progress}%`;
            }, 100);
            
            // Upload and process the image
            const response = await fetch('/process/photo', {
                method: 'POST',
                body: formData
            });
            
            // Clear progress interval
            clearInterval(progressInterval);
            
            // Complete progress bar
            progressBar.style.width = '100%';
            progressBar.setAttribute('aria-valuenow', 100);
            progressBar.textContent = '100%';
            
            // Process response
            const result = await response.json();
            
            if (result.confidence_fake !== undefined) {
                // Display results
                displayImageResults(result);
                showImageMessage('Image analysis complete!', 'success');
            } else {
                showImageMessage(`Error: ${result.detail || 'Analysis failed'}`, 'danger');
            }
            
            // Hide progress after a delay
            setTimeout(() => {
                imageUploadProgress.classList.add('d-none');
            }, 1000);
            
        } catch (error) {
            console.error('Error analyzing image:', error);
            showImageMessage(`Error: ${error.message}`, 'danger');
            imageUploadProgress.classList.add('d-none');
        }
    }
    
    // Display image analysis results
    function displayImageResults(result) {
        // Show results section
        const resultsSection = document.getElementById('image-results-section');
        resultsSection.classList.remove('d-none');
        
        // Set analyzed image (base64)
        if (result.processed_image) {
            document.getElementById('result-analyzed-image').src = 'data:image/jpeg;base64,' + result.processed_image;
        }
        
        // Update confidence bar and text
        const confidenceValue = result.confidence_fake || 0;
        const confidencePercent = Math.round(confidenceValue * 100);
        const confidenceBar = document.getElementById('image-confidence-bar');
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceBar.setAttribute('aria-valuenow', confidencePercent);
        confidenceBar.textContent = `${confidencePercent}%`;
        
        // Set confidence bar color based on value
        if (confidencePercent > 70) {
            confidenceBar.className = 'progress-bar bg-danger';
        } else if (confidencePercent > 30) {
            confidenceBar.className = 'progress-bar bg-warning';
        } else {
            confidenceBar.className = 'progress-bar bg-success';
        }
        
        // Update confidence text
        document.getElementById('image-confidence-text').textContent = `Deepfake confidence: ${confidencePercent}%`;
        
        // Update result summary
        const resultSummary = document.getElementById('image-result-summary');
        if (confidencePercent > 70) {
            resultSummary.className = 'alert alert-danger';
            resultSummary.textContent = 'This image is likely a deepfake.';
        } else if (confidencePercent > 30) {
            resultSummary.className = 'alert alert-warning';
            resultSummary.textContent = 'This image may contain manipulated elements.';
        } else {
            resultSummary.className = 'alert alert-success';
            resultSummary.textContent = 'This image appears to be authentic.';
        }
        
        // Update technical details
        document.getElementById('image-model-used').textContent = typeof result.model === 'object' ? result.model.name : result.model;
        document.getElementById('image-confidence-score').textContent = `${confidencePercent}% (${confidenceValue.toFixed(4)})`;
        document.getElementById('image-processing-time').textContent = result.processing_time ? `${result.processing_time.toFixed(2)} seconds` : 'N/A';
        document.getElementById('image-resolution').textContent = result.image_info ? `${result.image_info.width} × ${result.image_info.height}` : 'N/A';
    }
}

// Function to manually restart the analysis environment
async function initRestartButton() {
    const restartBtn = document.getElementById('restart-analysis-btn');
    if (restartBtn) {
        restartBtn.addEventListener('click', async function() {
            try {
                // Disable button during restart
                restartBtn.disabled = true;
                restartBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Restarting...';
                
                // Call restart endpoint
                const response = await fetch('/restart', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.message) {
                    showMessage('Analysis environment restarted successfully. Ready for new uploads.', 'success');
                } else {
                    showMessage(`Restart failed: ${result.detail || 'Unknown error'}`, 'danger');
                }
            } catch (error) {
                console.error('Error restarting analysis:', error);
                showMessage(`Error restarting: ${error.message}`, 'danger');
            } finally {
                // Re-enable button
                restartBtn.disabled = false;
                restartBtn.textContent = 'Restart Analysis Environment';
            }
        });
    }
}

// Function to handle page load/refresh and restore previous analysis
function checkForPreviousAnalysis() {
    // Check for file ID in URL query parameters
    const urlParams = new URLSearchParams(window.location.search);
    const fileIdFromUrl = urlParams.get('file_id');
    
    // Check for file ID in cookies
    const fileIdFromCookie = getCookie('last_analysis_file_id');
    
    // Check for file ID in localStorage
    const fileIdFromStorage = localStorage.getItem('last_analysis_file_id');
    
    // Use file ID from any available source, prioritizing URL
    const fileId = fileIdFromUrl || fileIdFromCookie || fileIdFromStorage;
    
    if (fileId) {
        console.log(`Found previous analysis file ID: ${fileId}`);
        
        // Store in localStorage for persistence
        localStorage.setItem('last_analysis_file_id', fileId);
        
        // Restore analysis results
        restoreAnalysis(fileId);
    }
}

// Function to restore analysis from a file ID
async function restoreAnalysis(fileId) {
    try {
        showMessage('Restoring previous analysis...', 'info');
        
        // Check if the analysis is complete
        const statusResponse = await fetch(`/status/${fileId}`);
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
            // Fetch and display results
            await requestAnalysis(fileId);
        } else if (statusData.status === 'processing') {
            // Start polling for status
            uploadStatusElement.classList.remove('d-none');
            uploadProgressBar.style.width = '50%';
            uploadProgressBar.setAttribute('aria-valuenow', 50);
            uploadProgressBar.textContent = 'Processing video...';
            
            // Set the global file ID
            uploadedFileId = fileId;
            
            // Start polling
            startProgressPolling(fileId);
        } else {
            // Analysis not found
            console.log('Previous analysis not found or expired');
            localStorage.removeItem('last_analysis_file_id');
        }
    } catch (error) {
        console.error('Error restoring analysis:', error);
    }
}

// Helper function to get cookie value
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

// Show details for a specific frame
function showFrameDetails(result) {
    // This function can be expanded to show more details about a specific frame
    console.log('Frame details:', result);
    
    // If there's an overlay frame available, you could display it
    if (result.overlay_frame) {
        // Display overlay frame if needed
    }
    
    // Show confidence in a toast or other UI element
    const confidencePercent = Math.round(result.confidence_fake * 100);
    showMessage(`Frame ${result.frame}: ${confidencePercent}% deepfake confidence`, 'info');
}



