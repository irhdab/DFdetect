// Video loading fix
document.addEventListener('DOMContentLoaded', function() {
    console.log("Video loading fix applied");
    
    // Get references to video elements
    const videoUploadInput = document.getElementById('video-upload');
    const uploadVideoPreview = document.getElementById('upload-video-preview');
    const uploadBtn = document.getElementById('upload-btn');
    
    // Add event listener for file selection
    if (videoUploadInput) {
        videoUploadInput.addEventListener('change', function(e) {
            console.log("File selected:", this.files);
            if (this.files && this.files[0]) {
                const file = this.files[0];
                
                // Create object URL for the video
                const videoURL = URL.createObjectURL(file);
                
                // Set the video source
                if (uploadVideoPreview) {
                    // Reset the video element
                    uploadVideoPreview.pause();
                    uploadVideoPreview.removeAttribute('src');
                    uploadVideoPreview.load();
                    
                    // Set new source
                    uploadVideoPreview.src = videoURL;
                    uploadVideoPreview.style.display = 'block';
                    
                    // Log when video loads
                    uploadVideoPreview.onloadeddata = function() {
                        console.log("Video loaded successfully:", uploadVideoPreview.duration);
                    };
                    
                    // Log video errors
                    uploadVideoPreview.onerror = function() {
                        console.error("Error loading video:", uploadVideoPreview.error);
                    };
                    
                    // Start playing the video
                    uploadVideoPreview.play().catch(e => {
                        console.warn("Auto-play prevented:", e);
                    });
                    
                    // Enable the analyze button
                    if (uploadBtn) {
                        uploadBtn.disabled = false;
                    }
                } else {
                    console.error("Video preview element not found");
                }
            }
        });
    } else {
        console.error("Video upload input element not found");
    }
}); 