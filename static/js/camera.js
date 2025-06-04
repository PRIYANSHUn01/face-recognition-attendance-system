class CameraHandler {
    constructor() {
        this.stream = null;
        this.video = null;
    }

    async startCamera(videoElementId, hideButtonId, showButtonId) {
        try {
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access not supported in this browser');
            }

            const constraints = {
                video: {
                    width: { ideal: 640, min: 320 },
                    height: { ideal: 480, min: 240 },
                    facingMode: 'user'
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video = document.getElementById(videoElementId);
            
            if (this.video) {
                this.video.srcObject = this.stream;
                this.video.classList.remove('d-none');
                
                // Hide placeholder and camera button
                const placeholder = document.getElementById('cameraPlaceholder');
                if (placeholder) {
                    placeholder.classList.add('d-none');
                }
                
                if (hideButtonId) {
                    const hideButton = document.getElementById(hideButtonId);
                    if (hideButton) {
                        hideButton.classList.add('d-none');
                    }
                }
                
                // Show action button
                if (showButtonId) {
                    const showButton = document.getElementById(showButtonId);
                    if (showButton) {
                        showButton.classList.remove('d-none');
                    }
                }

                return true;
            }
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.handleCameraError(error);
            return false;
        }
    }

    capturePhoto(videoElementId, canvasElementId) {
        try {
            const video = document.getElementById(videoElementId);
            const canvas = document.getElementById(canvasElementId);
            
            if (!video || !canvas) {
                console.error('Video or canvas element not found');
                return null;
            }

            const context = canvas.getContext('2d');
            
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw the video frame to canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64 data URL
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Show canvas and hide video temporarily
            canvas.classList.remove('d-none');
            video.classList.add('d-none');
            
            return imageData;
        } catch (error) {
            console.error('Error capturing photo:', error);
            alert('Failed to capture photo. Please try again.');
            return null;
        }
    }

    stopCamera(videoElementId) {
        try {
            if (this.stream) {
                const tracks = this.stream.getTracks();
                tracks.forEach(track => track.stop());
                this.stream = null;
            }

            const video = document.getElementById(videoElementId);
            if (video) {
                video.srcObject = null;
                video.classList.add('d-none');
            }

            const canvas = document.getElementById('canvas');
            if (canvas) {
                canvas.classList.add('d-none');
            }

            const placeholder = document.getElementById('cameraPlaceholder');
            if (placeholder) {
                placeholder.classList.remove('d-none');
            }

            return true;
        } catch (error) {
            console.error('Error stopping camera:', error);
            return false;
        }
    }

    handleCameraError(error) {
        let errorMessage = 'Camera access failed. ';
        
        switch (error.name) {
            case 'NotAllowedError':
                errorMessage += 'Please allow camera permissions and refresh the page.';
                break;
            case 'NotFoundError':
                errorMessage += 'No camera device found on this device.';
                break;
            case 'NotReadableError':
                errorMessage += 'Camera is already in use by another application.';
                break;
            case 'OverconstrainedError':
                errorMessage += 'Camera constraints could not be satisfied.';
                break;
            case 'SecurityError':
                errorMessage += 'Camera access blocked due to security restrictions.';
                break;
            default:
                errorMessage += 'An unknown error occurred: ' + error.message;
        }

        // Show error to user
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger mt-3';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Camera Error:</strong> ${errorMessage}
        `;

        // Insert error message
        const container = document.querySelector('.card-body');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
        }

        // Auto-remove error after 10 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 10000);
    }

    // Check if camera is supported
    static isCameraSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    // Get available camera devices
    static async getCameraDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('Error getting camera devices:', error);
            return [];
        }
    }
}

// Initialize camera support check on page load
document.addEventListener('DOMContentLoaded', function() {
    if (!CameraHandler.isCameraSupported()) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'alert alert-warning';
        warningDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Camera Not Supported:</strong> Your browser doesn't support camera access. 
            Please use a modern browser like Chrome, Firefox, or Safari.
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(warningDiv, container.firstChild);
        }
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CameraHandler;
}
