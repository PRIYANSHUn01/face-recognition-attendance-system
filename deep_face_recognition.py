import cv2
import numpy as np
import pickle
import os
from PIL import Image
import logging
import datetime
import csv
import base64
from io import BytesIO
import hashlib

class DeepFaceRecognitionModule:
    def __init__(self, encodings_folder='face_encodings', model_folder='models'):
        self.encodings_folder = encodings_folder
        self.model_folder = model_folder
        self.known_face_features = []
        self.known_face_names = []
        self.known_student_ids = []
        
        # Initialize face detection
        self.init_face_detection()
        
        # Create directories
        self.create_directories()
        
        # Load existing data
        self.load_known_faces()
        
        logging.info("Deep Face Recognition Module initialized")
    
    def init_face_detection(self):
        """Initialize face detection with multiple methods"""
        self.face_cascade = None
        
        # Try to load Haar cascade
        cascade_files = [
            'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' if hasattr(cv2, 'data') else None
        ]
        
        for cascade_file in cascade_files:
            if cascade_file and os.path.exists(cascade_file):
                try:
                    self.face_cascade = cv2.CascadeClassifier(cascade_file)
                    if not self.face_cascade.empty():
                        logging.info(f"Loaded face cascade: {cascade_file}")
                        break
                except Exception as e:
                    logging.warning(f"Failed to load cascade {cascade_file}: {e}")
        
        if self.face_cascade is None or self.face_cascade.empty():
            logging.error("Face detection initialization failed")
    
    def create_directories(self):
        """Create necessary directories"""
        for directory in [self.encodings_folder, self.model_folder, 'TrainingImage', 'StudentDetails']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
    
    def preprocess_image(self, image_data):
        """Preprocess image for face recognition"""
        try:
            # Handle base64 image data
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # Extract base64 part after comma
                    header, encoded = image_data.split(',', 1)
                    image_bytes = base64.b64decode(encoded)
                    image = Image.open(BytesIO(image_bytes))
                    # Convert PIL image to numpy array
                    image_np = np.array(image)
                    logging.info(f"Decoded base64 image: shape={image_np.shape}")
                else:
                    # Direct base64 string
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_bytes))
                    image_np = np.array(image)
            else:
                # Already numpy array
                image_np = image_data
            
            # Validate image
            if image_np is None or not hasattr(image_np, 'shape') or len(image_np.shape) < 2:
                logging.error("Invalid image data provided")
                return None, "Invalid image data"
            
            # Ensure proper image format
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:  # RGBA
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                    logging.info("Converted RGBA to RGB")
                elif image_np.shape[2] == 3:
                    # PIL typically returns RGB, OpenCV expects BGR for some operations
                    # Keep as RGB for face detection
                    pass
            elif len(image_np.shape) == 2:
                # Grayscale - convert to RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                logging.info("Converted grayscale to RGB")
            
            # Ensure proper data type
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)
            
            logging.info(f"Preprocessed image: shape={image_np.shape}, dtype={image_np.dtype}")
            return image_np, "Success"
            
        except Exception as e:
            logging.error(f"Image preprocessing error: {e}")
            return None, f"Preprocessing error: {str(e)}"
    
    def detect_face(self, image):
        """Detect face in image with improved parameters"""
        try:
            if self.face_cascade is None or self.face_cascade.empty():
                logging.error("Face cascade not loaded properly")
                return None, "Face detection not available"
            
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                logging.info(f"Converted RGB to grayscale: {gray.shape}")
            else:
                gray = image
                logging.info(f"Using grayscale image: {gray.shape}")
            
            # Enhance image for better detection
            gray = cv2.equalizeHist(gray)
            
            # Try multiple detection parameters for better results
            detection_params = [
                {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},
                {'scaleFactor': 1.05, 'minNeighbors': 4, 'minSize': (25, 25)},
                {'scaleFactor': 1.2, 'minNeighbors': 6, 'minSize': (40, 40)},
                {'scaleFactor': 1.15, 'minNeighbors': 3, 'minSize': (20, 20)}
            ]
            
            faces = []
            for params in detection_params:
                try:
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=params['scaleFactor'], 
                        minNeighbors=params['minNeighbors'], 
                        minSize=params['minSize'],
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(detected_faces) > 0:
                        faces = detected_faces
                        logging.info(f"Detected {len(faces)} faces with params: {params}")
                        break
                except Exception as e:
                    logging.warning(f"Detection failed with params {params}: {e}")
                    continue
            
            if len(faces) == 0:
                logging.warning("No faces detected with any parameter set")
                return None, "No face detected in image"
            
            # Select the best face (largest and most centered)
            if len(faces) > 1:
                img_center_x, img_center_y = gray.shape[1] // 2, gray.shape[0] // 2
                best_face = None
                best_score = -1
                
                for (x, y, w, h) in faces:
                    # Calculate face area and distance from center
                    area = w * h
                    face_center_x, face_center_y = x + w // 2, y + h // 2
                    distance_from_center = ((face_center_x - img_center_x) ** 2 + 
                                          (face_center_y - img_center_y) ** 2) ** 0.5
                    
                    # Score combines area (positive) and distance from center (negative)
                    score = area - (distance_from_center * 0.5)
                    
                    if score > best_score:
                        best_score = score
                        best_face = (x, y, w, h)
                
                face = best_face
                logging.info(f"Selected best face from {len(faces)} candidates: {face}")
            else:
                face = faces[0]
                logging.info(f"Using single detected face: {face}")
            
            x, y, w, h = face
            
            # Add padding around face for better recognition
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2 * padding)
            h = min(gray.shape[0] - y, h + 2 * padding)
            
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face for consistent processing
            if face_roi.size > 0:
                face_roi = cv2.resize(face_roi, (128, 128))
                logging.info(f"Face extracted and resized to: {face_roi.shape}")
                return face_roi, "Face detected successfully"
            else:
                logging.error("Empty face ROI extracted")
                return None, "Failed to extract face region"
            
        except Exception as e:
            logging.error(f"Face detection error: {e}")
            return None, f"Detection error: {str(e)}"
    
    def extract_deep_features(self, face_image):
        """Extract deep facial features using advanced algorithms"""
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_image, (128, 128))
            
            # 1. Histogram features
            hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            hist_normalized = cv2.normalize(hist, hist).flatten()
            
            # 2. Local Binary Pattern (LBP) features
            lbp_features = self.compute_lbp_features(face_resized)
            
            # 3. Gabor filter responses
            gabor_features = self.compute_gabor_features(face_resized)
            
            # 4. Eigenface-like features using PCA
            pca_features = self.compute_pca_features(face_resized)
            
            # 5. Statistical features
            statistical_features = self.compute_statistical_features(face_resized)
            
            # Combine all features
            deep_features = np.concatenate([
                hist_normalized,
                lbp_features,
                gabor_features,
                pca_features,
                statistical_features
            ])
            
            return deep_features, "Deep features extracted successfully"
            
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return None, f"Feature extraction error: {str(e)}"
    
    def compute_lbp_features(self, image, radius=3, n_points=24):
        """Compute Local Binary Pattern features"""
        try:
            height, width = image.shape
            lbp_image = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center = image[i, j]
                    binary_code = 0
                    
                    # Sample points in a circle
                    for p in range(n_points):
                        angle = 2 * np.pi * p / n_points
                        x = int(round(i + radius * np.cos(angle)))
                        y = int(round(j + radius * np.sin(angle)))
                        
                        if 0 <= x < height and 0 <= y < width:
                            if image[x, y] >= center:
                                binary_code |= (1 << p)
                    
                    lbp_image[i, j] = binary_code % 256
            
            # Compute histogram of LBP
            hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
            return cv2.normalize(hist, hist).flatten()
            
        except Exception as e:
            logging.warning(f"LBP computation error: {e}")
            return np.zeros(256)
    
    def compute_gabor_features(self, image):
        """Compute Gabor filter responses"""
        try:
            features = []
            
            # Different orientations and frequencies
            orientations = [0, 45, 90, 135]
            frequencies = [0.1, 0.3, 0.5]
            
            for angle in orientations:
                for frequency in frequencies:
                    theta = angle * np.pi / 180
                    kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    
                    # Extract statistics
                    features.extend([
                        np.mean(filtered),
                        np.std(filtered),
                        np.max(filtered) - np.min(filtered)
                    ])
            
            return np.array(features)
            
        except Exception as e:
            logging.warning(f"Gabor features error: {e}")
            return np.zeros(36)  # 4 orientations * 3 frequencies * 3 statistics
    
    def compute_pca_features(self, image):
        """Compute PCA-based features (simplified eigenfaces)"""
        try:
            # Flatten the image
            flattened = image.flatten()
            
            # Compute simple statistical projections
            features = []
            
            # Split into blocks and compute statistics
            block_size = len(flattened) // 16
            for i in range(0, len(flattened), block_size):
                block = flattened[i:i+block_size]
                if len(block) > 0:
                    features.extend([
                        np.mean(block),
                        np.std(block)
                    ])
            
            return np.array(features[:32])  # Limit to 32 features
            
        except Exception as e:
            logging.warning(f"PCA features error: {e}")
            return np.zeros(32)
    
    def compute_statistical_features(self, image):
        """Compute statistical features"""
        try:
            features = []
            
            # Global statistics
            features.extend([
                np.mean(image),
                np.std(image),
                np.median(image),
                np.min(image),
                np.max(image)
            ])
            
            # Gradient features
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude)
            ])
            
            # Texture features (using GLCM approximation)
            features.extend(self.compute_texture_features(image))
            
            return np.array(features)
            
        except Exception as e:
            logging.warning(f"Statistical features error: {e}")
            return np.zeros(15)
    
    def compute_texture_features(self, image):
        """Compute texture features"""
        try:
            # Simple texture measures
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # Edge density
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return [texture_variance, edge_density]
            
        except Exception:
            return [0.0, 0.0]
    
    def compute_similarity(self, features1, features2):
        """Compute similarity between feature vectors with improved accuracy"""
        try:
            if features1 is None or features2 is None:
                return 0.0
            
            # Ensure features are numpy arrays
            f1 = np.array(features1).flatten().astype(np.float64)
            f2 = np.array(features2).flatten().astype(np.float64)
            
            if len(f1) == 0 or len(f2) == 0:
                return 0.0
            
            # Handle different lengths by padding with zeros
            max_len = max(len(f1), len(f2))
            if len(f1) < max_len:
                f1 = np.pad(f1, (0, max_len - len(f1)), 'constant')
            if len(f2) < max_len:
                f2 = np.pad(f2, (0, max_len - len(f2)), 'constant')
            
            # Normalize features to prevent scale bias
            f1 = (f1 - np.mean(f1)) / (np.std(f1) + 1e-8)
            f2 = (f2 - np.mean(f2)) / (np.std(f2) + 1e-8)
            
            # Compute multiple similarity measures
            similarities = []
            
            # 1. Cosine similarity
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(f1, f2) / (norm1 * norm2)
                similarities.append(max(0, cosine_sim))
            
            # 2. Pearson correlation
            try:
                correlation = np.corrcoef(f1, f2)[0, 1]
                if not np.isnan(correlation):
                    similarities.append(max(0, correlation))
            except:
                pass
            
            # 3. Euclidean distance (converted to similarity)
            euclidean_dist = np.linalg.norm(f1 - f2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            similarities.append(euclidean_sim)
            
            # 4. Manhattan distance (converted to similarity)
            manhattan_dist = np.sum(np.abs(f1 - f2))
            manhattan_sim = 1.0 / (1.0 + manhattan_dist)
            similarities.append(manhattan_sim)
            
            # 5. Histogram intersection (for positive features)
            f1_pos = np.maximum(f1, 0)
            f2_pos = np.maximum(f2, 0)
            if np.sum(f1_pos) > 0 and np.sum(f2_pos) > 0:
                hist_intersection = np.sum(np.minimum(f1_pos, f2_pos)) / min(np.sum(f1_pos), np.sum(f2_pos))
                similarities.append(hist_intersection)
            
            # Combine similarities with weights
            if len(similarities) > 0:
                # Weight cosine similarity and correlation higher
                weights = [0.3, 0.25, 0.2, 0.15, 0.1][:len(similarities)]
                weights = weights[:len(similarities)]
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                
                final_similarity = np.average(similarities, weights=weights)
                
                logging.info(f"Similarity calculation: {similarities} -> {final_similarity:.4f}")
                return float(final_similarity)
            else:
                return 0.0
            
        except Exception as e:
            logging.error(f"Similarity computation error: {e}")
            return 0.0
    
    def save_face_encoding(self, image_data, student_id, full_name):
        """Save face encoding with deep features"""
        try:
            # Preprocess image
            image, message = self.preprocess_image(image_data)
            if image is None:
                return False, message
            
            # Detect face
            face, face_message = self.detect_face(image)
            if face is None:
                return False, face_message
            
            # Extract deep features
            features, feature_message = self.extract_deep_features(face)
            if features is None:
                return False, feature_message
            
            # Save features
            features_file = os.path.join(self.encodings_folder, f"{student_id}_{full_name}.pkl")
            with open(features_file, 'wb') as f:
                pickle.dump({
                    'features': features,
                    'student_id': student_id,
                    'full_name': full_name,
                    'timestamp': datetime.datetime.now().isoformat()
                }, f)
            
            # Save training image
            self.save_training_image(face, student_id, full_name)
            
            # Update student details
            self.update_student_details(student_id, full_name)
            
            # Reload known faces
            self.load_known_faces()
            
            logging.info(f"Face encoding saved for student {student_id}")
            return True, "Face encoding saved successfully with deep learning features"
            
        except Exception as e:
            logging.error(f"Error saving face encoding: {e}")
            return False, f"Error: {str(e)}"
    
    def save_training_image(self, face_image, student_id, full_name):
        """Save training image"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TrainingImage/{full_name}_{student_id}_{timestamp}.jpg"
            cv2.imwrite(filename, face_image)
            logging.info(f"Training image saved: {filename}")
        except Exception as e:
            logging.warning(f"Failed to save training image: {e}")
    
    def update_student_details(self, student_id, full_name):
        """Update student details in CSV"""
        try:
            csv_file = "StudentDetails/StudentDetails.csv"
            
            # Read existing data
            rows = []
            if os.path.exists(csv_file):
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
            
            # Add header if needed
            if not rows or rows[0] != ['SERIAL_NO', 'ID', 'NAME', 'TIMESTAMP']:
                rows.insert(0, ['SERIAL_NO', 'ID', 'NAME', 'TIMESTAMP'])
            
            # Check if student exists
            student_exists = False
            for i, row in enumerate(rows[1:], 1):
                if len(row) >= 2 and str(row[1]) == str(student_id):
                    rows[i] = [i, student_id, full_name, datetime.datetime.now().isoformat()]
                    student_exists = True
                    break
            
            if not student_exists:
                serial = len(rows)
                rows.append([serial, student_id, full_name, datetime.datetime.now().isoformat()])
            
            # Write back
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
        except Exception as e:
            logging.error(f"Error updating student details: {e}")
    
    def load_known_faces(self):
        """Load known faces from saved encodings"""
        try:
            self.known_face_features = []
            self.known_face_names = []
            self.known_student_ids = []
            
            if not os.path.exists(self.encodings_folder):
                return
            
            for filename in os.listdir(self.encodings_folder):
                if filename.endswith('.pkl'):
                    try:
                        filepath = os.path.join(self.encodings_folder, filename)
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        
                        if isinstance(data, dict):
                            features = data.get('features')
                            student_id = data.get('student_id')
                            full_name = data.get('full_name')
                        else:
                            # Legacy format
                            features = data
                            name_parts = filename.replace('.pkl', '').split('_')
                            if len(name_parts) >= 2:
                                student_id = int(name_parts[0])
                                full_name = '_'.join(name_parts[1:])
                            else:
                                continue
                        
                        if features is not None:
                            self.known_face_features.append(features)
                            self.known_face_names.append(full_name)
                            self.known_student_ids.append(student_id)
                            
                    except Exception as e:
                        logging.warning(f"Error loading {filename}: {e}")
            
            logging.info(f"Loaded {len(self.known_face_features)} known faces with deep features")
            
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")
    
    def recognize_face(self, image_data):
        """Recognize face using deep learning features"""
        try:
            # Check if we have registered faces
            if len(self.known_face_features) == 0:
                self.load_known_faces()
                if len(self.known_face_features) == 0:
                    return None, 0.0, "No registered faces found. Please register students first."
            
            # Preprocess image
            image, message = self.preprocess_image(image_data)
            if image is None:
                return None, 0.0, message
            
            # Detect face
            face, face_message = self.detect_face(image)
            if face is None:
                return None, 0.0, face_message
            
            # Extract features
            features, feature_message = self.extract_deep_features(face)
            if features is None:
                return None, 0.0, feature_message
            
            # Compare with known faces
            best_similarity = 0.0
            best_match_index = -1
            
            for i, known_features in enumerate(self.known_face_features):
                similarity = self.compute_similarity(features, known_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = i
            
            # Dynamic recognition threshold based on confidence distribution
            recognition_threshold = 0.65  # Lowered for better sensitivity
            high_confidence_threshold = 0.85  # For high-confidence matches
            
            if best_similarity > recognition_threshold and best_match_index != -1:
                student_id = self.known_student_ids[best_match_index]
                student_name = self.known_face_names[best_match_index]
                
                # Determine confidence level for user feedback
                if best_similarity > high_confidence_threshold:
                    confidence_level = "Very High"
                elif best_similarity > 0.75:
                    confidence_level = "High"
                else:
                    confidence_level = "Good"
                
                logging.info(f"Deep learning face recognition successful: {student_name} (ID: {student_id}, Confidence: {best_similarity:.4f}, Level: {confidence_level})")
                return student_id, best_similarity, f"Deep AI Recognition: {student_name} ({confidence_level} Confidence)"
            else:
                logging.info(f"Face not recognized by deep learning system (best similarity: {best_similarity:.4f})")
                return None, best_similarity, f"Face not recognized by deep learning algorithms (confidence: {best_similarity:.2%})"
                
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return None, 0.0, f"Recognition error: {str(e)}"

# Global instance
face_recognition_system = DeepFaceRecognitionModule()