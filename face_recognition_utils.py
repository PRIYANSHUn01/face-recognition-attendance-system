import cv2
import numpy as np
import pickle
import os
from PIL import Image
import logging
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

class DeepFaceRecognitionSystem:
    def __init__(self, encodings_folder='face_encodings'):
        self.encodings_folder = encodings_folder
        self.known_face_features = []
        self.known_face_names = []
        self.known_student_ids = []
        
        # Initialize OpenCV face detector with robust error handling
        self.face_cascade = None
        cascade_paths = [
            'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            try:
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    logging.info(f"Successfully loaded face cascade from: {path}")
                    break
            except Exception as e:
                logging.warning(f"Failed to load cascade from {path}: {e}")
                continue
        
        if self.face_cascade is None or self.face_cascade.empty():
            logging.error("Could not initialize face detection - creating minimal cascade")
            self.face_cascade = cv2.CascadeClassifier()
        
        # Create encodings folder
        if not os.path.exists(self.encodings_folder):
            os.makedirs(self.encodings_folder)
            
        self.load_known_faces()
    
    def extract_face_features(self, image):
        """Extract simple face features using OpenCV"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            if len(faces) == 0:
                return None, "No face detected in the image"
            
            if len(faces) > 1:
                return None, "Multiple faces detected. Please ensure only one face is visible"
            
            # Get the largest face
            (x, y, w, h) = faces[0]
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (128, 128))
            
            # Calculate histogram features
            hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            
            # Normalize histogram
            hist = cv2.normalize(hist, hist).flatten()
            
            # Calculate additional features
            # LBP-like features (simplified)
            mean_val = np.mean(face_resized)
            std_val = np.std(face_resized)
            
            # Combine features
            features = np.concatenate([hist, [mean_val, std_val]])
            
            return features, "Face features extracted successfully"
            
        except Exception as e:
            return None, f"Error extracting features: {str(e)}"
    
    def save_face_encoding(self, image_data, student_id, full_name):
        """Save face features for a student"""
        try:
            # Convert base64 image to numpy array
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
                import base64
                image_bytes = base64.b64decode(image_data)
                
                # Convert to PIL Image
                from io import BytesIO
                image = Image.open(BytesIO(image_bytes))
                image_np = np.array(image)
            else:
                image_np = image_data
            
            # Convert BGR to RGB if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # Assume it's BGR from OpenCV, convert to RGB
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_np
            
            # Extract face features
            features, message = self.extract_face_features(image_rgb)
            if features is None:
                return False, message
            
            # Save features to file
            encoding_file = os.path.join(self.encodings_folder, f"{student_id}.pkl")
            encoding_data = {
                'features': features,
                'student_id': student_id,
                'name': full_name
            }
            
            with open(encoding_file, 'wb') as f:
                pickle.dump(encoding_data, f)
            
            # Reload known faces
            self.load_known_faces()
            
            return True, "Face registered successfully"
            
        except Exception as e:
            logging.error(f"Error saving face encoding: {str(e)}")
            return False, f"Error processing image: {str(e)}"
    
    def load_known_faces(self):
        """Load all known face features"""
        self.known_face_features = []
        self.known_face_names = []
        self.known_student_ids = []
        
        if not os.path.exists(self.encodings_folder):
            os.makedirs(self.encodings_folder)
            return
        
        for filename in os.listdir(self.encodings_folder):
            if filename.endswith('.pkl'):
                try:
                    with open(os.path.join(self.encodings_folder, filename), 'rb') as f:
                        data = pickle.load(f)
                        self.known_face_features.append(data['features'])
                        self.known_face_names.append(data['name'])
                        self.known_student_ids.append(data['student_id'])
                except Exception as e:
                    logging.error(f"Error loading {filename}: {str(e)}")
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        try:
            # Use cosine similarity
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def recognize_face(self, image_data):
        """Recognize face from image data"""
        try:
            # Convert base64 image to numpy array
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
                import base64
                image_bytes = base64.b64decode(image_data)
                
                from io import BytesIO
                image = Image.open(BytesIO(image_bytes))
                image_np = np.array(image)
            else:
                image_np = image_data
            
            # Convert to RGB
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_np
            
            # Extract features from input image
            input_features, message = self.extract_face_features(image_rgb)
            if input_features is None:
                return None, 0.0, message
            
            if len(self.known_face_features) == 0:
                # Try to reload known faces before giving up
                self.load_known_faces()
                if len(self.known_face_features) == 0:
                    return None, 0.0, "No registered faces found. Please register students with face enrollment first."
            
            # Compare with known faces
            best_similarity = 0.0
            best_match_index = -1
            
            for i, known_features in enumerate(self.known_face_features):
                similarity = self.calculate_similarity(input_features, known_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = i
            
            # Set threshold for recognition (you can adjust this)
            recognition_threshold = 0.7
            
            if best_similarity > recognition_threshold and best_match_index != -1:
                student_id = self.known_student_ids[best_match_index]
                name = self.known_face_names[best_match_index]
                confidence = best_similarity
                
                return student_id, confidence, f"Recognized: {name}"
            else:
                return None, best_similarity, f"Face not recognized (similarity: {best_similarity:.2f})"
            
        except Exception as e:
            logging.error(f"Error in face recognition: {str(e)}")
            return None, 0.0, f"Error: {str(e)}"

# Global instance
face_recognition_system = DeepFaceRecognitionSystem()
