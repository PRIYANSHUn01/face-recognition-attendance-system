import cv2
import numpy as np
import pickle
import os
from PIL import Image
import logging
import datetime
import csv

class LBPHFaceRecognitionSystem:
    def __init__(self, training_path='TrainingImage', model_path='TrainingImageLabel'):
        self.training_path = training_path
        self.model_path = model_path
        self.known_face_features = []
        self.known_face_names = []
        self.known_student_ids = []
        
        # Initialize face cascade
        self.face_cascade = None
        cascade_paths = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_advanced.xml'
        ]
        
        for path in cascade_paths:
            try:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    if not self.face_cascade.empty():
                        logging.info(f"Successfully loaded face cascade from: {path}")
                        break
            except Exception as e:
                logging.warning(f"Failed to load cascade from {path}: {e}")
                continue
        
        if self.face_cascade is None or self.face_cascade.empty():
            logging.error("Could not initialize face detection")
            return
        
        # Create necessary directories
        self.create_directories()
        
        # Load existing data
        self.load_known_faces()
        
    def create_directories(self):
        """Create necessary directories for the face recognition system"""
        directories = [self.training_path, self.model_path, 'StudentDetails', 'face_encodings']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
    
    def save_training_image(self, image_data, student_id, full_name):
        """Save training images for a student"""
        try:
            # Convert base64 image to numpy array
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                import base64
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                
                from io import BytesIO
                image = Image.open(BytesIO(image_bytes))
                image_np = np.array(image)
            else:
                image_np = image_data
            
            # Validate image
            if image_np is None:
                return False, "Invalid image data"
            if not hasattr(image_np, 'shape') or len(image_np.shape) < 2:
                return False, "Invalid image format"
            
            # Convert to grayscale
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            # Detect faces
            if self.face_cascade is None or self.face_cascade.empty():
                return False, "Face detection not initialized"
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) != 1:
                return False, f"Expected 1 face, found {len(faces)}"
            
            # Extract face region
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face to standard size
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Save multiple training images for better accuracy
            serial = self.get_next_serial(student_id)
            saved_count = 0
            
            for i in range(10):  # Save 10 variations with slight modifications
                if i == 0:
                    # Original face
                    face_to_save = face_resized
                else:
                    # Create variations with noise and transformations
                    noise = np.random.normal(0, 5, face_resized.shape).astype(np.uint8)
                    face_to_save = cv2.add(face_resized, noise)
                    face_to_save = np.clip(face_to_save, 0, 255)
                
                filename = f"{self.training_path}/{full_name}.{serial}.{student_id}.{i+1}.jpg"
                if cv2.imwrite(filename, face_to_save):
                    saved_count += 1
            
            logging.info(f"Saved {saved_count} training images for student {student_id}")
            return True, f"Training images saved successfully ({saved_count} images)"
            
        except Exception as e:
            logging.error(f"Error saving training images: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def get_next_serial(self, student_id):
        """Get the next serial number for a student"""
        try:
            csv_path = "StudentDetails/StudentDetails.csv"
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                    return len(rows)
            return 1
        except Exception:
            return 1
    
    def extract_face_features(self, image):
        """Extract face features using histogram and statistical analysis"""
        try:
            # Convert to proper numpy array if needed
            if isinstance(image, str):
                import base64
                image_data = base64.b64decode(image.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if not hasattr(image, 'shape') or len(image.shape) < 2:
                return None, "Invalid image data"
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            
            if len(faces) != 1:
                return None, f"Expected 1 face, found {len(faces)}"
            
            # Extract face region
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_roi, (200, 200))
            
            # Calculate histogram features
            hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            hist_normalized = cv2.normalize(hist, hist).flatten()
            
            # Calculate LBP (Local Binary Pattern) features
            lbp_features = self.calculate_lbp(face_resized)
            
            # Statistical features
            mean_val = np.mean(face_resized)
            std_val = np.std(face_resized)
            skew_val = self.calculate_skewness(face_resized)
            
            # Combine all features
            features = np.concatenate([
                hist_normalized,
                lbp_features,
                [mean_val, std_val, skew_val]
            ])
            
            return features, "Face features extracted successfully"
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return None, f"Error extracting features: {str(e)}"
    
    def calculate_lbp(self, image):
        """Calculate Local Binary Pattern features"""
        try:
            height, width = image.shape
            lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = image[i, j]
                    binary_string = ''
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp_image[i-1, j-1] = int(binary_string, 2)
            
            # Calculate histogram of LBP image
            lbp_hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
            lbp_hist_normalized = cv2.normalize(lbp_hist, lbp_hist).flatten()
            
            return lbp_hist_normalized
            
        except Exception as e:
            logging.error(f"Error calculating LBP: {e}")
            return np.zeros(256)
    
    def calculate_skewness(self, image):
        """Calculate skewness of image intensities"""
        try:
            flattened = image.flatten()
            mean_val = np.mean(flattened)
            std_val = np.std(flattened)
            if std_val == 0:
                return 0
            skew = np.mean(((flattened - mean_val) / std_val) ** 3)
            return skew
        except Exception:
            return 0
    
    def calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        try:
            if features1 is None or features2 is None:
                return 0.0
            
            # Normalize vectors
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return max(0.0, similarity)
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def save_face_encoding(self, image_data, student_id, full_name):
        """Save face encoding for a student"""
        try:
            # Extract and save features
            features, message = self.extract_face_features(image_data)
            if features is None:
                return False, message
            
            # Save training images
            success, train_message = self.save_training_image(image_data, student_id, full_name)
            if not success:
                return False, train_message
            
            # Save features to file
            features_file = f"face_encodings/{student_id}_{full_name}.pkl"
            with open(features_file, 'wb') as f:
                pickle.dump(features, f)
            
            # Update student details
            self.update_student_details(student_id, full_name)
            
            # Reload known faces
            self.load_known_faces()
            
            return True, "Face encoding saved successfully"
            
        except Exception as e:
            logging.error(f"Error saving face encoding: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def update_student_details(self, student_id, full_name):
        """Update student details in CSV file"""
        try:
            csv_path = "StudentDetails/StudentDetails.csv"
            
            # Read existing data
            rows = []
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
            
            # Add header if file is empty
            if not rows:
                rows.append(['SERIAL NO.', '', 'ID', '', 'NAME'])
            
            # Check if student already exists
            student_exists = False
            for row in rows[1:]:  # Skip header
                if len(row) >= 5 and str(row[2]) == str(student_id):
                    student_exists = True
                    break
            
            if not student_exists:
                serial = len(rows)
                new_row = [serial, '', student_id, '', full_name]
                rows.append(new_row)
                
                # Write back to file
                with open(csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)
                
                logging.info(f"Added student {student_id} to details")
            
        except Exception as e:
            logging.error(f"Error updating student details: {e}")
    
    def load_known_faces(self):
        """Load known faces from saved encodings"""
        try:
            self.known_face_features = []
            self.known_face_names = []
            self.known_student_ids = []
            
            encodings_dir = "face_encodings"
            if not os.path.exists(encodings_dir):
                return
            
            for filename in os.listdir(encodings_dir):
                if filename.endswith('.pkl'):
                    try:
                        # Extract student info from filename
                        name_parts = filename.replace('.pkl', '').split('_')
                        if len(name_parts) >= 2:
                            student_id = int(name_parts[0])
                            student_name = '_'.join(name_parts[1:])
                            
                            # Load features
                            with open(os.path.join(encodings_dir, filename), 'rb') as f:
                                features = pickle.load(f)
                            
                            self.known_face_features.append(features)
                            self.known_face_names.append(student_name)
                            self.known_student_ids.append(student_id)
                            
                    except Exception as e:
                        logging.warning(f"Error loading {filename}: {e}")
                        continue
            
            logging.info(f"Loaded {len(self.known_face_features)} known faces")
            
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")
    
    def recognize_face(self, image_data):
        """Recognize face from image data"""
        try:
            # Check if we have any registered faces
            if len(self.known_face_features) == 0:
                self.load_known_faces()
                if len(self.known_face_features) == 0:
                    return None, 0.0, "No registered faces found. Please register students with face enrollment first."
            
            # Extract features from input image
            input_features, message = self.extract_face_features(image_data)
            if input_features is None:
                return None, 0.0, message
            
            # Compare with known faces
            best_similarity = 0.0
            best_match_index = -1
            
            for i, known_features in enumerate(self.known_face_features):
                similarity = self.calculate_similarity(input_features, known_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = i
            
            # Set threshold for recognition
            recognition_threshold = 0.75
            
            if best_similarity > recognition_threshold and best_match_index != -1:
                student_id = self.known_student_ids[best_match_index]
                student_name = self.known_face_names[best_match_index]
                
                return student_id, best_similarity, f"Recognized: {student_name}"
            else:
                return None, best_similarity, f"Face not recognized (similarity: {best_similarity:.2f})"
                
        except Exception as e:
            logging.error(f"Error in face recognition: {str(e)}")
            return None, 0.0, f"Error: {str(e)}"

# Global instance
face_recognition_system = LBPHFaceRecognitionSystem()