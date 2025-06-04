import cv2
import numpy as np
import pickle
import os
from PIL import Image
import logging
import hashlib
import datetime
import csv

class AdvancedFaceRecognitionSystem:
    def __init__(self, training_path='TrainingImage', model_path='TrainingImageLabel'):
        self.training_path = training_path
        self.model_path = model_path
        self.known_face_features = []
        self.known_face_names = []
        self.known_student_ids = []
        
        # Initialize LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Initialize OpenCV face detector with robust error handling
        self.face_cascade = None
        cascade_paths = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_advanced.xml',
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
            logging.error("Could not initialize face detection")
            self.face_cascade = cv2.CascadeClassifier()
        
        # Create necessary directories
        self.create_directories()
        
        # Load existing model if available
        self.load_trained_model()
        
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
            
            # Convert to grayscale
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) != 1:
                return False, f"Expected 1 face, found {len(faces)}"
            
            # Extract face region
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Save multiple training images for better accuracy
            serial = self.get_next_serial(student_id)
            for i in range(5):  # Save 5 variations
                filename = f"{self.training_path}/{full_name}.{serial}.{student_id}.{i+1}.jpg"
                cv2.imwrite(filename, face_roi)
            
            logging.info(f"Saved training images for student {student_id}")
            return True, "Training images saved successfully"
            
        except Exception as e:
            logging.error(f"Error saving training images: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def get_next_serial(self, student_id):
        """Get the next serial number for a student"""
        try:
            csv_path = "StudentDetails/StudentDetails.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if not df.empty:
                    return len(df) + 1
            return 1
        except Exception:
            return 1
    
    def train_model(self):
        """Train the LBPH face recognition model"""
        try:
            faces, ids = self.get_images_and_labels()
            
            if len(faces) == 0:
                return False, "No training images found"
            
            # Train the recognizer
            self.recognizer.train(faces, np.array(ids))
            
            # Save the trained model
            model_file = f"{self.model_path}/Trainner.yml"
            self.recognizer.save(model_file)
            
            logging.info(f"Model trained successfully with {len(faces)} images")
            return True, f"Model trained with {len(faces)} images"
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False, f"Training error: {str(e)}"
    
    def get_images_and_labels(self):
        """Get training images and their labels"""
        faces = []
        ids = []
        
        if not os.path.exists(self.training_path):
            return faces, ids
        
        image_paths = [os.path.join(self.training_path, f) 
                      for f in os.listdir(self.training_path) 
                      if f.endswith('.jpg')]
        
        for image_path in image_paths:
            try:
                # Load image and convert to grayscale
                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')
                
                # Extract ID from filename
                filename = os.path.split(image_path)[-1]
                parts = filename.split('.')
                if len(parts) >= 3:
                    student_id = int(parts[2])
                    faces.append(image_np)
                    ids.append(student_id)
                    
            except Exception as e:
                logging.warning(f"Error processing {image_path}: {e}")
                continue
        
        return faces, ids
    
    def load_trained_model(self):
        """Load the trained LBPH model"""
        try:
            model_file = f"{self.model_path}/Trainner.yml"
            if os.path.exists(model_file):
                self.recognizer.read(model_file)
                logging.info("Loaded trained model successfully")
                return True
            else:
                logging.warning("No trained model found")
                return False
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    def recognize_face(self, image_data):
        """Recognize face using the trained LBPH model"""
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
            if not hasattr(image_np, 'shape') or len(image_np.shape) < 2:
                return None, 0.0, "Invalid image format"
            
            # Convert to grayscale
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            
            if len(faces) == 0:
                return None, 0.0, "No face detected"
            
            if len(faces) > 1:
                return None, 0.0, "Multiple faces detected"
            
            # Extract face region
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict using LBPH recognizer
            student_id, confidence = self.recognizer.predict(face_roi)
            
            # Convert confidence to similarity (lower confidence = higher similarity)
            # LBPH confidence: lower is better, 0 is perfect match
            if confidence < 50:  # Threshold for recognition
                similarity = max(0.0, (100 - confidence) / 100.0)
                
                # Get student details
                student_name = self.get_student_name(student_id)
                if student_name:
                    return student_id, similarity, "Face recognized successfully"
                else:
                    return None, 0.0, "Student not found in database"
            else:
                return None, 0.0, "Face not recognized with sufficient confidence"
                
        except Exception as e:
            logging.error(f"Error in face recognition: {str(e)}")
            return None, 0.0, f"Recognition error: {str(e)}"
    
    def get_student_name(self, student_id):
        """Get student name from CSV file"""
        try:
            csv_path = "StudentDetails/StudentDetails.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                student_row = df[df['ID'] == student_id]
                if not student_row.empty:
                    return student_row['NAME'].iloc[0]
            return None
        except Exception as e:
            logging.error(f"Error getting student name: {e}")
            return None
    
    def save_face_encoding(self, image_data, student_id, full_name):
        """Save face encoding for a student"""
        try:
            # Save training images
            success, message = self.save_training_image(image_data, student_id, full_name)
            if not success:
                return False, message
            
            # Update student details CSV
            self.update_student_details(student_id, full_name)
            
            # Retrain the model with new data
            train_success, train_message = self.train_model()
            if train_success:
                return True, "Face encoding saved and model updated successfully"
            else:
                return True, f"Face saved but model training had issues: {train_message}"
                
        except Exception as e:
            logging.error(f"Error saving face encoding: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def update_student_details(self, student_id, full_name):
        """Update student details in CSV file"""
        try:
            csv_path = "StudentDetails/StudentDetails.csv"
            
            # Create CSV if it doesn't exist
            if not os.path.exists(csv_path):
                columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
                df = pd.DataFrame(columns=columns)
                df.to_csv(csv_path, index=False)
            
            # Read existing CSV
            df = pd.read_csv(csv_path)
            
            # Check if student already exists
            if not df[df['ID'] == student_id].empty:
                logging.info(f"Student {student_id} already exists, updating...")
                return
            
            # Add new student
            serial = len(df) + 1
            new_row = pd.DataFrame({
                'SERIAL NO.': [serial],
                '': [''],
                'ID': [student_id],
                '': [''],
                'NAME': [full_name]
            })
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
            
            logging.info(f"Added student {student_id} to details")
            
        except Exception as e:
            logging.error(f"Error updating student details: {e}")
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between face features"""
        try:
            return cosine_similarity([features1], [features2])[0][0]
        except Exception:
            return 0.0
    
    def load_known_faces(self):
        """Load known faces from the trained model"""
        try:
            # This method is for compatibility with the existing interface
            # The LBPH model handles face recognition differently
            model_file = f"{self.model_path}/Trainner.yml"
            if os.path.exists(model_file):
                self.recognizer.read(model_file)
                
                # Load student details
                csv_path = "StudentDetails/StudentDetails.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    self.known_student_ids = df['ID'].tolist() if 'ID' in df.columns else []
                    self.known_face_names = df['NAME'].tolist() if 'NAME' in df.columns else []
                
                logging.info(f"Loaded {len(self.known_student_ids)} known faces")
            
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")

# Global instance
face_recognition_system = AdvancedFaceRecognitionSystem()