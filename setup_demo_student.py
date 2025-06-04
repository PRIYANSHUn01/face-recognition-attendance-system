#!/usr/bin/env python3
"""
Setup Demo Student Script
Creates a demo student in the database with face enrollment for testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import Student, Admin
from deep_face_recognition import face_recognition_system
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageDraw

def create_demo_face_image():
    """Create a realistic demo face image"""
    # Create a more realistic face pattern in RGB
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
    
    # Face oval (skin color)
    cv2.ellipse(img, (100, 100), (80, 100), 0, 0, 360, (220, 180, 150), -1)
    
    # Eyes (white)
    cv2.ellipse(img, (75, 80), (12, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (125, 80), (12, 8), 0, 0, 360, (255, 255, 255), -1)
    # Pupils (black)
    cv2.circle(img, (75, 80), 4, (0, 0, 0), -1)
    cv2.circle(img, (125, 80), 4, (0, 0, 0), -1)
    
    # Nose
    cv2.line(img, (100, 90), (100, 110), (180, 140, 120), 2)
    cv2.ellipse(img, (100, 110), (8, 4), 0, 0, 180, (180, 140, 120), 1)
    
    # Mouth (red)
    cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, (180, 120, 120), 2)
    
    # Add some subtle texture for better feature extraction
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def image_to_base64(image_array):
    """Convert numpy image to base64"""
    # Convert to PIL Image
    if len(image_array.shape) == 2:
        # Grayscale
        pil_img = Image.fromarray(image_array, mode='L')
    else:
        # RGB
        pil_img = Image.fromarray(image_array, mode='RGB')
    
    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def setup_demo_student():
    """Setup demo student with face enrollment"""
    with app.app_context():
        try:
            # Check if demo student already exists
            existing_student = Student.query.filter_by(student_id='DEMO001').first()
            if existing_student:
                print("Demo student already exists, updating face encoding...")
                student = existing_student
            else:
                # Create demo student
                student = Student(
                    student_id='DEMO001',
                    username='demo_student',
                    email='demo@example.com',
                    full_name='Demo Student',
                    is_active=True
                )
                student.set_password('demo123')
                db.session.add(student)
                db.session.commit()
                print("Created demo student: DEMO001")
            
            # Create demo face image
            print("Creating demo face image...")
            face_image = create_demo_face_image()
            
            # Convert to base64
            face_base64 = image_to_base64(face_image)
            
            # Enroll face
            print("Enrolling demo face...")
            success, message = face_recognition_system.save_face_encoding(
                face_base64, 
                student_id='DEMO001', 
                full_name='Demo Student'
            )
            
            if success:
                print(f"Success: {message}")
                print(f"Known faces count: {len(face_recognition_system.known_face_features)}")
                
                # Test recognition
                print("Testing face recognition...")
                recognized_id, confidence, test_message = face_recognition_system.recognize_face(face_base64)
                print(f"Recognition test - ID: {recognized_id}, Confidence: {confidence:.2f}, Message: {test_message}")
                
                return True
            else:
                print(f"Failed to enroll face: {message}")
                return False
                
        except Exception as e:
            print(f"Error setting up demo student: {e}")
            import traceback
            traceback.print_exc()
            return False

def create_admin_if_needed():
    """Create default admin if not exists"""
    with app.app_context():
        try:
            admin = Admin.query.filter_by(username='admin').first()
            if not admin:
                admin = Admin(
                    username='admin',
                    email='admin@example.com'
                )
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                print("Created default admin: admin/admin123")
            else:
                print("Admin already exists")
        except Exception as e:
            print(f"Error creating admin: {e}")

if __name__ == "__main__":
    print("Setting up demo environment...")
    create_admin_if_needed()
    success = setup_demo_student()
    if success:
        print("\nDemo setup completed successfully!")
        print("You can now test attendance with the demo student:")
        print("- Student ID: DEMO001")
        print("- Username: demo_student") 
        print("- Password: demo123")
        print("\nAdmin credentials:")
        print("- Username: admin")
        print("- Password: admin123")
    else:
        print("Demo setup failed!")