import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_change_in_production")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///attendance_system.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

# Create upload folder for face images
UPLOAD_FOLDER = 'face_encodings'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with app.app_context():
    # Import models and routes
    import models
    import routes
    
    # Create all tables
    db.create_all()
    
    # Create default admin and sample students
    from models import Admin, Student
    from werkzeug.security import generate_password_hash
    
    # Create default admin
    if not Admin.query.filter_by(username='admin').first():
        admin = Admin(
            username='admin',
            email='admin@system.com',
            password_hash=generate_password_hash('admin123')
        )
        db.session.add(admin)
        logging.info("Default admin created - username: admin, password: admin123")
    
    # Create sample students for testing
    sample_students = [
        {'student_id': 'STU001', 'username': 'john_doe', 'email': 'john@example.com', 'full_name': 'John Doe', 'password': 'student123'},
        {'student_id': 'STU002', 'username': 'jane_smith', 'email': 'jane@example.com', 'full_name': 'Jane Smith', 'password': 'student123'},
        {'student_id': 'STU003', 'username': 'mike_wilson', 'email': 'mike@example.com', 'full_name': 'Mike Wilson', 'password': 'student123'},
        {'student_id': 'STU004', 'username': 'sarah_brown', 'email': 'sarah@example.com', 'full_name': 'Sarah Brown', 'password': 'student123'},
        {'student_id': 'STU005', 'username': 'alex_jones', 'email': 'alex@example.com', 'full_name': 'Alex Jones', 'password': 'student123'}
    ]
    
    for student_data in sample_students:
        if not Student.query.filter_by(username=student_data['username']).first():
            student = Student(
                student_id=student_data['student_id'],
                username=student_data['username'],
                email=student_data['email'],
                full_name=student_data['full_name']
            )
            student.set_password(student_data['password'])
            db.session.add(student)
            logging.info(f"Sample student created - username: {student_data['username']}, password: {student_data['password']}")
    
    db.session.commit()
