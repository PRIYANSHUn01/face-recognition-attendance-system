"""
Create Sample Attendance Data for Heatmap Visualization
Generates realistic attendance patterns for demonstration
"""

from app import app, db
from models import Student, Attendance
from datetime import datetime, date, timedelta
import random

def create_sample_attendance():
    """Create sample attendance data for heatmap visualization"""
    with app.app_context():
        # Get existing students
        students = Student.query.all()
        if not students:
            print("No students found. Please create students first.")
            return
        
        print(f"Creating sample attendance for {len(students)} students...")
        
        # Generate attendance for the last 4 weeks
        start_date = date.today() - timedelta(weeks=4)
        end_date = date.today()
        
        # Define peak hours with higher probability
        peak_hours = {
            8: 0.8,   # 8 AM - high attendance
            9: 0.9,   # 9 AM - peak
            10: 0.7,  # 10 AM - good
            11: 0.6,  # 11 AM - moderate
            12: 0.4,  # 12 PM - lunch break
            13: 0.5,  # 1 PM - returning from lunch
            14: 0.8,  # 2 PM - afternoon peak
            15: 0.7,  # 3 PM - good
            16: 0.6,  # 4 PM - moderate
            17: 0.4,  # 5 PM - leaving
            18: 0.3,  # 6 PM - low
            19: 0.2,  # 7 PM - very low
        }
        
        attendance_count = 0
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends for more realistic patterns
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                
                for student in students:
                    # Each student has 70% chance to attend on any given day
                    if random.random() < 0.7:
                        
                        # Select random hours for attendance (multiple entries per day)
                        num_entries = random.randint(1, 3)  # 1-3 entries per day
                        
                        for _ in range(num_entries):
                            # Choose hour based on peak hour probabilities
                            hour = random.choices(
                                list(peak_hours.keys()),
                                weights=list(peak_hours.values())
                            )[0]
                            
                            # Add some randomness to minutes
                            minute = random.randint(0, 59)
                            
                            # Create attendance time
                            attendance_time = datetime.combine(
                                current_date, 
                                datetime.min.time().replace(hour=hour, minute=minute)
                            )
                            
                            # Check if attendance already exists for this student on this date
                            existing = Attendance.query.filter_by(
                                student_id=student.id,
                                date=current_date
                            ).first()
                            
                            if not existing:
                                # Create new attendance record
                                attendance = Attendance(
                                    student_id=student.id,
                                    date=current_date,
                                    time_in=attendance_time,
                                    status='Present',
                                    recognition_confidence=random.uniform(0.75, 0.95)
                                )
                                db.session.add(attendance)
                                attendance_count += 1
            
            current_date += timedelta(days=1)
        
        # Commit all changes
        try:
            db.session.commit()
            print(f"Successfully created {attendance_count} attendance records!")
            print("Sample data includes:")
            print("- 4 weeks of attendance data")
            print("- Realistic peak hour patterns (8-9 AM, 2-3 PM)")
            print("- Weekend exclusions")
            print("- Multiple daily entries per student")
            print("- Varied attendance rates")
            
        except Exception as e:
            db.session.rollback()
            print(f"Error creating sample data: {e}")

if __name__ == "__main__":
    create_sample_attendance()