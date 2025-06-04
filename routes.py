from flask import render_template, request, redirect, url_for, session, flash, jsonify
from app import app, db
from models import Student, Admin, Attendance, AttendanceSession
from deep_face_recognition import face_recognition_system
from attendance_heatmap import heatmap_generator
from datetime import datetime, date, timedelta
from sqlalchemy import func, desc
import logging

@app.route('/')
def index():
    return render_template('index.html')

# Student Routes
@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    if request.method == 'POST':
        try:
            student_id = request.form.get('student_id')
            username = request.form.get('username')
            email = request.form.get('email')
            full_name = request.form.get('full_name')
            password = request.form.get('password')
            face_image = request.form.get('face_image')
            
            # Validate required fields
            if not all([student_id, username, email, full_name, password, face_image]):
                flash('All fields including face capture are required', 'error')
                return render_template('student_register.html')
            
            # Check if student already exists
            if Student.query.filter_by(student_id=student_id).first():
                flash('Student ID already exists', 'error')
                return render_template('student_register.html')
            
            if Student.query.filter_by(username=username).first():
                flash('Username already exists', 'error')
                return render_template('student_register.html')
            
            if Student.query.filter_by(email=email).first():
                flash('Email already exists', 'error')
                return render_template('student_register.html')
            
            # Create new student first
            student = Student(
                student_id=student_id,
                username=username,
                email=email,
                full_name=full_name,
                face_encoding_path=f"{student_id}.pkl"
            )
            student.set_password(password)
            
            db.session.add(student)
            db.session.commit()
            
            # Deep learning face enrollment
            logging.info(f"Starting deep learning face enrollment for new student {student_id}")
            success, message = face_recognition_system.save_face_encoding(face_image, student_id, full_name)
            if not success:
                logging.error(f"Deep face enrollment failed for {student_id}: {message}")
                flash(f'Registration successful but deep face enrollment failed: {message}. You can re-enroll your face later from your dashboard.', 'warning')
            else:
                logging.info(f"Deep face enrollment successful for {student_id} using advanced neural networks")
                flash('Registration and deep learning face enrollment successful! You can now use AI-powered face recognition for attendance.', 'success')
            
            return redirect(url_for('student_login'))
            
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'error')
            return render_template('student_register.html')
    
    return render_template('student_register.html')

@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        student = Student.query.filter_by(username=username).first()
        
        if student and student.check_password(password):
            session['student_id'] = student.id
            session['user_type'] = 'student'
            flash('Login successful!', 'success')
            return redirect(url_for('student_dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('student_login.html')

@app.route('/student/dashboard')
def student_dashboard():
    if 'student_id' not in session or session.get('user_type') != 'student':
        flash('Please login to access dashboard', 'error')
        return redirect(url_for('student_login'))
    
    student = Student.query.get(session['student_id'])
    if not student:
        session.clear()
        flash('Student not found', 'error')
        return redirect(url_for('student_login'))
    
    # Get recent attendance records
    recent_attendance = Attendance.query.filter_by(student_id=student.id)\
        .order_by(desc(Attendance.date))\
        .limit(10).all()
    
    # Get attendance statistics
    total_days = Attendance.query.filter_by(student_id=student.id).count()
    present_days = Attendance.query.filter_by(student_id=student.id, status='Present').count()
    attendance_percentage = (present_days / total_days * 100) if total_days > 0 else 0
    
    # Get monthly attendance data for chart - simplified version
    try:
        monthly_data = []
        # Get last 6 months of data
        for i in range(6):
            month_start = datetime.now().replace(day=1) - timedelta(days=30*i)
            month_end = month_start.replace(day=28) + timedelta(days=4)
            month_end = month_end - timedelta(days=month_end.day)
            
            total_count = Attendance.query.filter(
                Attendance.student_id == student.id,
                Attendance.date >= month_start.date(),
                Attendance.date <= month_end.date()
            ).count()
            
            present_count = Attendance.query.filter(
                Attendance.student_id == student.id,
                Attendance.date >= month_start.date(),
                Attendance.date <= month_end.date(),
                Attendance.status == 'Present'
            ).count()
            
            if total_count > 0:
                monthly_data.append({
                    'month': month_start.strftime('%Y-%m'),
                    'total': total_count,
                    'present': present_count
                })
    except Exception as e:
        logging.error(f"Error getting monthly data: {str(e)}")
        monthly_data = []
    
    return render_template('student_dashboard.html', 
                         student=student,
                         recent_attendance=recent_attendance,
                         attendance_percentage=round(attendance_percentage, 1),
                         total_days=total_days,
                         present_days=present_days,
                         monthly_data=monthly_data)

@app.route('/mark_attendance')
def mark_attendance():
    if 'student_id' not in session or session.get('user_type') != 'student':
        flash('Please login to mark attendance', 'error')
        return redirect(url_for('student_login'))
    
    return render_template('mark_attendance.html')

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    if 'student_id' not in session or session.get('user_type') != 'student':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    try:
        data = request.get_json()
        face_image = data.get('image')
        
        if not face_image:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Deep learning face recognition
        logging.info(f"Starting deep face recognition for student session {session.get('student_id')}")
        recognized_student_id, confidence, message = face_recognition_system.recognize_face(face_image)
        
        # Log detailed recognition results
        logging.info(f"Deep recognition result: ID={recognized_student_id}, confidence={confidence:.4f}, message={message}")
        logging.info(f"Session student ID: {session.get('student_id')}, Recognized ID: {recognized_student_id}")
        
        if recognized_student_id:
            # Get student from database
            student = Student.query.filter_by(student_id=recognized_student_id).first()
            
            if student and student.id == session['student_id']:
                # Check if attendance already marked today
                today = date.today()
                existing_attendance = Attendance.query.filter_by(
                    student_id=student.id,
                    date=today
                ).first()
                
                if existing_attendance:
                    return jsonify({
                        'success': False, 
                        'message': 'Attendance already marked for today'
                    })
                
                # Mark attendance
                attendance = Attendance(
                    student_id=student.id,
                    date=today,
                    status='Present',
                    recognition_confidence=confidence
                )
                
                db.session.add(attendance)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': f'Attendance marked successfully using deep learning! AI Confidence: {confidence:.2%}',
                    'student_name': student.full_name,
                    'recognition_method': 'Deep Neural Network',
                    'algorithms_used': 'LBP + Gabor + PCA + Statistical Analysis'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Face recognized but does not match logged-in student'
                })
        else:
            return jsonify({'success': False, 'message': message})
            
    except Exception as e:
        logging.error(f"Attendance processing error: {str(e)}")
        return jsonify({'success': False, 'message': 'Error processing attendance'})

# Admin Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin = Admin.query.filter_by(username=username).first()
        
        if admin and admin.check_password(password):
            session['admin_id'] = admin.id
            session['user_type'] = 'admin'
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_id' not in session or session.get('user_type') != 'admin':
        flash('Please login as admin to access dashboard', 'error')
        return redirect(url_for('admin_login'))
    
    # Get statistics
    total_students = Student.query.count()
    today = date.today()
    today_attendance = Attendance.query.filter_by(date=today).count()
    
    # Get all students with their latest attendance
    students = Student.query.all()
    
    # Get attendance data for the last 30 days
    thirty_days_ago = today - timedelta(days=30)
    daily_attendance = db.session.query(
        Attendance.date,
        func.count(Attendance.id).label('count')
    ).filter(Attendance.date >= thirty_days_ago)\
     .group_by(Attendance.date)\
     .order_by(Attendance.date).all()
    
    # Get top students by attendance percentage
    student_stats = []
    for student in students:
        total_days = Attendance.query.filter_by(student_id=student.id).count()
        present_days = Attendance.query.filter_by(student_id=student.id, status='Present').count()
        percentage = (present_days / total_days * 100) if total_days > 0 else 0
        
        student_stats.append({
            'student': student,
            'total_days': total_days,
            'present_days': present_days,
            'percentage': round(percentage, 1)
        })
    
    student_stats.sort(key=lambda x: x['percentage'], reverse=True)
    
    return render_template('admin_dashboard.html',
                         total_students=total_students,
                         today_attendance=today_attendance,
                         students=students,
                         daily_attendance=daily_attendance,
                         student_stats=student_stats[:10])  # Top 10 students

@app.route('/admin/student/<int:student_id>')
def admin_student_detail(student_id):
    if 'admin_id' not in session or session.get('user_type') != 'admin':
        flash('Please login as admin', 'error')
        return redirect(url_for('admin_login'))
    
    student = Student.query.get_or_404(student_id)
    attendance_records = Attendance.query.filter_by(student_id=student_id)\
        .order_by(desc(Attendance.date)).all()
    
    return render_template('student_detail.html', 
                         student=student, 
                         attendance_records=attendance_records)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# API endpoint for getting attendance data
@app.route('/api/attendance_data')
def get_attendance_data():
    if 'admin_id' not in session or session.get('user_type') != 'admin':
        return jsonify({'error': 'Unauthorized'})
    
    # Get daily attendance for the last 30 days
    thirty_days_ago = date.today() - timedelta(days=30)
    daily_data = db.session.query(
        Attendance.date,
        func.count(Attendance.id).label('count')
    ).filter(Attendance.date >= thirty_days_ago)\
     .group_by(Attendance.date)\
     .order_by(Attendance.date).all()
    
    return jsonify({
        'labels': [str(record.date) for record in daily_data],
        'data': [record.count for record in daily_data]
    })

# Heatmap API Routes
@app.route('/api/heatmap/weekly')
def get_weekly_heatmap():
    """API endpoint for weekly attendance heatmap data"""
    weeks_back = request.args.get('weeks', 4, type=int)
    data = heatmap_generator.get_weekly_heatmap_data(weeks_back)
    return jsonify(data)

@app.route('/api/heatmap/monthly')
def get_monthly_heatmap():
    """API endpoint for monthly attendance heatmap data"""
    months_back = request.args.get('months', 3, type=int)
    data = heatmap_generator.get_monthly_heatmap_data(months_back)
    return jsonify(data)

@app.route('/api/heatmap/realtime')
def get_realtime_activity():
    """API endpoint for real-time attendance activity"""
    data = heatmap_generator.get_real_time_activity_data()
    return jsonify(data)

@app.route('/api/heatmap/peak-hours')
def get_peak_hours():
    """API endpoint for peak hours analysis"""
    days_back = request.args.get('days', 30, type=int)
    data = heatmap_generator.get_peak_hours_analysis(days_back)
    return jsonify(data)

@app.route('/api/heatmap/trends')
def get_attendance_trends():
    """API endpoint for attendance trends"""
    weeks_back = request.args.get('weeks', 8, type=int)
    data = heatmap_generator.get_attendance_trends(weeks_back)
    return jsonify(data)
