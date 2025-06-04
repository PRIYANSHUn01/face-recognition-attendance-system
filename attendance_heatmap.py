"""
Real-time Attendance Heatmap Visualization Module
Generates dynamic heatmaps for attendance patterns and real-time monitoring
"""

import json
import logging
from datetime import datetime, timedelta, time
from collections import defaultdict
from flask import jsonify
from models import Attendance, Student, db
from sqlalchemy import func, text
import calendar

logging.basicConfig(level=logging.INFO)

class AttendanceHeatmapGenerator:
    def __init__(self):
        """Initialize the attendance heatmap generator"""
        self.time_slots = self._generate_time_slots()
        self.weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
    def _generate_time_slots(self):
        """Generate time slots for heatmap (every hour from 6 AM to 10 PM)"""
        slots = []
        for hour in range(6, 23):  # 6 AM to 10 PM
            slots.append(f"{hour:02d}:00")
        return slots
    
    def get_weekly_heatmap_data(self, weeks_back=4):
        """
        Generate weekly attendance heatmap data
        Returns attendance count for each day/time slot combination
        """
        try:
            # Calculate date range for the last few weeks
            end_date = datetime.now().date()
            start_date = end_date - timedelta(weeks=weeks_back)
            
            # Query attendance data
            attendance_records = db.session.query(
                Attendance.date,
                func.extract('hour', Attendance.time_in).label('hour'),
                func.count(Attendance.id).label('count')
            ).filter(
                Attendance.date >= start_date,
                Attendance.date <= end_date,
                Attendance.status == 'Present'
            ).group_by(
                Attendance.date,
                func.extract('hour', Attendance.time_in)
            ).all()
            
            # Initialize heatmap data structure
            heatmap_data = []
            max_count = 0
            
            # Process data for each day and time slot
            for day_idx, day_name in enumerate(self.weekdays):
                for slot_idx, time_slot in enumerate(self.time_slots):
                    hour = int(time_slot.split(':')[0])
                    
                    # Count attendance for this day/hour combination
                    count = 0
                    for record in attendance_records:
                        record_hour = int(record.hour)
                        record_weekday = record.date.weekday()
                        
                        if record_weekday == day_idx and record_hour == hour:
                            count += record.count
                    
                    heatmap_data.append({
                        'day': day_name,
                        'time': time_slot,
                        'value': count,
                        'day_index': day_idx,
                        'time_index': slot_idx
                    })
                    
                    max_count = max(max_count, count)
            
            # Normalize values for better visualization
            for data_point in heatmap_data:
                if max_count > 0:
                    data_point['normalized'] = data_point['value'] / max_count
                else:
                    data_point['normalized'] = 0
            
            logging.info(f"Generated weekly heatmap data: {len(heatmap_data)} data points, max count: {max_count}")
            return {
                'data': heatmap_data,
                'max_count': max_count,
                'weeks_analyzed': weeks_back,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"Error generating weekly heatmap data: {e}")
            return {'data': [], 'max_count': 0, 'weeks_analyzed': 0}
    
    def get_monthly_heatmap_data(self, months_back=3):
        """
        Generate monthly attendance heatmap data
        Returns attendance patterns by day of month and hour
        """
        try:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date.replace(day=1) - timedelta(days=30 * months_back)
            
            # Query attendance data
            attendance_records = db.session.query(
                Attendance.time_in,
                Attendance.date,
                Student.full_name
            ).join(Student).filter(
                Attendance.date >= start_date,
                Attendance.date <= end_date,
                Attendance.status == 'Present'
            ).all()
            
            # Group data by day of month and hour
            daily_hourly_counts = defaultdict(lambda: defaultdict(int))
            
            for record in attendance_records:
                day_of_month = record.date.day
                hour = record.time_in.hour
                daily_hourly_counts[day_of_month][hour] += 1
            
            # Generate heatmap data
            heatmap_data = []
            max_count = 0
            
            for day in range(1, 32):  # Days 1-31
                for hour in range(6, 23):  # 6 AM to 10 PM
                    count = daily_hourly_counts[day][hour]
                    heatmap_data.append({
                        'day': day,
                        'hour': hour,
                        'count': count,
                        'time_label': f"{hour:02d}:00"
                    })
                    max_count = max(max_count, count)
            
            logging.info(f"Generated monthly heatmap data: {len(heatmap_data)} data points")
            return {
                'data': heatmap_data,
                'max_count': max_count,
                'months_analyzed': months_back
            }
            
        except Exception as e:
            logging.error(f"Error generating monthly heatmap data: {e}")
            return {'data': [], 'max_count': 0}
    
    def get_real_time_activity_data(self):
        """
        Get real-time attendance activity for today
        Shows current activity patterns and live updates
        """
        try:
            today = datetime.now().date()
            current_hour = datetime.now().hour
            
            # Get today's attendance records
            todays_attendance = db.session.query(
                Attendance.time_in,
                Student.full_name,
                Student.student_id
            ).join(Student).filter(
                Attendance.date == today,
                Attendance.status == 'Present'
            ).order_by(Attendance.time_in.desc()).all()
            
            # Group by hour
            hourly_activity = defaultdict(list)
            total_today = len(todays_attendance)
            
            for record in todays_attendance:
                hour = record.time_in.hour
                hourly_activity[hour].append({
                    'student_name': record.full_name,
                    'student_id': record.student_id,
                    'time': record.time_in.strftime('%H:%M:%S')
                })
            
            # Generate real-time activity data
            activity_data = []
            for hour in range(6, 23):
                count = len(hourly_activity[hour])
                activity_data.append({
                    'hour': hour,
                    'time_label': f"{hour:02d}:00",
                    'count': count,
                    'students': hourly_activity[hour],
                    'is_current_hour': hour == current_hour
                })
            
            # Get recent activity (last 30 minutes)
            recent_time = datetime.now() - timedelta(minutes=30)
            recent_attendance = db.session.query(
                Attendance.time_in,
                Student.full_name,
                Student.student_id
            ).join(Student).filter(
                Attendance.date == today,
                Attendance.time_in >= recent_time,
                Attendance.status == 'Present'
            ).order_by(Attendance.time_in.desc()).limit(10).all()
            
            recent_activity = [{
                'student_name': record.full_name,
                'student_id': record.student_id,
                'time': record.time_in.strftime('%H:%M:%S'),
                'minutes_ago': int((datetime.now() - record.time_in).total_seconds() / 60)
            } for record in recent_attendance]
            
            logging.info(f"Generated real-time activity data: {total_today} total attendees today")
            return {
                'hourly_data': activity_data,
                'recent_activity': recent_activity,
                'total_today': total_today,
                'current_hour': current_hour,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error generating real-time activity data: {e}")
            return {
                'hourly_data': [],
                'recent_activity': [],
                'total_today': 0,
                'current_hour': datetime.now().hour
            }
    
    def get_peak_hours_analysis(self, days_back=30):
        """
        Analyze peak attendance hours and patterns
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Query attendance data grouped by hour
            hourly_stats = db.session.query(
                func.extract('hour', Attendance.time_in).label('hour'),
                func.count(Attendance.id).label('total_count'),
                func.count(func.distinct(Attendance.date)).label('days_with_activity')
            ).filter(
                Attendance.date >= start_date,
                Attendance.date <= end_date,
                Attendance.status == 'Present'
            ).group_by(
                func.extract('hour', Attendance.time_in)
            ).all()
            
            # Calculate average attendance per hour
            peak_hours_data = []
            max_avg = 0
            
            for stat in hourly_stats:
                hour = int(stat.hour)
                avg_attendance = stat.total_count / max(stat.days_with_activity, 1)
                
                peak_hours_data.append({
                    'hour': hour,
                    'time_label': f"{hour:02d}:00",
                    'total_count': stat.total_count,
                    'avg_attendance': round(avg_attendance, 2),
                    'active_days': stat.days_with_activity
                })
                
                max_avg = max(max_avg, avg_attendance)
            
            # Sort by average attendance
            peak_hours_data.sort(key=lambda x: x['avg_attendance'], reverse=True)
            
            # Identify peak periods
            peak_periods = []
            if peak_hours_data:
                top_3_hours = peak_hours_data[:3]
                peak_periods = [{
                    'rank': idx + 1,
                    'time_range': f"{hour['time_label']} - {hour['hour']+1:02d}:00",
                    'avg_attendance': hour['avg_attendance'],
                    'description': self._get_period_description(hour['hour'])
                } for idx, hour in enumerate(top_3_hours)]
            
            logging.info(f"Generated peak hours analysis: {len(peak_hours_data)} hours analyzed")
            return {
                'hourly_data': peak_hours_data,
                'peak_periods': peak_periods,
                'max_avg_attendance': round(max_avg, 2),
                'analysis_period_days': days_back
            }
            
        except Exception as e:
            logging.error(f"Error generating peak hours analysis: {e}")
            return {'hourly_data': [], 'peak_periods': [], 'max_avg_attendance': 0}
    
    def _get_period_description(self, hour):
        """Get descriptive text for time periods"""
        if 6 <= hour < 9:
            return "Early Morning"
        elif 9 <= hour < 12:
            return "Mid Morning"
        elif 12 <= hour < 14:
            return "Lunch Time"
        elif 14 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 20:
            return "Evening"
        else:
            return "Late Hours"
    
    def get_attendance_trends(self, weeks_back=8):
        """
        Generate attendance trend data for visualization
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(weeks=weeks_back)
            
            # Query weekly attendance trends
            weekly_data = db.session.query(
                func.date_trunc('week', Attendance.date).label('week'),
                func.count(Attendance.id).label('total_attendance'),
                func.count(func.distinct(Attendance.student_id)).label('unique_students')
            ).filter(
                Attendance.date >= start_date,
                Attendance.date <= end_date,
                Attendance.status == 'Present'
            ).group_by(
                func.date_trunc('week', Attendance.date)
            ).order_by('week').all()
            
            trend_data = []
            for week_data in weekly_data:
                week_start = week_data.week.date()
                trend_data.append({
                    'week_start': week_start.isoformat(),
                    'week_label': week_start.strftime('%b %d'),
                    'total_attendance': week_data.total_attendance,
                    'unique_students': week_data.unique_students,
                    'avg_daily_attendance': round(week_data.total_attendance / 7, 1)
                })
            
            logging.info(f"Generated attendance trends: {len(trend_data)} weeks")
            return {
                'weekly_trends': trend_data,
                'weeks_analyzed': weeks_back
            }
            
        except Exception as e:
            logging.error(f"Error generating attendance trends: {e}")
            return {'weekly_trends': [], 'weeks_analyzed': 0}

# Global instance
heatmap_generator = AttendanceHeatmapGenerator()