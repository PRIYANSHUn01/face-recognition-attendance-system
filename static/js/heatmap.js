/**
 * Real-time Attendance Heatmap Visualization
 * Provides dynamic heatmap charts and real-time activity monitoring
 */

class AttendanceHeatmap {
    constructor() {
        this.weeklyChart = null;
        this.monthlyChart = null;
        this.realtimeChart = null;
        this.peakHoursChart = null;
        this.trendsChart = null;
        this.updateInterval = null;
        this.colorScales = {
            intensity: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            activity: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
        };
    }

    /**
     * Initialize all heatmap visualizations
     */
    init() {
        this.loadWeeklyHeatmap();
        this.loadMonthlyHeatmap();
        this.loadRealtimeActivity();
        this.loadPeakHours();
        this.loadTrends();
        this.startRealtimeUpdates();
    }

    /**
     * Load and display weekly attendance heatmap
     */
    async loadWeeklyHeatmap() {
        try {
            const response = await fetch('/api/heatmap/weekly');
            const data = await response.json();
            
            if (data.data && data.data.length > 0) {
                this.renderWeeklyHeatmap(data);
                this.updateHeatmapStats('weekly', data);
            }
        } catch (error) {
            console.error('Error loading weekly heatmap:', error);
            this.showError('weekly-heatmap', 'Failed to load weekly attendance data');
        }
    }

    /**
     * Render weekly heatmap using Chart.js with matrix display
     */
    renderWeeklyHeatmap(data) {
        const ctx = document.getElementById('weeklyHeatmapChart');
        if (!ctx) return;

        // Prepare data for heatmap visualization
        const heatmapData = this.prepareHeatmapData(data.data, 'weekly');
        
        // Destroy existing chart
        if (this.weeklyChart) {
            this.weeklyChart.destroy();
        }

        this.weeklyChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Attendance Activity',
                    data: heatmapData.points,
                    backgroundColor: (context) => {
                        const value = context.parsed.v || 0;
                        return this.getHeatmapColor(value, data.max_count);
                    },
                    pointRadius: 15,
                    pointHoverRadius: 18
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Weekly Attendance Heatmap (${data.weeks_analyzed} weeks)`,
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (context) => {
                                const point = context[0];
                                const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
                                const timeSlot = heatmapData.timeSlots[point.parsed.x] || '';
                                return `${dayNames[point.parsed.y]} at ${timeSlot}`;
                            },
                            label: (context) => {
                                return `Attendance: ${context.parsed.v || 0} students`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: heatmapData.timeSlots.length - 1,
                        ticks: {
                            stepSize: 1,
                            callback: (value) => heatmapData.timeSlots[value] || ''
                        },
                        title: {
                            display: true,
                            text: 'Time of Day'
                        }
                    },
                    y: {
                        type: 'linear',
                        min: 0,
                        max: 6,
                        ticks: {
                            stepSize: 1,
                            callback: (value) => {
                                const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
                                return days[value] || '';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Day of Week'
                        }
                    }
                }
            }
        });
    }

    /**
     * Load and display real-time activity data
     */
    async loadRealtimeActivity() {
        try {
            const response = await fetch('/api/heatmap/realtime');
            const data = await response.json();
            
            this.renderRealtimeActivity(data);
            this.updateRealtimeStats(data);
        } catch (error) {
            console.error('Error loading real-time activity:', error);
            this.showError('realtime-activity', 'Failed to load real-time data');
        }
    }

    /**
     * Render real-time activity chart
     */
    renderRealtimeActivity(data) {
        const ctx = document.getElementById('realtimeActivityChart');
        if (!ctx) return;

        // Destroy existing chart
        if (this.realtimeChart) {
            this.realtimeChart.destroy();
        }

        const hourlyData = data.hourly_data || [];
        const labels = hourlyData.map(item => item.time_label);
        const counts = hourlyData.map(item => item.count);
        const currentHourIndex = hourlyData.findIndex(item => item.is_current_hour);

        this.realtimeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Students Present',
                    data: counts,
                    backgroundColor: (context) => {
                        return context.dataIndex === currentHourIndex ? 
                            'rgba(255, 99, 132, 0.8)' : 'rgba(54, 162, 235, 0.6)';
                    },
                    borderColor: (context) => {
                        return context.dataIndex === currentHourIndex ? 
                            'rgba(255, 99, 132, 1)' : 'rgba(54, 162, 235, 1)';
                    },
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Today's Attendance Activity (Total: ${data.total_today})`,
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            afterLabel: (context) => {
                                const hourData = hourlyData[context.dataIndex];
                                if (hourData && hourData.students && hourData.students.length > 0) {
                                    const studentNames = hourData.students.slice(0, 5).map(s => s.student_name);
                                    const remaining = hourData.students.length - 5;
                                    let tooltip = studentNames.join(', ');
                                    if (remaining > 0) {
                                        tooltip += ` and ${remaining} more`;
                                    }
                                    return tooltip;
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Students'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time of Day'
                        }
                    }
                }
            }
        });
    }

    /**
     * Load and display peak hours analysis
     */
    async loadPeakHours() {
        try {
            const response = await fetch('/api/heatmap/peak-hours');
            const data = await response.json();
            
            this.renderPeakHours(data);
            this.updatePeakHoursInfo(data);
        } catch (error) {
            console.error('Error loading peak hours:', error);
            this.showError('peak-hours', 'Failed to load peak hours data');
        }
    }

    /**
     * Render peak hours analysis chart
     */
    renderPeakHours(data) {
        const ctx = document.getElementById('peakHoursChart');
        if (!ctx) return;

        // Destroy existing chart
        if (this.peakHoursChart) {
            this.peakHoursChart.destroy();
        }

        const hourlyData = data.hourly_data || [];
        const labels = hourlyData.map(item => item.time_label);
        const avgAttendance = hourlyData.map(item => item.avg_attendance);

        this.peakHoursChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Attendance',
                    data: avgAttendance,
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Peak Hours Analysis (${data.analysis_period_days} days)`,
                        font: { size: 16, weight: 'bold' }
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: (context) => {
                                const hourData = hourlyData[context.dataIndex];
                                return [
                                    `Total attendances: ${hourData.total_count}`,
                                    `Active days: ${hourData.active_days}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Average Students'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time of Day'
                        }
                    }
                }
            }
        });
    }

    /**
     * Load and display attendance trends
     */
    async loadTrends() {
        try {
            const response = await fetch('/api/heatmap/trends');
            const data = await response.json();
            
            this.renderTrends(data);
        } catch (error) {
            console.error('Error loading trends:', error);
            this.showError('trends', 'Failed to load trends data');
        }
    }

    /**
     * Render attendance trends chart
     */
    renderTrends(data) {
        const ctx = document.getElementById('trendsChart');
        if (!ctx) return;

        // Destroy existing chart
        if (this.trendsChart) {
            this.trendsChart.destroy();
        }

        const weeklyData = data.weekly_trends || [];
        const labels = weeklyData.map(item => item.week_label);
        const totalAttendance = weeklyData.map(item => item.total_attendance);
        const uniqueStudents = weeklyData.map(item => item.unique_students);

        this.trendsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Total Attendance',
                    data: totalAttendance,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    yAxisID: 'y'
                }, {
                    label: 'Unique Students',
                    data: uniqueStudents,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Attendance Trends (${data.weeks_analyzed} weeks)`,
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Week'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Total Attendance'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Unique Students'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    /**
     * Start real-time updates
     */
    startRealtimeUpdates() {
        // Update every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadRealtimeActivity();
        }, 30000);
    }

    /**
     * Stop real-time updates
     */
    stopRealtimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Prepare data for heatmap visualization
     */
    prepareHeatmapData(data, type) {
        const points = [];
        const timeSlots = [];
        
        if (type === 'weekly') {
            // Generate time slots (6 AM to 10 PM)
            for (let hour = 6; hour <= 22; hour++) {
                timeSlots.push(`${hour.toString().padStart(2, '0')}:00`);
            }
            
            data.forEach(item => {
                const timeIndex = timeSlots.indexOf(item.time);
                if (timeIndex !== -1) {
                    points.push({
                        x: timeIndex,
                        y: item.day_index,
                        v: item.value
                    });
                }
            });
        }
        
        return { points, timeSlots };
    }

    /**
     * Get color for heatmap based on value intensity
     */
    getHeatmapColor(value, maxValue) {
        if (maxValue === 0) return this.colorScales.intensity[0];
        
        const intensity = value / maxValue;
        const colorIndex = Math.min(Math.floor(intensity * this.colorScales.intensity.length), 
                                   this.colorScales.intensity.length - 1);
        return this.colorScales.intensity[colorIndex];
    }

    /**
     * Update heatmap statistics display
     */
    updateHeatmapStats(type, data) {
        const statsContainer = document.getElementById(`${type}-stats`);
        if (statsContainer && data) {
            statsContainer.innerHTML = `
                <div class="row">
                    <div class="col-md-4">
                        <div class="stat-card">
                            <h6>Max Attendance</h6>
                            <span class="stat-value">${data.max_count}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card">
                            <h6>Data Points</h6>
                            <span class="stat-value">${data.data.length}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-card">
                            <h6>Analysis Period</h6>
                            <span class="stat-value">${data.weeks_analyzed || data.months_analyzed} ${type === 'weekly' ? 'weeks' : 'months'}</span>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    /**
     * Update real-time statistics
     */
    updateRealtimeStats(data) {
        const statsContainer = document.getElementById('realtime-stats');
        if (statsContainer && data) {
            const recentActivity = data.recent_activity || [];
            
            statsContainer.innerHTML = `
                <div class="row mb-3">
                    <div class="col-md-6">
                        <div class="stat-card">
                            <h6>Total Today</h6>
                            <span class="stat-value text-primary">${data.total_today}</span>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="stat-card">
                            <h6>Current Hour</h6>
                            <span class="stat-value text-success">${data.current_hour}:00</span>
                        </div>
                    </div>
                </div>
                <div class="recent-activity">
                    <h6>Recent Activity (Last 30 minutes)</h6>
                    <div class="activity-list">
                        ${recentActivity.slice(0, 5).map(activity => `
                            <div class="activity-item">
                                <strong>${activity.student_name}</strong> 
                                <small class="text-muted">(${activity.student_id})</small>
                                <span class="float-end">${activity.minutes_ago}m ago</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    }

    /**
     * Update peak hours information
     */
    updatePeakHoursInfo(data) {
        const infoContainer = document.getElementById('peak-hours-info');
        if (infoContainer && data.peak_periods) {
            infoContainer.innerHTML = `
                <h6>Top Peak Periods</h6>
                ${data.peak_periods.map(period => `
                    <div class="peak-period-item">
                        <span class="badge bg-primary">#${period.rank}</span>
                        <strong>${period.time_range}</strong>
                        <small class="text-muted">${period.description}</small>
                        <span class="float-end">${period.avg_attendance} avg</span>
                    </div>
                `).join('')}
            `;
        }
    }

    /**
     * Show error message
     */
    showError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${message}
                </div>
            `;
        }
    }

    /**
     * Refresh all heatmap data
     */
    refreshAll() {
        this.init();
    }

    /**
     * Export heatmap data as JSON
     */
    async exportData() {
        try {
            const [weekly, monthly, realtime, peakHours, trends] = await Promise.all([
                fetch('/api/heatmap/weekly').then(r => r.json()),
                fetch('/api/heatmap/monthly').then(r => r.json()),
                fetch('/api/heatmap/realtime').then(r => r.json()),
                fetch('/api/heatmap/peak-hours').then(r => r.json()),
                fetch('/api/heatmap/trends').then(r => r.json())
            ]);

            const exportData = {
                timestamp: new Date().toISOString(),
                weekly_heatmap: weekly,
                monthly_heatmap: monthly,
                realtime_activity: realtime,
                peak_hours: peakHours,
                trends: trends
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `attendance-heatmap-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error exporting data:', error);
        }
    }
}

// Initialize heatmap when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (typeof Chart !== 'undefined') {
        window.attendanceHeatmap = new AttendanceHeatmap();
        // Only initialize if we're on a page with heatmap elements
        if (document.getElementById('weeklyHeatmapChart') || 
            document.getElementById('realtimeActivityChart')) {
            window.attendanceHeatmap.init();
        }
    }
});