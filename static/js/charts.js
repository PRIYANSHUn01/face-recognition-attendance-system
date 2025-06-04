// Chart.js Configuration and Utilities for Attendance System

class AttendanceCharts {
    constructor() {
        this.defaultColors = {
            primary: '#0d6efd',
            success: '#198754',
            info: '#0dcaf0',
            warning: '#ffc107',
            danger: '#dc3545',
            secondary: '#6c757d'
        };
        
        this.chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#dee2e6',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        color: '#6c757d'
                    }
                },
                y: {
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        color: '#6c757d',
                        beginAtZero: true
                    }
                }
            }
        };
    }

    // Create line chart for daily attendance
    createDailyAttendanceChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element with id '${canvasId}' not found`);
            return null;
        }

        const config = {
            type: 'line',
            data: {
                labels: data.labels || [],
                datasets: [{
                    label: 'Daily Attendance',
                    data: data.values || [],
                    borderColor: this.defaultColors.primary,
                    backgroundColor: this.defaultColors.primary + '20',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: this.defaultColors.primary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Daily Attendance Trend',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    ...this.chartDefaults.scales,
                    y: {
                        ...this.chartDefaults.scales.y,
                        ticks: {
                            ...this.chartDefaults.scales.y.ticks,
                            stepSize: 1
                        }
                    }
                }
            }
        };

        return new Chart(ctx, config);
    }

    // Create monthly attendance chart for students
    createMonthlyAttendanceChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element with id '${canvasId}' not found`);
            return null;
        }

        const config = {
            type: 'line',
            data: {
                labels: data.labels || [],
                datasets: [
                    {
                        label: 'Days Present',
                        data: data.present || [],
                        borderColor: this.defaultColors.success,
                        backgroundColor: this.defaultColors.success + '20',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Total Days',
                        data: data.total || [],
                        borderColor: this.defaultColors.info,
                        backgroundColor: this.defaultColors.info + '20',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Monthly Attendance Overview',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                }
            }
        };

        return new Chart(ctx, config);
    }

    // Create attendance percentage pie chart
    createAttendancePercentageChart(canvasId, presentDays, totalDays) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element with id '${canvasId}' not found`);
            return null;
        }

        const absentDays = totalDays - presentDays;
        const presentPercentage = totalDays > 0 ? (presentDays / totalDays * 100).toFixed(1) : 0;
        const absentPercentage = totalDays > 0 ? (absentDays / totalDays * 100).toFixed(1) : 0;

        const config = {
            type: 'doughnut',
            data: {
                labels: ['Present', 'Absent'],
                datasets: [{
                    data: [presentDays, absentDays],
                    backgroundColor: [
                        this.defaultColors.success,
                        this.defaultColors.danger
                    ],
                    borderColor: [
                        this.defaultColors.success,
                        this.defaultColors.danger
                    ],
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const percentage = totalDays > 0 ? ((value / totalDays) * 100).toFixed(1) : 0;
                                return `${label}: ${value} days (${percentage}%)`;
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Attendance Distribution',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                cutout: '60%'
            }
        };

        return new Chart(ctx, config);
    }

    // Create bar chart for top students
    createTopStudentsChart(canvasId, studentsData) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.error(`Canvas element with id '${canvasId}' not found`);
            return null;
        }

        const config = {
            type: 'bar',
            data: {
                labels: studentsData.names || [],
                datasets: [{
                    label: 'Attendance Percentage',
                    data: studentsData.percentages || [],
                    backgroundColor: studentsData.percentages.map(percentage => {
                        if (percentage >= 90) return this.defaultColors.success + '80';
                        if (percentage >= 75) return this.defaultColors.info + '80';
                        if (percentage >= 60) return this.defaultColors.warning + '80';
                        return this.defaultColors.danger + '80';
                    }),
                    borderColor: studentsData.percentages.map(percentage => {
                        if (percentage >= 90) return this.defaultColors.success;
                        if (percentage >= 75) return this.defaultColors.info;
                        if (percentage >= 60) return this.defaultColors.warning;
                        return this.defaultColors.danger;
                    }),
                    borderWidth: 2,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Top Students by Attendance',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y}% attendance`;
                            }
                        }
                    }
                },
                scales: {
                    ...this.chartDefaults.scales,
                    y: {
                        ...this.chartDefaults.scales.y,
                        max: 100,
                        ticks: {
                            ...this.chartDefaults.scales.y.ticks,
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        };

        return new Chart(ctx, config);
    }

    // Utility function to generate gradient
    createGradient(ctx, colorStart, colorEnd) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, colorStart);
        gradient.addColorStop(1, colorEnd);
        return gradient;
    }

    // Update chart data dynamically
    updateChartData(chart, newData) {
        if (!chart) return;
        
        chart.data.labels = newData.labels || chart.data.labels;
        chart.data.datasets.forEach((dataset, index) => {
            if (newData.datasets && newData.datasets[index]) {
                dataset.data = newData.datasets[index].data || dataset.data;
            }
        });
        
        chart.update('active');
    }

    // Export chart as image
    exportChart(chart, filename = 'chart.png') {
        if (!chart) return;
        
        const link = document.createElement('a');
        link.download = filename;
        link.href = chart.toBase64Image();
        link.click();
    }

    // Destroy chart properly
    destroyChart(chart) {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    }
}

// Utility functions for data formatting
const ChartUtils = {
    formatDate: function(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric' 
        });
    },

    formatMonth: function(monthString) {
        const date = new Date(monthString + '-01');
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            year: 'numeric' 
        });
    },

    generateDateRange: function(startDate, endDate) {
        const dates = [];
        const currentDate = new Date(startDate);
        const finalDate = new Date(endDate);

        while (currentDate <= finalDate) {
            dates.push(new Date(currentDate).toISOString().split('T')[0]);
            currentDate.setDate(currentDate.getDate() + 1);
        }

        return dates;
    },

    calculatePercentage: function(part, total) {
        return total > 0 ? Math.round((part / total) * 100) : 0;
    }
};

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Make AttendanceCharts globally available
    window.AttendanceCharts = AttendanceCharts;
    window.ChartUtils = ChartUtils;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AttendanceCharts, ChartUtils };
}
