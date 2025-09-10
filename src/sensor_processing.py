"""
Sensor Data Processing Module for Agricultural Monitoring MVP
Analyzes sensor data and identifies critical thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class SensorAnalyzer:
    """Analyze sensor data and detect anomalies"""
    
    def __init__(self):
        """Initialize with optimal ranges for various sensors"""
        self.optimal_ranges = {
            'soil_moisture': (25, 45),      # % - optimal range for most crops
            'temperature': (18, 30),         # Celsius
            'humidity': (50, 70),            # %
            'ph': (6.0, 7.5),               # pH scale
            'nitrogen': (30, 70),            # ppm
            'phosphorus': (15, 35),          # ppm
            'potassium': (120, 250),         # ppm
            'leaf_wetness': (0, 30)          # % of time
        }
        
        self.critical_thresholds = {
            'soil_moisture': {'low': 15, 'high': 60},
            'temperature': {'low': 10, 'high': 38},
            'humidity': {'low': 30, 'high': 85},
            'ph': {'low': 5.5, 'high': 8.0},
            'nitrogen': {'low': 20, 'high': 100},
            'phosphorus': {'low': 10, 'high': 50},
            'potassium': {'low': 80, 'high': 300},
            'leaf_wetness': {'low': 0, 'high': 50}
        }
    
    def analyze_current_conditions(self, sensor_data: pd.DataFrame) -> Dict:
        """
        Analyze current sensor conditions
        
        Args:
            sensor_data: DataFrame with sensor readings
            
        Returns:
            Dictionary with analysis results
        """
        if sensor_data.empty:
            return {'status': 'error', 'message': 'No sensor data available'}
        
        # Get latest reading
        latest = sensor_data.iloc[-1]
        
        analysis = {
            'timestamp': latest['timestamp'],
            'current_readings': {},
            'status_indicators': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Analyze each sensor
        for sensor in self.optimal_ranges.keys():
            if sensor in latest:
                value = latest[sensor]
                analysis['current_readings'][sensor] = value
                
                # Check status
                optimal = self.optimal_ranges[sensor]
                critical = self.critical_thresholds[sensor]
                
                if optimal[0] <= value <= optimal[1]:
                    status = 'optimal'
                elif critical['low'] <= value <= critical['high']:
                    status = 'suboptimal'
                else:
                    status = 'critical'
                    
                analysis['status_indicators'][sensor] = status
                
                # Generate alerts for critical conditions
                if status == 'critical':
                    if value < critical['low']:
                        analysis['alerts'].append(f"CRITICAL: {sensor} too low ({value:.1f})")
                    else:
                        analysis['alerts'].append(f"CRITICAL: {sensor} too high ({value:.1f})")
        
        # Generate recommendations based on conditions
        analysis['recommendations'] = self._generate_recommendations(analysis['status_indicators'], latest)
        
        # Overall health status
        critical_count = sum(1 for s in analysis['status_indicators'].values() if s == 'critical')
        suboptimal_count = sum(1 for s in analysis['status_indicators'].values() if s == 'suboptimal')
        
        if critical_count > 0:
            analysis['overall_status'] = 'critical'
        elif suboptimal_count > 2:
            analysis['overall_status'] = 'warning'
        else:
            analysis['overall_status'] = 'healthy'
            
        return analysis
    
    def _generate_recommendations(self, status_indicators: Dict, readings: pd.Series) -> List[str]:
        """Generate actionable recommendations based on sensor status"""
        recommendations = []
        
        # Soil moisture recommendations
        if status_indicators.get('soil_moisture') == 'critical':
            if readings['soil_moisture'] < self.critical_thresholds['soil_moisture']['low']:
                recommendations.append("🚨 Immediate irrigation required - soil moisture critically low")
            else:
                recommendations.append("⚠️ Reduce irrigation - risk of waterlogging")
        
        # Temperature recommendations
        if status_indicators.get('temperature') == 'critical':
            if readings['temperature'] > self.critical_thresholds['temperature']['high']:
                recommendations.append("🌡️ Implement cooling measures - consider shade nets or misting")
            else:
                recommendations.append("❄️ Frost protection needed - consider row covers")
        
        # Nutrient recommendations
        if status_indicators.get('nitrogen') in ['critical', 'suboptimal']:
            if readings['nitrogen'] < self.optimal_ranges['nitrogen'][0]:
                recommendations.append("🌱 Apply nitrogen fertilizer - levels below optimal")
        
        if status_indicators.get('ph') == 'critical':
            if readings['ph'] < self.critical_thresholds['ph']['low']:
                recommendations.append("📊 Soil too acidic - consider lime application")
            else:
                recommendations.append("📊 Soil too alkaline - consider sulfur application")
        
        # Disease risk based on humidity and leaf wetness
        if readings.get('humidity', 0) > 75 and readings.get('leaf_wetness', 0) > 30:
            recommendations.append("🦠 High disease risk - consider fungicide application")
        
        return recommendations
    
    def calculate_trends(self, sensor_data: pd.DataFrame, window_hours: int = 24) -> Dict:
        """
        Calculate trends over specified time window
        
        Args:
            sensor_data: DataFrame with sensor readings
            window_hours: Time window for trend calculation
            
        Returns:
            Dictionary with trend information
        """
        if len(sensor_data) < 2:
            return {'status': 'insufficient_data'}
        
        # Get data for time window
        latest_time = sensor_data['timestamp'].max()
        window_start = latest_time - timedelta(hours=window_hours)
        window_data = sensor_data[sensor_data['timestamp'] >= window_start]
        
        trends = {}
        
        for sensor in self.optimal_ranges.keys():
            if sensor in window_data.columns:
                values = window_data[sensor].values
                if len(values) > 1:
                    # Simple linear trend
                    x = np.arange(len(values))
                    coeffs = np.polyfit(x, values, 1)
                    slope = coeffs[0]
                    
                    # Determine trend direction
                    if abs(slope) < 0.01:
                        trend = 'stable'
                    elif slope > 0:
                        trend = 'increasing'
                    else:
                        trend = 'decreasing'
                    
                    trends[sensor] = {
                        'trend': trend,
                        'slope': float(slope),
                        'current': float(values[-1]),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
        
        return trends
    
    def detect_stress_patterns(self, sensor_data: pd.DataFrame) -> Dict:
        """
        Detect stress patterns in sensor data
        
        Args:
            sensor_data: DataFrame with sensor readings
            
        Returns:
            Dictionary with stress pattern analysis
        """
        stress_indicators = {
            'water_stress': False,
            'nutrient_stress': False,
            'temperature_stress': False,
            'disease_conducive': False,
            'stress_level': 0  # 0-100 scale
        }
        
        if sensor_data.empty:
            return stress_indicators
        
        recent_data = sensor_data.tail(10)  # Last 10 readings
        
        # Water stress detection
        avg_moisture = recent_data['soil_moisture'].mean()
        if avg_moisture < 20:
            stress_indicators['water_stress'] = True
            stress_indicators['stress_level'] += 30
        
        # Nutrient stress detection
        avg_nitrogen = recent_data['nitrogen'].mean() if 'nitrogen' in recent_data else 50
        avg_phosphorus = recent_data['phosphorus'].mean() if 'phosphorus' in recent_data else 25
        avg_potassium = recent_data['potassium'].mean() if 'potassium' in recent_data else 150
        
        if avg_nitrogen < 30 or avg_phosphorus < 15 or avg_potassium < 100:
            stress_indicators['nutrient_stress'] = True
            stress_indicators['stress_level'] += 25
        
        # Temperature stress detection
        avg_temp = recent_data['temperature'].mean()
        temp_std = recent_data['temperature'].std()
        if avg_temp > 35 or avg_temp < 15 or temp_std > 5:
            stress_indicators['temperature_stress'] = True
            stress_indicators['stress_level'] += 25
        
        # Disease conducive conditions
        avg_humidity = recent_data['humidity'].mean()
        avg_leaf_wetness = recent_data['leaf_wetness'].mean() if 'leaf_wetness' in recent_data else 10
        if avg_humidity > 75 and avg_leaf_wetness > 30:
            stress_indicators['disease_conducive'] = True
            stress_indicators['stress_level'] += 20
        
        # Cap stress level at 100
        stress_indicators['stress_level'] = min(100, stress_indicators['stress_level'])
        
        return stress_indicators

class AlertSystem:
    """Generate and manage alerts based on sensor data"""
    
    def __init__(self):
        self.alert_history = []
        self.active_alerts = []
        
    def generate_alerts(self, analysis: Dict, stress_patterns: Dict) -> List[Dict]:
        """
        Generate alerts based on analysis and stress patterns
        
        Args:
            analysis: Current condition analysis
            stress_patterns: Detected stress patterns
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        timestamp = datetime.now()
        
        # Critical sensor alerts
        for alert_msg in analysis.get('alerts', []):
            alerts.append({
                'timestamp': timestamp,
                'type': 'critical',
                'category': 'sensor',
                'message': alert_msg,
                'priority': 1
            })
        
        # Stress pattern alerts
        if stress_patterns['water_stress']:
            alerts.append({
                'timestamp': timestamp,
                'type': 'warning',
                'category': 'stress',
                'message': 'Water stress detected - irrigation recommended',
                'priority': 2
            })
        
        if stress_patterns['nutrient_stress']:
            alerts.append({
                'timestamp': timestamp,
                'type': 'warning',
                'category': 'stress',
                'message': 'Nutrient deficiency detected - fertilization recommended',
                'priority': 2
            })
        
        if stress_patterns['disease_conducive']:
            alerts.append({
                'timestamp': timestamp,
                'type': 'warning',
                'category': 'disease',
                'message': 'Conditions favorable for disease - preventive measures recommended',
                'priority': 2
            })
        
        # High stress level alert
        if stress_patterns['stress_level'] > 70:
            alerts.append({
                'timestamp': timestamp,
                'type': 'critical',
                'category': 'overall',
                'message': f"High stress level detected ({stress_patterns['stress_level']}%) - immediate attention required",
                'priority': 1
            })
        
        self.active_alerts = alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of current alerts"""
        return {
            'total_active': len(self.active_alerts),
            'critical': sum(1 for a in self.active_alerts if a['type'] == 'critical'),
            'warnings': sum(1 for a in self.active_alerts if a['type'] == 'warning'),
            'alerts': self.active_alerts
        }
