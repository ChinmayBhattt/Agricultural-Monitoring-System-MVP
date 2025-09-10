"""
Data Fusion Module for Agricultural Monitoring MVP
Combines multispectral imaging and sensor data for comprehensive analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import json

# Import our modules
from image_processing import compute_ndvi, compute_evi, zscore_anomaly, simple_health_score
from sensor_processing import SensorAnalyzer, AlertSystem
from ai_models import CropHealthClassifier, PestRiskPredictor, YieldPredictor


class DataFusionEngine:
    """Main engine for fusing spectral and sensor data"""
    
    def __init__(self):
        """Initialize the fusion engine with all components"""
        self.sensor_analyzer = SensorAnalyzer()
        self.alert_system = AlertSystem()
        self.health_classifier = CropHealthClassifier()
        self.pest_predictor = PestRiskPredictor()
        self.yield_predictor = YieldPredictor()
        
        # Store latest analysis results
        self.latest_analysis = {}
        
    def analyze_field(self,
                     spectral_images: Dict[str, np.ndarray],
                     sensor_data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive field analysis
        
        Args:
            spectral_images: Dictionary of spectral band images
            sensor_data: DataFrame with sensor readings
            
        Returns:
            Complete analysis results
        """
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'spectral_analysis': {},
            'sensor_analysis': {},
            'health_assessment': {},
            'pest_assessment': {},
            'yield_prediction': {},
            'alerts': [],
            'recommendations': [],
            'summary': {}
        }
        
        # 1. Spectral Analysis
        ndvi = compute_ndvi(spectral_images)
        evi = compute_evi(spectral_images)
        health_score = simple_health_score(ndvi, evi)
        anomaly_map = zscore_anomaly(ndvi, thresh=2.0)
        
        analysis_results['spectral_analysis'] = {
            'ndvi': {
                'mean': float(np.mean(ndvi)),
                'std': float(np.std(ndvi)),
                'min': float(np.min(ndvi)),
                'max': float(np.max(ndvi))
            },
            'evi': {
                'mean': float(np.mean(evi)),
                'std': float(np.std(evi)),
                'min': float(np.min(evi)),
                'max': float(np.max(evi))
            },
            'health_score': {
                'mean': float(np.mean(health_score)),
                'std': float(np.std(health_score))
            },
            'anomaly_percentage': float(np.sum(anomaly_map) / anomaly_map.size * 100)
        }
        
        # 2. Sensor Analysis
        current_conditions = self.sensor_analyzer.analyze_current_conditions(sensor_data)
        trends = self.sensor_analyzer.calculate_trends(sensor_data)
        stress_patterns = self.sensor_analyzer.detect_stress_patterns(sensor_data)
        
        analysis_results['sensor_analysis'] = {
            'current_conditions': current_conditions,
            'trends': trends,
            'stress_patterns': stress_patterns
        }
        
        # 3. AI-based Health Assessment
        latest_sensor = sensor_data.iloc[-1].to_dict() if not sensor_data.empty else {}
        health_prediction = self.health_classifier.predict(ndvi, evi, latest_sensor)
        
        analysis_results['health_assessment'] = health_prediction
        
        # 4. Pest Risk Assessment
        pest_risks = self.pest_predictor.predict_risks(
            latest_sensor,
            health_prediction['predicted_class']
        )
        
        analysis_results['pest_assessment'] = pest_risks
        
        # 5. Yield Prediction
        yield_impact = self.yield_predictor.predict_yield_impact(
            health_prediction['predicted_class'],
            stress_patterns['stress_level'],
            pest_risks['overall_risk']
        )
        
        analysis_results['yield_prediction'] = yield_impact
        
        # 6. Generate Alerts
        alerts = self.alert_system.generate_alerts(current_conditions, stress_patterns)
        analysis_results['alerts'] = alerts
        
        # 7. Compile Recommendations
        all_recommendations = []
        all_recommendations.extend(current_conditions.get('recommendations', []))
        all_recommendations.extend(pest_risks.get('recommendations', []))
        all_recommendations.extend(yield_impact.get('recommendations', []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        analysis_results['recommendations'] = unique_recommendations
        
        # 8. Generate Summary
        analysis_results['summary'] = self._generate_summary(analysis_results)
        
        # Store results
        self.latest_analysis = analysis_results
        
        return analysis_results
    
    def _generate_summary(self, analysis: Dict) -> Dict:
        """Generate executive summary of the analysis"""
        summary = {
            'overall_status': 'unknown',
            'key_metrics': {},
            'critical_issues': [],
            'action_priority': 'low'
        }
        
        # Determine overall status
        health_status = analysis['health_assessment'].get('predicted_class', 'unknown')
        stress_level = analysis['sensor_analysis']['stress_patterns'].get('stress_level', 0)
        pest_risk_level = analysis['pest_assessment'].get('risk_level', 'low')
        
        if health_status == 'diseased' or stress_level > 70 or pest_risk_level == 'critical':
            summary['overall_status'] = 'critical'
            summary['action_priority'] = 'immediate'
        elif health_status == 'stressed' or stress_level > 50 or pest_risk_level == 'high':
            summary['overall_status'] = 'warning'
            summary['action_priority'] = 'high'
        elif stress_level > 30 or pest_risk_level == 'moderate':
            summary['overall_status'] = 'caution'
            summary['action_priority'] = 'moderate'
        else:
            summary['overall_status'] = 'healthy'
            summary['action_priority'] = 'low'
        
        # Key metrics
        summary['key_metrics'] = {
            'health_status': health_status,
            'health_confidence': analysis['health_assessment'].get('confidence', 0),
            'stress_level': stress_level,
            'pest_risk': analysis['pest_assessment'].get('overall_risk', 0),
            'predicted_yield': analysis['yield_prediction'].get('predicted_yield_percentage', 100),
            'anomaly_coverage': analysis['spectral_analysis'].get('anomaly_percentage', 0)
        }
        
        # Critical issues
        if health_status == 'diseased':
            summary['critical_issues'].append('Crop disease detected')
        
        if stress_level > 70:
            summary['critical_issues'].append('High stress levels detected')
        
        high_risk_pests = analysis['pest_assessment'].get('high_risk_pests', [])
        if high_risk_pests:
            summary['critical_issues'].append(f"High risk for: {', '.join(high_risk_pests)}")
        
        critical_alerts = [a for a in analysis['alerts'] if a['type'] == 'critical']
        if critical_alerts:
            summary['critical_issues'].append(f"{len(critical_alerts)} critical alerts active")
        
        return summary
    
    def get_field_zones(self, 
                       health_score: np.ndarray,
                       threshold_healthy: float = 0.7,
                       threshold_stressed: float = 0.4) -> np.ndarray:
        """
        Segment field into management zones based on health score
        
        Args:
            health_score: 2D array of health scores
            threshold_healthy: Threshold for healthy zones
            threshold_stressed: Threshold for stressed zones
            
        Returns:
            Zone map (0=diseased, 1=stressed, 2=healthy)
        """
        zones = np.zeros_like(health_score, dtype=np.uint8)
        zones[health_score < threshold_stressed] = 0  # Diseased
        zones[(health_score >= threshold_stressed) & (health_score < threshold_healthy)] = 1  # Stressed
        zones[health_score >= threshold_healthy] = 2  # Healthy
        
        return zones
    
    def generate_report(self, analysis: Dict = None) -> Dict:
        """
        Generate a formatted report from analysis results
        
        Args:
            analysis: Analysis results (uses latest if not provided)
            
        Returns:
            Formatted report dictionary
        """
        if analysis is None:
            analysis = self.latest_analysis
        
        if not analysis:
            return {'error': 'No analysis available'}
        
        report = {
            'report_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'generated_at': datetime.now().isoformat(),
            'executive_summary': analysis.get('summary', {}),
            'detailed_findings': {
                'crop_health': {
                    'status': analysis['health_assessment'].get('predicted_class'),
                    'confidence': analysis['health_assessment'].get('confidence'),
                    'vegetation_indices': {
                        'ndvi': analysis['spectral_analysis']['ndvi'],
                        'evi': analysis['spectral_analysis']['evi']
                    }
                },
                'environmental_conditions': {
                    'current': analysis['sensor_analysis']['current_conditions'].get('current_readings', {}),
                    'trends': analysis['sensor_analysis'].get('trends', {}),
                    'stress_indicators': analysis['sensor_analysis']['stress_patterns']
                },
                'pest_and_disease': {
                    'risk_assessment': analysis['pest_assessment'].get('pest_risks', {}),
                    'overall_risk': analysis['pest_assessment'].get('risk_level'),
                    'high_risk_threats': analysis['pest_assessment'].get('high_risk_pests', [])
                },
                'yield_forecast': {
                    'predicted_yield': analysis['yield_prediction'].get('predicted_yield_percentage'),
                    'impact_breakdown': analysis['yield_prediction'].get('breakdown', {}),
                    'improvement_potential': analysis['yield_prediction'].get('improvement_potential')
                }
            },
            'alerts_and_actions': {
                'active_alerts': analysis.get('alerts', []),
                'recommended_actions': analysis.get('recommendations', []),
                'priority_level': analysis['summary'].get('action_priority', 'low')
            }
        }
        
        return report
    
    def export_analysis(self, filepath: str, format: str = 'json') -> bool:
        """
        Export analysis results to file
        
        Args:
            filepath: Path to save the file
            format: Export format ('json' or 'csv')
            
        Returns:
            Success status
        """
        if not self.latest_analysis:
            return False
        
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(self.latest_analysis, f, indent=2, default=str)
            elif format == 'csv':
                # Flatten the nested structure for CSV
                flat_data = self._flatten_dict(self.latest_analysis)
                df = pd.DataFrame([flat_data])
                df.to_csv(filepath, index=False)
            else:
                return False
            
            return True
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            return False
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


class RealtimeMonitor:
    """Simulates real-time monitoring capabilities"""
    
    def __init__(self, fusion_engine: DataFusionEngine):
        """
        Initialize real-time monitor
        
        Args:
            fusion_engine: Data fusion engine instance
        """
        self.fusion_engine = fusion_engine
        self.monitoring_active = False
        self.update_interval = 60  # seconds
        self.history = []
        
    def process_update(self,
                       spectral_images: Dict[str, np.ndarray],
                       sensor_data: pd.DataFrame) -> Dict:
        """
        Process a real-time update
        
        Args:
            spectral_images: New spectral images
            sensor_data: New sensor readings
            
        Returns:
            Analysis results
        """
        # Perform analysis
        results = self.fusion_engine.analyze_field(spectral_images, sensor_data)
        
        # Store in history
        self.history.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        # Keep only last 24 hours of history
        if len(self.history) > 1440:  # 24 hours * 60 minutes
            self.history = self.history[-1440:]
        
        # Check for significant changes
        changes = self._detect_significant_changes(results)
        if changes:
            results['significant_changes'] = changes
        
        return results
    
    def _detect_significant_changes(self, current_results: Dict) -> List[str]:
        """Detect significant changes from previous analysis"""
        changes = []
        
        if len(self.history) < 2:
            return changes
        
        previous = self.history[-2]['results'] if len(self.history) > 1 else None
        
        if previous:
            # Check health status change
            prev_health = previous['health_assessment'].get('predicted_class')
            curr_health = current_results['health_assessment'].get('predicted_class')
            if prev_health != curr_health:
                changes.append(f"Health status changed from {prev_health} to {curr_health}")
            
            # Check stress level change
            prev_stress = previous['sensor_analysis']['stress_patterns'].get('stress_level', 0)
            curr_stress = current_results['sensor_analysis']['stress_patterns'].get('stress_level', 0)
            if abs(curr_stress - prev_stress) > 20:
                changes.append(f"Significant stress level change: {prev_stress}% → {curr_stress}%")
            
            # Check pest risk change
            prev_risk = previous['pest_assessment'].get('overall_risk', 0)
            curr_risk = current_results['pest_assessment'].get('overall_risk', 0)
            if abs(curr_risk - prev_risk) > 0.3:
                changes.append(f"Pest risk changed significantly")
        
        return changes
