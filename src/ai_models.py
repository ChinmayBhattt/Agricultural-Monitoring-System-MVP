"""
AI Models for Agricultural Monitoring MVP
Simple models for crop health classification and pest risk prediction
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Tuple, List
import joblib
import os

class CropHealthClassifier:
    """Simple classifier for crop health status"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier
        
        Args:
            model_type: 'random_forest' or 'decision_tree'
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=8,
                random_state=42
            )
        
        self.is_trained = False
        self.feature_names = [
            'ndvi_mean', 'ndvi_std', 'evi_mean', 'evi_std',
            'soil_moisture', 'temperature', 'humidity', 'ph',
            'nitrogen', 'phosphorus', 'potassium', 'leaf_wetness'
        ]
        
        # Class labels
        self.classes = ['healthy', 'stressed', 'diseased']
        
    def prepare_features(self, 
                        ndvi: np.ndarray, 
                        evi: np.ndarray,
                        sensor_data: Dict) -> np.ndarray:
        """
        Prepare feature vector from indices and sensor data
        
        Args:
            ndvi: NDVI array
            evi: EVI array
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Feature vector
        """
        features = []
        
        # Spectral features
        features.append(np.mean(ndvi))
        features.append(np.std(ndvi))
        features.append(np.mean(evi))
        features.append(np.std(evi))
        
        # Sensor features
        features.append(sensor_data.get('soil_moisture', 35))
        features.append(sensor_data.get('temperature', 25))
        features.append(sensor_data.get('humidity', 60))
        features.append(sensor_data.get('ph', 6.5))
        features.append(sensor_data.get('nitrogen', 45))
        features.append(sensor_data.get('phosphorus', 25))
        features.append(sensor_data.get('potassium', 175))
        features.append(sensor_data.get('leaf_wetness', 15))
        
        return np.array(features).reshape(1, -1)
    
    def train_with_synthetic_data(self):
        """Train the model with synthetic data for MVP"""
        # Generate synthetic training data
        n_samples = 300
        X_train = []
        y_train = []
        
        # Generate samples for each class
        for class_idx, class_name in enumerate(self.classes):
            for _ in range(n_samples // 3):
                if class_name == 'healthy':
                    features = [
                        np.random.uniform(0.6, 0.9),    # ndvi_mean
                        np.random.uniform(0.05, 0.15),  # ndvi_std
                        np.random.uniform(0.5, 0.8),    # evi_mean
                        np.random.uniform(0.05, 0.15),  # evi_std
                        np.random.uniform(30, 45),      # soil_moisture
                        np.random.uniform(20, 28),      # temperature
                        np.random.uniform(60, 75),      # humidity
                        np.random.uniform(6.0, 7.0),    # ph
                        np.random.uniform(40, 60),      # nitrogen
                        np.random.uniform(20, 30),      # phosphorus
                        np.random.uniform(150, 200),    # potassium
                        np.random.uniform(0, 20)        # leaf_wetness
                    ]
                elif class_name == 'stressed':
                    features = [
                        np.random.uniform(0.3, 0.6),
                        np.random.uniform(0.1, 0.2),
                        np.random.uniform(0.25, 0.5),
                        np.random.uniform(0.1, 0.2),
                        np.random.uniform(15, 25),
                        np.random.uniform(28, 35),
                        np.random.uniform(40, 55),
                        np.random.uniform(5.5, 6.0),
                        np.random.uniform(20, 35),
                        np.random.uniform(10, 20),
                        np.random.uniform(100, 140),
                        np.random.uniform(25, 40)
                    ]
                else:  # diseased
                    features = [
                        np.random.uniform(0.1, 0.3),
                        np.random.uniform(0.15, 0.25),
                        np.random.uniform(0.05, 0.25),
                        np.random.uniform(0.15, 0.25),
                        np.random.uniform(10, 20),
                        np.random.uniform(30, 38),
                        np.random.uniform(75, 90),
                        np.random.uniform(5.0, 5.8),
                        np.random.uniform(15, 25),
                        np.random.uniform(8, 15),
                        np.random.uniform(80, 120),
                        np.random.uniform(40, 60)
                    ]
                
                X_train.append(features)
                y_train.append(class_idx)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        return True
    
    def predict(self, 
                ndvi: np.ndarray, 
                evi: np.ndarray,
                sensor_data: Dict) -> Dict:
        """
        Predict crop health status
        
        Args:
            ndvi: NDVI array
            evi: EVI array
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            self.train_with_synthetic_data()
        
        # Prepare features
        features = self.prepare_features(ndvi, evi, sensor_data)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get feature importance (for random forest)
        feature_importance = {}
        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
            for i, name in enumerate(self.feature_names):
                feature_importance[name] = float(importances[i])
        
        return {
            'predicted_class': self.classes[prediction],
            'confidence': float(np.max(probabilities)),
            'class_probabilities': {
                self.classes[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'feature_importance': feature_importance
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.is_trained:
            joblib.dump(self.model, filepath)
            return True
        return False
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
            return True
        return False


class PestRiskPredictor:
    """Rule-based pest risk prediction system"""
    
    def __init__(self):
        """Initialize the pest risk predictor"""
        self.risk_thresholds = {
            'aphids': {
                'temp_range': (20, 30),
                'humidity_max': 70,
                'risk_factors': ['temperature', 'humidity', 'nitrogen']
            },
            'fungal_disease': {
                'humidity_min': 75,
                'leaf_wetness_min': 30,
                'risk_factors': ['humidity', 'leaf_wetness', 'temperature']
            },
            'spider_mites': {
                'temp_min': 30,
                'humidity_max': 50,
                'risk_factors': ['temperature', 'humidity']
            },
            'root_rot': {
                'soil_moisture_min': 50,
                'risk_factors': ['soil_moisture', 'temperature']
            }
        }
    
    def predict_risks(self, 
                     sensor_data: Dict,
                     health_status: str = 'healthy') -> Dict:
        """
        Predict pest risks based on environmental conditions
        
        Args:
            sensor_data: Dictionary of sensor readings
            health_status: Current crop health status
            
        Returns:
            Dictionary of pest risks
        """
        risks = {}
        recommendations = []
        
        # Base risk multiplier based on health status
        health_multiplier = {
            'healthy': 0.8,
            'stressed': 1.2,
            'diseased': 1.5
        }.get(health_status, 1.0)
        
        # Aphids risk
        temp = sensor_data.get('temperature', 25)
        humidity = sensor_data.get('humidity', 60)
        nitrogen = sensor_data.get('nitrogen', 45)
        
        aphid_risk = 0.0
        if 20 <= temp <= 30 and humidity < 70:
            aphid_risk = 0.5 + (nitrogen - 30) / 100
            if temp >= 23 and temp <= 27:
                aphid_risk += 0.2
        
        risks['aphids'] = min(1.0, aphid_risk * health_multiplier)
        
        if risks['aphids'] > 0.6:
            recommendations.append("Monitor for aphid infestations - consider insecticidal soap")
        
        # Fungal disease risk
        leaf_wetness = sensor_data.get('leaf_wetness', 10)
        
        fungal_risk = 0.0
        if humidity > 75 and leaf_wetness > 30:
            fungal_risk = 0.6 + (humidity - 75) / 50 + (leaf_wetness - 30) / 70
        elif humidity > 70 or leaf_wetness > 25:
            fungal_risk = 0.3
        
        risks['fungal_disease'] = min(1.0, fungal_risk * health_multiplier)
        
        if risks['fungal_disease'] > 0.7:
            recommendations.append("High fungal disease risk - apply preventive fungicide")
        
        # Spider mites risk
        mite_risk = 0.0
        if temp > 30 and humidity < 50:
            mite_risk = 0.6 + (temp - 30) / 20 - humidity / 100
        elif temp > 28 and humidity < 60:
            mite_risk = 0.3
        
        risks['spider_mites'] = min(1.0, mite_risk * health_multiplier)
        
        if risks['spider_mites'] > 0.6:
            recommendations.append("Spider mite risk elevated - increase monitoring")
        
        # Root rot risk
        soil_moisture = sensor_data.get('soil_moisture', 35)
        
        rot_risk = 0.0
        if soil_moisture > 50:
            rot_risk = 0.5 + (soil_moisture - 50) / 50
            if temp < 20:
                rot_risk += 0.2
        
        risks['root_rot'] = min(1.0, rot_risk * health_multiplier)
        
        if risks['root_rot'] > 0.6:
            recommendations.append("Root rot risk high - improve drainage and reduce watering")
        
        # Calculate overall risk
        overall_risk = np.mean(list(risks.values()))
        
        return {
            'pest_risks': risks,
            'overall_risk': float(overall_risk),
            'risk_level': self._get_risk_level(overall_risk),
            'recommendations': recommendations,
            'high_risk_pests': [pest for pest, risk in risks.items() if risk > 0.6]
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level"""
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'moderate'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'critical'


class YieldPredictor:
    """Simple yield prediction based on health indicators"""
    
    def __init__(self):
        """Initialize yield predictor"""
        self.baseline_yield = 100  # Baseline yield percentage
        
    def predict_yield_impact(self,
                            health_status: str,
                            stress_level: int,
                            pest_risk: float) -> Dict:
        """
        Predict yield impact based on current conditions
        
        Args:
            health_status: Current crop health status
            stress_level: Stress level (0-100)
            pest_risk: Overall pest risk (0-1)
            
        Returns:
            Yield prediction results
        """
        # Base yield based on health status
        health_impact = {
            'healthy': 0,
            'stressed': -15,
            'diseased': -35
        }.get(health_status, 0)
        
        # Stress impact
        stress_impact = -(stress_level / 100) * 20
        
        # Pest impact
        pest_impact = -pest_risk * 25
        
        # Calculate predicted yield
        total_impact = health_impact + stress_impact + pest_impact
        predicted_yield = max(0, self.baseline_yield + total_impact)
        
        # Yield improvement potential with interventions
        improvement_potential = 0
        if health_status != 'healthy':
            improvement_potential += 10
        if stress_level > 50:
            improvement_potential += 15
        if pest_risk > 0.5:
            improvement_potential += 10
        
        return {
            'predicted_yield_percentage': float(predicted_yield),
            'yield_impact': float(total_impact),
            'breakdown': {
                'health_impact': float(health_impact),
                'stress_impact': float(stress_impact),
                'pest_impact': float(pest_impact)
            },
            'improvement_potential': float(improvement_potential),
            'recommendations': self._get_yield_recommendations(
                health_status, stress_level, pest_risk
            )
        }
    
    def _get_yield_recommendations(self, 
                                  health_status: str,
                                  stress_level: int,
                                  pest_risk: float) -> List[str]:
        """Generate recommendations to improve yield"""
        recommendations = []
        
        if health_status == 'diseased':
            recommendations.append("Immediate disease management required to prevent yield loss")
        elif health_status == 'stressed':
            recommendations.append("Address stress factors to improve crop health")
        
        if stress_level > 70:
            recommendations.append("Critical stress levels - immediate intervention needed")
        elif stress_level > 50:
            recommendations.append("Moderate stress detected - monitor closely")
        
        if pest_risk > 0.7:
            recommendations.append("Implement integrated pest management strategies")
        
        if not recommendations:
            recommendations.append("Maintain current management practices")
        
        return recommendations
