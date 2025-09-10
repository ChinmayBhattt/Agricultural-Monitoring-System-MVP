"""
Synthetic Data Generator for Agricultural Monitoring MVP
Generates simulated multispectral images and sensor data for testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, Tuple, List
import json

class SyntheticDataGenerator:
    """Generate synthetic agricultural data for MVP testing"""
    
    def __init__(self, field_size: Tuple[int, int] = (100, 100)):
        """
        Initialize the data generator
        
        Args:
            field_size: Tuple of (height, width) for field dimensions
        """
        self.field_size = field_size
        self.bands = {
            'blue': (450, 495),      # Blue band (nm)
            'green': (495, 570),     # Green band
            'red': (620, 750),       # Red band
            'nir': (750, 900),       # Near-infrared band
            'swir': (1550, 1750)     # Short-wave infrared
        }
        
    def generate_multispectral_image(self, 
                                    health_status: str = 'healthy',
                                    noise_level: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Generate synthetic multispectral image data
        
        Args:
            health_status: 'healthy', 'stressed', or 'diseased'
            noise_level: Amount of noise to add (0-1)
            
        Returns:
            Dictionary of spectral bands
        """
        height, width = self.field_size
        images = {}
        
        # Base reflectance values for different health states
        base_values = {
            'healthy': {'blue': 0.05, 'green': 0.12, 'red': 0.08, 'nir': 0.45, 'swir': 0.25},
            'stressed': {'blue': 0.08, 'green': 0.15, 'red': 0.12, 'nir': 0.35, 'swir': 0.30},
            'diseased': {'blue': 0.10, 'green': 0.18, 'red': 0.15, 'nir': 0.25, 'swir': 0.35}
        }
        
        # Generate base image with spatial variation
        for band in self.bands.keys():
            base_value = base_values[health_status][band]
            
            # Create spatial pattern (simulating field variability)
            x = np.linspace(0, 4*np.pi, width)
            y = np.linspace(0, 4*np.pi, height)
            X, Y = np.meshgrid(x, y)
            
            # Add sinusoidal patterns for realistic field variation
            pattern = np.sin(X/2) * np.cos(Y/3) * 0.05
            
            # Generate band image
            image = np.ones((height, width)) * base_value
            image += pattern
            
            # Add random noise
            noise = np.random.normal(0, noise_level * base_value, (height, width))
            image += noise
            
            # Add some hotspots (potential problem areas)
            if health_status in ['stressed', 'diseased']:
                n_hotspots = random.randint(3, 8)
                for _ in range(n_hotspots):
                    x_center = random.randint(10, width-10)
                    y_center = random.randint(10, height-10)
                    radius = random.randint(5, 15)
                    
                    Y_grid, X_grid = np.ogrid[:height, :width]
                    mask = (X_grid - x_center)**2 + (Y_grid - y_center)**2 <= radius**2
                    image[mask] *= (1.2 if health_status == 'stressed' else 1.4)
            
            # Clip values to valid range [0, 1]
            image = np.clip(image, 0, 1)
            images[band] = image
            
        return images
    
    def generate_sensor_data(self, 
                           days: int = 30,
                           health_status: str = 'healthy') -> pd.DataFrame:
        """
        Generate synthetic sensor data over time
        
        Args:
            days: Number of days of data to generate
            health_status: Overall health status of the field
            
        Returns:
            DataFrame with sensor readings
        """
        timestamps = []
        data = []
        
        # Base sensor values for different health states
        sensor_profiles = {
            'healthy': {
                'soil_moisture': (30, 45),      # % range
                'temperature': (20, 28),         # Celsius
                'humidity': (60, 75),            # %
                'ph': (6.0, 7.0),               # pH scale
                'nitrogen': (40, 60),            # ppm
                'phosphorus': (20, 30),          # ppm
                'potassium': (150, 200),         # ppm
                'leaf_wetness': (0, 20)          # % of time
            },
            'stressed': {
                'soil_moisture': (15, 25),
                'temperature': (28, 35),
                'humidity': (40, 55),
                'ph': (5.5, 6.0),
                'nitrogen': (20, 35),
                'phosphorus': (10, 20),
                'potassium': (100, 140),
                'leaf_wetness': (25, 40)
            },
            'diseased': {
                'soil_moisture': (10, 20),
                'temperature': (30, 38),
                'humidity': (75, 90),
                'ph': (5.0, 5.8),
                'nitrogen': (15, 25),
                'phosphorus': (8, 15),
                'potassium': (80, 120),
                'leaf_wetness': (40, 60)
            }
        }
        
        profile = sensor_profiles[health_status]
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            # Generate 4 readings per day (every 6 hours)
            for hour in [0, 6, 12, 18]:
                timestamp = start_date + timedelta(days=day, hours=hour)
                timestamps.append(timestamp)
                
                reading = {
                    'timestamp': timestamp,
                    'soil_moisture': random.uniform(*profile['soil_moisture']),
                    'temperature': random.uniform(*profile['temperature']),
                    'humidity': random.uniform(*profile['humidity']),
                    'ph': random.uniform(*profile['ph']),
                    'nitrogen': random.uniform(*profile['nitrogen']),
                    'phosphorus': random.uniform(*profile['phosphorus']),
                    'potassium': random.uniform(*profile['potassium']),
                    'leaf_wetness': random.uniform(*profile['leaf_wetness'])
                }
                
                # Add daily variation
                if hour == 12:  # Noon - peak temperature
                    reading['temperature'] += random.uniform(2, 5)
                    reading['humidity'] -= random.uniform(5, 10)
                elif hour == 0:  # Midnight - coolest
                    reading['temperature'] -= random.uniform(3, 6)
                    reading['humidity'] += random.uniform(5, 15)
                    
                data.append(reading)
        
        return pd.DataFrame(data)
    
    def generate_pest_risk_data(self, conditions: Dict) -> Dict[str, float]:
        """
        Generate pest risk assessment based on environmental conditions
        
        Args:
            conditions: Dictionary of environmental conditions
            
        Returns:
            Dictionary of pest risks (0-1 scale)
        """
        risks = {}
        
        # Simple rule-based pest risk calculation
        temp = conditions.get('temperature', 25)
        humidity = conditions.get('humidity', 60)
        leaf_wetness = conditions.get('leaf_wetness', 10)
        
        # Aphids: prefer moderate temps and low humidity
        if 20 <= temp <= 30 and humidity < 70:
            risks['aphids'] = min(0.8, (30 - abs(temp - 25)) / 30)
        else:
            risks['aphids'] = 0.2
            
        # Fungal diseases: high humidity and leaf wetness
        if humidity > 75 and leaf_wetness > 30:
            risks['fungal_disease'] = min(0.9, (humidity - 60) / 40 + leaf_wetness / 100)
        else:
            risks['fungal_disease'] = 0.1
            
        # Spider mites: hot and dry conditions
        if temp > 30 and humidity < 50:
            risks['spider_mites'] = min(0.85, (temp - 25) / 15)
        else:
            risks['spider_mites'] = 0.15
            
        # Root rot: excessive moisture
        soil_moisture = conditions.get('soil_moisture', 35)
        if soil_moisture > 50:
            risks['root_rot'] = min(0.9, (soil_moisture - 40) / 30)
        else:
            risks['root_rot'] = 0.1
            
        return risks
    
    def save_synthetic_dataset(self, output_dir: str = './data'):
        """
        Generate and save a complete synthetic dataset
        
        Args:
            output_dir: Directory to save the data
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data for different scenarios
        scenarios = ['healthy', 'stressed', 'diseased']
        
        for scenario in scenarios:
            # Generate multispectral images
            images = self.generate_multispectral_image(scenario)
            
            # Save as numpy arrays
            for band, image in images.items():
                np.save(f"{output_dir}/{scenario}_{band}.npy", image)
            
            # Generate and save sensor data
            sensor_data = self.generate_sensor_data(30, scenario)
            sensor_data.to_csv(f"{output_dir}/{scenario}_sensors.csv", index=False)
            
            # Generate pest risk data for latest conditions
            latest_conditions = sensor_data.iloc[-1].to_dict()
            pest_risks = self.generate_pest_risk_data(latest_conditions)
            
            with open(f"{output_dir}/{scenario}_pest_risks.json", 'w') as f:
                json.dump(pest_risks, f, indent=2)
        
        print(f"Synthetic dataset saved to {output_dir}/")
        return True

if __name__ == "__main__":
    # Test the generator
    generator = SyntheticDataGenerator()
    generator.save_synthetic_dataset()
    print("Synthetic data generation complete!")
