#!/usr/bin/env python3
"""
Test script for Agricultural Monitoring System MVP
Validates all components and generates sample outputs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime
import json

# Import our modules
from data_generator import SyntheticDataGenerator
from image_processing import compute_ndvi, compute_evi, simple_health_score, zscore_anomaly
from sensor_processing import SensorAnalyzer, AlertSystem
from ai_models import CropHealthClassifier, PestRiskPredictor, YieldPredictor
from data_fusion import DataFusionEngine, RealtimeMonitor

def test_data_generation():
    """Test synthetic data generation"""
    print("\n" + "="*50)
    print("Testing Data Generation Module")
    print("="*50)
    
    generator = SyntheticDataGenerator(field_size=(50, 50))
    
    # Test image generation
    for scenario in ['healthy', 'stressed', 'diseased']:
        print(f"\nGenerating {scenario} scenario...")
        images = generator.generate_multispectral_image(scenario)
        print(f"  ✓ Generated {len(images)} spectral bands")
        for band, img in images.items():
            print(f"    - {band}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    
    # Test sensor data generation
    sensor_data = generator.generate_sensor_data(days=7, health_status='stressed')
    print(f"\n  ✓ Generated sensor data: {len(sensor_data)} readings over 7 days")
    print(f"    Columns: {list(sensor_data.columns)}")
    
    # Test pest risk data
    conditions = sensor_data.iloc[-1].to_dict()
    pest_risks = generator.generate_pest_risk_data(conditions)
    print(f"\n  ✓ Generated pest risk data:")
    for pest, risk in pest_risks.items():
        print(f"    - {pest}: {risk:.2%}")
    
    return True

def test_image_processing():
    """Test image processing functions"""
    print("\n" + "="*50)
    print("Testing Image Processing Module")
    print("="*50)
    
    generator = SyntheticDataGenerator(field_size=(50, 50))
    images = generator.generate_multispectral_image('stressed')
    
    # Test NDVI
    ndvi = compute_ndvi(images)
    print(f"\n  ✓ NDVI computed: mean={ndvi.mean():.3f}, std={ndvi.std():.3f}")
    
    # Test EVI
    evi = compute_evi(images)
    print(f"  ✓ EVI computed: mean={evi.mean():.3f}, std={evi.std():.3f}")
    
    # Test health score
    health = simple_health_score(ndvi, evi)
    print(f"  ✓ Health score computed: mean={health.mean():.3f}, std={health.std():.3f}")
    
    # Test anomaly detection
    anomaly = zscore_anomaly(ndvi)
    anomaly_pct = (anomaly.sum() / anomaly.size) * 100
    print(f"  ✓ Anomaly detection: {anomaly_pct:.1f}% of field flagged")
    
    return True

def test_sensor_processing():
    """Test sensor data processing"""
    print("\n" + "="*50)
    print("Testing Sensor Processing Module")
    print("="*50)
    
    generator = SyntheticDataGenerator()
    sensor_data = generator.generate_sensor_data(days=7, health_status='diseased')
    
    analyzer = SensorAnalyzer()
    
    # Test current conditions analysis
    analysis = analyzer.analyze_current_conditions(sensor_data)
    print(f"\n  ✓ Current conditions analyzed:")
    print(f"    - Overall status: {analysis['overall_status']}")
    print(f"    - Alerts: {len(analysis['alerts'])}")
    print(f"    - Recommendations: {len(analysis['recommendations'])}")
    
    # Test trend calculation
    trends = analyzer.calculate_trends(sensor_data, window_hours=24)
    if trends != {'status': 'insufficient_data'}:
        print(f"\n  ✓ Trends calculated for {len(trends)} sensors")
        for sensor in list(trends.keys())[:3]:
            print(f"    - {sensor}: {trends[sensor]['trend']}")
    
    # Test stress detection
    stress = analyzer.detect_stress_patterns(sensor_data)
    print(f"\n  ✓ Stress patterns detected:")
    print(f"    - Stress level: {stress['stress_level']}%")
    print(f"    - Water stress: {stress['water_stress']}")
    print(f"    - Disease conducive: {stress['disease_conducive']}")
    
    # Test alert system
    alert_system = AlertSystem()
    alerts = alert_system.generate_alerts(analysis, stress)
    print(f"\n  ✓ Generated {len(alerts)} alerts")
    
    return True

def test_ai_models():
    """Test AI models"""
    print("\n" + "="*50)
    print("Testing AI Models")
    print("="*50)
    
    generator = SyntheticDataGenerator(field_size=(50, 50))
    
    # Generate test data
    images = generator.generate_multispectral_image('stressed')
    sensor_data = generator.generate_sensor_data(days=1, health_status='stressed')
    
    ndvi = compute_ndvi(images)
    evi = compute_evi(images)
    latest_sensor = sensor_data.iloc[-1].to_dict()
    
    # Test health classifier
    print("\n  Testing Health Classifier...")
    classifier = CropHealthClassifier()
    health_pred = classifier.predict(ndvi, evi, latest_sensor)
    print(f"    ✓ Predicted class: {health_pred['predicted_class']}")
    print(f"    ✓ Confidence: {health_pred['confidence']:.2%}")
    
    # Test pest predictor
    print("\n  Testing Pest Risk Predictor...")
    pest_predictor = PestRiskPredictor()
    pest_risks = pest_predictor.predict_risks(latest_sensor, 'stressed')
    print(f"    ✓ Overall risk: {pest_risks['overall_risk']:.2%}")
    print(f"    ✓ Risk level: {pest_risks['risk_level']}")
    
    # Test yield predictor
    print("\n  Testing Yield Predictor...")
    yield_predictor = YieldPredictor()
    yield_pred = yield_predictor.predict_yield_impact('stressed', 45, 0.6)
    print(f"    ✓ Predicted yield: {yield_pred['predicted_yield_percentage']:.0f}%")
    print(f"    ✓ Impact: {yield_pred['yield_impact']:.1f}%")
    
    return True

def test_data_fusion():
    """Test data fusion engine"""
    print("\n" + "="*50)
    print("Testing Data Fusion Engine")
    print("="*50)
    
    generator = SyntheticDataGenerator(field_size=(50, 50))
    fusion_engine = DataFusionEngine()
    
    # Test for each scenario
    for scenario in ['healthy', 'stressed', 'diseased']:
        print(f"\n  Testing {scenario} scenario...")
        
        # Generate data
        images = generator.generate_multispectral_image(scenario)
        sensor_data = generator.generate_sensor_data(days=7, health_status=scenario)
        
        # Run analysis
        results = fusion_engine.analyze_field(images, sensor_data)
        
        print(f"    ✓ Analysis complete")
        print(f"      - Overall status: {results['summary']['overall_status']}")
        print(f"      - Health: {results['health_assessment']['predicted_class']}")
        print(f"      - Stress level: {results['sensor_analysis']['stress_patterns']['stress_level']}%")
        print(f"      - Pest risk: {results['pest_assessment']['risk_level']}")
        print(f"      - Predicted yield: {results['yield_prediction']['predicted_yield_percentage']:.0f}%")
        print(f"      - Alerts: {len(results['alerts'])}")
        print(f"      - Recommendations: {len(results['recommendations'])}")
    
    # Test report generation
    print("\n  Testing Report Generation...")
    report = fusion_engine.generate_report()
    print(f"    ✓ Report generated with {len(report)} sections")
    
    # Test export
    print("\n  Testing Export...")
    success = fusion_engine.export_analysis('test_output.json', 'json')
    if success:
        print(f"    ✓ Analysis exported to test_output.json")
        os.remove('test_output.json')  # Clean up
    
    return True

def test_realtime_monitor():
    """Test real-time monitoring"""
    print("\n" + "="*50)
    print("Testing Real-time Monitor")
    print("="*50)
    
    generator = SyntheticDataGenerator(field_size=(50, 50))
    fusion_engine = DataFusionEngine()
    monitor = RealtimeMonitor(fusion_engine)
    
    print("\n  Simulating real-time updates...")
    
    # Simulate 3 updates
    scenarios = ['healthy', 'healthy', 'stressed']
    for i, scenario in enumerate(scenarios):
        print(f"\n  Update {i+1}: {scenario} scenario")
        
        images = generator.generate_multispectral_image(scenario)
        sensor_data = generator.generate_sensor_data(days=1, health_status=scenario)
        
        results = monitor.process_update(images, sensor_data)
        
        if 'significant_changes' in results:
            print(f"    ✓ Significant changes detected:")
            for change in results['significant_changes']:
                print(f"      - {change}")
        else:
            print(f"    ✓ No significant changes")
    
    print(f"\n  ✓ Monitor history: {len(monitor.history)} updates recorded")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" AGRICULTURAL MONITORING SYSTEM MVP - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Image Processing", test_image_processing),
        ("Sensor Processing", test_sensor_processing),
        ("AI Models", test_ai_models),
        ("Data Fusion", test_data_fusion),
        ("Real-time Monitor", test_realtime_monitor)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"\n✅ {test_name} - PASSED")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ {test_name} - FAILED: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:20} {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        print("\n📌 Next steps:")
        print("   1. Run 'streamlit run app.py' to launch the dashboard")
        print("   2. Select a scenario and click 'Run Analysis'")
        print("   3. Explore the different tabs to see the results")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
