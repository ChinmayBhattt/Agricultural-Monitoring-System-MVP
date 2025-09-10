<<<<<<< HEAD
# Agricultural-Monitoring-System-MVP
=======
# 🌾 Agricultural Monitoring System MVP

An AI-powered platform for monitoring crop health, soil conditions, and pest risks using multispectral/hyperspectral imaging and sensor data fusion.

## 🚀 Features

### Core Capabilities
- **Multispectral Image Analysis**: Process and analyze multispectral bands (Blue, Green, Red, NIR, SWIR)
- **Vegetation Index Calculation**: Compute NDVI, EVI, and custom health scores
- **Sensor Data Integration**: Monitor soil moisture, temperature, humidity, pH, and nutrients
- **AI-Powered Health Classification**: Machine learning models for crop health assessment
- **Pest Risk Prediction**: Rule-based and ML-driven pest and disease risk analysis
- **Yield Impact Forecasting**: Predict potential yield based on current conditions
- **Real-time Alerts**: Generate actionable alerts for critical conditions
- **Management Zone Mapping**: Segment fields into healthy, stressed, and diseased zones

### Dashboard Features
- Interactive Streamlit web interface
- Real-time data visualization
- Spectral image display and analysis
- Sensor trend monitoring
- Alert management system
- Downloadable reports

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory:**
```bash
cd ~/agri-monitor-mvp
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎮 Running the Application

### Launch the Dashboard
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Test the System
```bash
python test_system.py
```

## 📊 Using the Dashboard

### 1. Select a Scenario
- **Healthy**: Simulates optimal crop conditions
- **Stressed**: Simulates moderate stress conditions
- **Diseased**: Simulates severe crop health issues

### 2. Run Analysis
Click the "🔍 Run Analysis" button to process the data and generate insights

### 3. Explore Results
- **Overview Tab**: View alerts, critical issues, and key metrics
- **Spectral Analysis Tab**: Examine multispectral images, NDVI/EVI maps, and zone distribution
- **Sensor Data Tab**: Monitor environmental conditions and trends
- **Pest & Disease Tab**: Assess pest risks and yield impact
- **Recommendations Tab**: Get actionable insights and management recommendations

## 🏗️ Project Structure

```
agri-monitor-mvp/
├── app.py                  # Streamlit dashboard application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── data_generator.py  # Synthetic data generation
│   ├── image_processing.py # Spectral image analysis
│   ├── sensor_processing.py # Sensor data analysis
│   ├── ai_models.py       # ML models for health and pest prediction
│   └── data_fusion.py     # Data fusion and integration engine
├── data/                  # Generated synthetic data
├── models/                # Saved ML models
└── static/                # Static assets

```

## 🔧 Key Components

### 1. Data Generator (`src/data_generator.py`)
- Generates synthetic multispectral images
- Creates realistic sensor data patterns
- Simulates different health scenarios

### 2. Image Processing (`src/image_processing.py`)
- Calculates vegetation indices (NDVI, EVI)
- Performs anomaly detection
- Generates health scores

### 3. Sensor Processing (`src/sensor_processing.py`)
- Analyzes sensor readings
- Detects stress patterns
- Generates alerts for critical conditions

### 4. AI Models (`src/ai_models.py`)
- Random Forest classifier for health status
- Rule-based pest risk predictor
- Yield impact calculator

### 5. Data Fusion (`src/data_fusion.py`)
- Integrates spectral and sensor data
- Performs comprehensive field analysis
- Generates reports and recommendations

## 📈 Data Flow

1. **Data Generation/Input**: Multispectral images and sensor readings
2. **Processing**: Individual analysis of spectral and sensor data
3. **Fusion**: Combine insights from multiple data sources
4. **AI Analysis**: Apply ML models for health and risk assessment
5. **Alert Generation**: Identify critical conditions requiring attention
6. **Visualization**: Present results through interactive dashboard
7. **Recommendations**: Provide actionable insights for farmers

## 🎯 Use Cases

### For Farmers
- Monitor crop health in real-time
- Receive early warnings about pest and disease risks
- Optimize irrigation and fertilization
- Maximize yield through data-driven decisions

### For Agronomists
- Analyze field-level patterns
- Track temporal changes in crop health
- Validate intervention effectiveness
- Generate detailed reports for clients

### For Researchers
- Study crop stress responses
- Develop precision agriculture strategies
- Test new monitoring methodologies
- Collect data for agricultural research

## 🚦 Alert Levels

- **🟢 Healthy**: Optimal conditions, no action required
- **🟡 Caution**: Minor issues detected, monitor closely
- **🟠 Warning**: Significant stress or risk, intervention recommended
- **🔴 Critical**: Immediate action required to prevent crop loss

## 📊 Metrics Explained

### Vegetation Indices
- **NDVI**: Normalized Difference Vegetation Index (-1 to 1)
  - Higher values indicate healthier vegetation
- **EVI**: Enhanced Vegetation Index (-1 to 1)
  - Similar to NDVI but less sensitive to atmospheric conditions

### Stress Indicators
- **Water Stress**: Detected through low soil moisture and high temperature
- **Nutrient Stress**: Identified via NPK levels and pH imbalance
- **Disease Risk**: Calculated from humidity and leaf wetness

## 🔄 Future Enhancements

### Planned Features
- Integration with real satellite imagery APIs
- Support for drone-captured imagery
- Mobile application for field technicians
- Historical data analysis and trends
- Weather forecast integration
- Automated irrigation and fertilization recommendations
- Multi-field management capabilities
- Export to agricultural management systems

### Technical Improvements
- Deep learning models for disease identification
- Time-series forecasting for yield prediction
- Edge computing for real-time field processing
- IoT sensor integration
- Cloud deployment for scalability

## 🤝 Contributing

This is an MVP demonstration. For production deployment:
1. Replace synthetic data with real sensor and imagery inputs
2. Train models on actual agricultural data
3. Implement proper authentication and user management
4. Add database for historical data storage
5. Integrate with existing farm management systems

## 📝 License

This is a demonstration MVP for educational and evaluation purposes.

## 📞 Support

For questions about this MVP implementation, please refer to the documentation or examine the source code comments.

## 🎓 Technical Background

### Multispectral Imaging
The system analyzes light reflected from crops across different wavelengths:
- Blue (450-495 nm): Chlorophyll absorption
- Green (495-570 nm): Peak vegetation reflectance
- Red (620-750 nm): Chlorophyll absorption
- NIR (750-900 nm): Cell structure reflectance
- SWIR (1550-1750 nm): Water content

### Machine Learning Models
- **Random Forest**: Ensemble learning for robust health classification
- **Decision Trees**: Interpretable models for rule-based decisions
- **Statistical Analysis**: Trend detection and anomaly identification

---

**Note**: This is a simplified MVP demonstration. Production deployment would require:
- Real hardware integration (satellites, drones, IoT sensors)
- Calibrated spectral cameras
- Field-specific model training
- Compliance with agricultural data standards
- Integration with existing farm management systems
>>>>>>> add
