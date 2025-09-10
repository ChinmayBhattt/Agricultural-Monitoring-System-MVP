"""
Agricultural Monitoring System MVP - Streamlit Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from data_generator import SyntheticDataGenerator
from data_fusion import DataFusionEngine, RealtimeMonitor
from image_processing import compute_ndvi, compute_evi, simple_health_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Page configuration
st.set_page_config(
    page_title="AgriMonitor MVP",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .critical { background-color: #ffebee; border-left: 4px solid #f44336; }
    .warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
    .info { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = SyntheticDataGenerator()
    st.session_state.fusion_engine = DataFusionEngine()
    st.session_state.monitor = RealtimeMonitor(st.session_state.fusion_engine)
    st.session_state.current_scenario = 'healthy'
    st.session_state.analysis_results = None

def load_data(scenario):
    """Load or generate data for the selected scenario"""
    generator = st.session_state.data_generator
    
    # Generate spectral images
    images = generator.generate_multispectral_image(scenario)
    
    # Generate sensor data
    sensor_data = generator.generate_sensor_data(days=30, health_status=scenario)
    
    return images, sensor_data

def create_spectral_visualization(images, ndvi, evi, health_score):
    """Create visualization for spectral data"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # RGB Composite
    rgb = np.stack([
        images['red'],
        images['green'],
        images['blue']
    ], axis=2)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Composite')
    axes[0, 0].axis('off')
    
    # NIR
    im1 = axes[0, 1].imshow(images['nir'], cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 1].set_title('Near Infrared')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # NDVI
    im2 = axes[0, 2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 2].set_title('NDVI')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # EVI
    im3 = axes[1, 0].imshow(evi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1, 0].set_title('EVI')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Health Score
    im4 = axes[1, 1].imshow(health_score, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title('Health Score')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Zone Map
    zones = st.session_state.fusion_engine.get_field_zones(health_score)
    colors = ['#d32f2f', '#ffa726', '#66bb6a']  # Red, Orange, Green
    cmap = mcolors.ListedColormap(colors)
    im5 = axes[1, 2].imshow(zones, cmap=cmap, vmin=0, vmax=2)
    axes[1, 2].set_title('Management Zones')
    axes[1, 2].axis('off')
    cbar = plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Diseased', 'Stressed', 'Healthy'])
    
    plt.tight_layout()
    return fig

def create_sensor_charts(sensor_data, analysis):
    """Create sensor data visualizations"""
    # Time series chart
    fig_time = go.Figure()
    
    # Add traces for key sensors
    fig_time.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['soil_moisture'],
        name='Soil Moisture (%)',
        line=dict(color='blue')
    ))
    
    fig_time.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['temperature'],
        name='Temperature (°C)',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig_time.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['humidity'],
        name='Humidity (%)',
        line=dict(color='green'),
        yaxis='y3'
    ))
    
    fig_time.update_layout(
        title='Sensor Trends (Last 30 Days)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Soil Moisture (%)', side='left'),
        yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right'),
        yaxis3=dict(title='Humidity (%)', overlaying='y', side='right', position=0.95),
        hovermode='x unified',
        height=400
    )
    
    # Current conditions gauge charts
    current = analysis['sensor_analysis']['current_conditions']['current_readings']
    
    fig_gauges = go.Figure()
    
    # Soil Moisture Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current.get('soil_moisture', 0),
        title={'text': "Soil Moisture (%)"},
        delta={'reference': 35},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 25], 'color': "lightgray"},
                   {'range': [25, 45], 'color': "lightgreen"},
                   {'range': [45, 100], 'color': "lightgray"}],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 15}},
        domain={'x': [0, 0.3], 'y': [0, 1]}
    ))
    
    # Temperature Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current.get('temperature', 0),
        title={'text': "Temperature (°C)"},
        delta={'reference': 25},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "darkred"},
               'steps': [
                   {'range': [0, 18], 'color': "lightgray"},
                   {'range': [18, 30], 'color': "lightgreen"},
                   {'range': [30, 50], 'color': "lightgray"}],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 38}},
        domain={'x': [0.35, 0.65], 'y': [0, 1]}
    ))
    
    # pH Gauge
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current.get('ph', 0),
        title={'text': "Soil pH"},
        delta={'reference': 6.5},
        gauge={'axis': {'range': [4, 9]},
               'bar': {'color': "darkgreen"},
               'steps': [
                   {'range': [4, 6], 'color': "lightgray"},
                   {'range': [6, 7.5], 'color': "lightgreen"},
                   {'range': [7.5, 9], 'color': "lightgray"}],
               'threshold': {
                   'line': {'color': "red", 'width': 4},
                   'thickness': 0.75,
                   'value': 5.5}},
        domain={'x': [0.7, 1], 'y': [0, 1]}
    ))
    
    fig_gauges.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    
    return fig_time, fig_gauges

def create_pest_risk_chart(pest_assessment):
    """Create pest risk visualization"""
    pest_risks = pest_assessment['pest_risks']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(pest_risks.keys()),
            y=list(pest_risks.values()),
            marker_color=['red' if v > 0.7 else 'orange' if v > 0.4 else 'green' 
                         for v in pest_risks.values()],
            text=[f'{v:.1%}' for v in pest_risks.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Pest & Disease Risk Assessment',
        xaxis_title='Pest/Disease Type',
        yaxis_title='Risk Level',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300
    )
    
    return fig

def display_alerts(alerts):
    """Display alerts in the dashboard"""
    if not alerts:
        st.success("✅ No active alerts")
        return
    
    for alert in alerts:
        alert_class = 'critical' if alert['type'] == 'critical' else 'warning'
        icon = '🚨' if alert['type'] == 'critical' else '⚠️'
        st.markdown(
            f'<div class="alert-box {alert_class}">'
            f'{icon} <strong>{alert["category"].upper()}</strong>: {alert["message"]}'
            f'</div>',
            unsafe_allow_html=True
        )

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Agricultural Monitoring System MVP</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        
        # Scenario selection
        scenario = st.selectbox(
            "Select Field Scenario",
            options=['healthy', 'stressed', 'diseased'],
            index=['healthy', 'stressed', 'diseased'].index(st.session_state.current_scenario)
        )
        
        if scenario != st.session_state.current_scenario:
            st.session_state.current_scenario = scenario
            st.session_state.analysis_results = None
        
        # Analysis button
        if st.button("🔍 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing field data..."):
                # Load data
                images, sensor_data = load_data(scenario)
                
                # Run analysis
                analysis = st.session_state.fusion_engine.analyze_field(images, sensor_data)
                st.session_state.analysis_results = analysis
                st.success("Analysis complete!")
        
        # Export options
        st.divider()
        st.subheader("Export Options")
        
        if st.session_state.analysis_results:
            if st.button("📄 Generate Report"):
                report = st.session_state.fusion_engine.generate_report()
                st.download_button(
                    label="Download Report (JSON)",
                    data=str(report),
                    file_name=f"agri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Info section
        st.divider()
        st.info("""
        **About this MVP:**
        - Multispectral imaging analysis
        - Real-time sensor monitoring
        - AI-powered health assessment
        - Pest risk prediction
        - Yield impact analysis
        """)
    
    # Main content
    if st.session_state.analysis_results:
        analysis = st.session_state.analysis_results
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status = analysis['summary']['overall_status']
            color = 'inverse' if status == 'critical' else 'primary' if status == 'warning' else 'normal'
            st.metric(
                "Overall Status",
                status.upper(),
                delta=None
            )
        
        with col2:
            st.metric(
                "Health Status",
                analysis['health_assessment']['predicted_class'].upper(),
                f"{analysis['health_assessment']['confidence']:.1%} confidence"
            )
        
        with col3:
            st.metric(
                "Stress Level",
                f"{analysis['sensor_analysis']['stress_patterns']['stress_level']}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Pest Risk",
                analysis['pest_assessment']['risk_level'].upper(),
                f"{analysis['pest_assessment']['overall_risk']:.1%}"
            )
        
        with col5:
            st.metric(
                "Yield Forecast",
                f"{analysis['yield_prediction']['predicted_yield_percentage']:.0f}%",
                f"{analysis['yield_prediction']['yield_impact']:.1f}%"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Overview", "🛰️ Spectral Analysis", "📈 Sensor Data", 
             "🐛 Pest & Disease", "📋 Recommendations"]
        )
        
        with tab1:
            # Alerts
            st.subheader("🚨 Active Alerts")
            display_alerts(analysis['alerts'])
            
            # Critical Issues
            if analysis['summary']['critical_issues']:
                st.subheader("⚠️ Critical Issues")
                for issue in analysis['summary']['critical_issues']:
                    st.warning(issue)
            
            # Quick Stats
            st.subheader("📊 Field Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Vegetation Indices:**")
                st.write(f"- NDVI: {analysis['spectral_analysis']['ndvi']['mean']:.3f}")
                st.write(f"- EVI: {analysis['spectral_analysis']['evi']['mean']:.3f}")
                st.write(f"- Anomaly Coverage: {analysis['spectral_analysis']['anomaly_percentage']:.1f}%")
            
            with col2:
                st.markdown("**Environmental Conditions:**")
                current = analysis['sensor_analysis']['current_conditions']['current_readings']
                st.write(f"- Soil Moisture: {current.get('soil_moisture', 0):.1f}%")
                st.write(f"- Temperature: {current.get('temperature', 0):.1f}°C")
                st.write(f"- Humidity: {current.get('humidity', 0):.1f}%")
        
        with tab2:
            st.subheader("🛰️ Multispectral Imaging Analysis")
            
            # Load images for visualization
            images, _ = load_data(st.session_state.current_scenario)
            ndvi = compute_ndvi(images)
            evi = compute_evi(images)
            health_score = simple_health_score(ndvi, evi)
            
            # Create and display spectral visualization
            fig = create_spectral_visualization(images, ndvi, evi, health_score)
            st.pyplot(fig)
            
            # Zone statistics
            zones = st.session_state.fusion_engine.get_field_zones(health_score)
            zone_stats = {
                'Healthy': np.sum(zones == 2) / zones.size * 100,
                'Stressed': np.sum(zones == 1) / zones.size * 100,
                'Diseased': np.sum(zones == 0) / zones.size * 100
            }
            
            st.subheader("Management Zone Distribution")
            fig_zones = px.pie(
                values=list(zone_stats.values()),
                names=list(zone_stats.keys()),
                color_discrete_map={'Healthy': '#66bb6a', 'Stressed': '#ffa726', 'Diseased': '#d32f2f'}
            )
            st.plotly_chart(fig_zones, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Sensor Data Analysis")
            
            # Load sensor data
            _, sensor_data = load_data(st.session_state.current_scenario)
            
            # Create and display sensor charts
            fig_time, fig_gauges = create_sensor_charts(sensor_data, analysis)
            
            st.plotly_chart(fig_gauges, use_container_width=True)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Trends
            st.subheader("📊 24-Hour Trends")
            trends = analysis['sensor_analysis']['trends']
            
            if trends and trends != {'status': 'insufficient_data'}:
                trend_cols = st.columns(4)
                for i, (sensor, data) in enumerate(list(trends.items())[:4]):
                    with trend_cols[i]:
                        trend_icon = "📈" if data['trend'] == 'increasing' else "📉" if data['trend'] == 'decreasing' else "➡️"
                        st.metric(
                            sensor.replace('_', ' ').title(),
                            f"{data['current']:.1f}",
                            f"{trend_icon} {data['trend']}"
                        )
        
        with tab4:
            st.subheader("🐛 Pest & Disease Risk Analysis")
            
            # Pest risk chart
            fig_pest = create_pest_risk_chart(analysis['pest_assessment'])
            st.plotly_chart(fig_pest, use_container_width=True)
            
            # Risk recommendations
            if analysis['pest_assessment']['recommendations']:
                st.subheader("Risk Mitigation Strategies")
                for rec in analysis['pest_assessment']['recommendations']:
                    st.info(rec)
            
            # Yield impact
            st.subheader("📉 Yield Impact Analysis")
            
            yield_data = analysis['yield_prediction']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Yield", f"{yield_data['predicted_yield_percentage']:.0f}%")
                st.metric("Improvement Potential", f"+{yield_data['improvement_potential']:.0f}%")
            
            with col2:
                # Impact breakdown
                breakdown = yield_data['breakdown']
                fig_impact = go.Figure(data=[
                    go.Bar(
                        x=['Health', 'Stress', 'Pest'],
                        y=[breakdown['health_impact'], breakdown['stress_impact'], breakdown['pest_impact']],
                        marker_color=['red' if v < 0 else 'green' for v in 
                                     [breakdown['health_impact'], breakdown['stress_impact'], breakdown['pest_impact']]],
                        text=[f"{v:+.1f}%" for v in 
                             [breakdown['health_impact'], breakdown['stress_impact'], breakdown['pest_impact']]],
                        textposition='auto'
                    )
                ])
                fig_impact.update_layout(
                    title='Yield Impact Factors',
                    yaxis_title='Impact (%)',
                    height=250
                )
                st.plotly_chart(fig_impact, use_container_width=True)
        
        with tab5:
            st.subheader("📋 Recommendations & Action Items")
            
            # Priority
            priority = analysis['summary']['action_priority']
            priority_color = 'red' if priority == 'immediate' else 'orange' if priority == 'high' else 'blue'
            st.markdown(f"**Action Priority:** <span style='color:{priority_color}; font-weight:bold'>{priority.upper()}</span>", 
                       unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Recommended Actions")
            
            if analysis['recommendations']:
                for i, rec in enumerate(analysis['recommendations'], 1):
                    st.write(f"{i}. {rec}")
            else:
                st.success("No immediate actions required. Continue monitoring.")
            
            # Feature importance (if available)
            if analysis['health_assessment'].get('feature_importance'):
                st.subheader("📊 Key Factors Influencing Health")
                
                importance = analysis['health_assessment']['feature_importance']
                if importance:
                    fig_imp = go.Figure(data=[
                        go.Bar(
                            x=list(importance.values()),
                            y=list(importance.keys()),
                            orientation='h',
                            marker_color='green'
                        )
                    ])
                    fig_imp.update_layout(
                        title='Feature Importance',
                        xaxis_title='Importance',
                        height=400
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
    
    else:
        # Landing page
        st.info("👈 Select a scenario and click 'Run Analysis' to begin monitoring")
        
        # Display demo information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🛰️ Spectral Imaging
            - Multispectral band analysis
            - NDVI & EVI calculation
            - Anomaly detection
            - Zone mapping
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Sensor Integration
            - Soil moisture monitoring
            - Temperature tracking
            - Nutrient analysis
            - pH measurement
            """)
        
        with col3:
            st.markdown("""
            ### 🤖 AI Analysis
            - Health classification
            - Pest risk prediction
            - Yield forecasting
            - Smart recommendations
            """)

if __name__ == "__main__":
    main()
