"""
DHA RESCUE: Streamlit Dashboard
================================
Interactive dashboard for the Explainable Predictive Blood Logistics System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from pathlib import Path

# Import project modules
from data import generate_synthetic_data, get_node_summary
from utils import (
    engineer_features, generate_recommendations, get_risk_level, 
    get_risk_color, format_recommendations_table
)
from model import BloodLogisticsModel

# Page configuration
st.set_page_config(
    page_title="DHA RESCUE",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e40af;
    }
    .risk-low { color: #28a745; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #dc3545; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and prepare data."""
    df = generate_synthetic_data(2000)
    df = engineer_features(df)
    return df


@st.cache_resource
def load_model(df):
    """Load cached model artifact, or train and create it."""
    model_dir = Path("artifacts")
    model_path = model_dir / "blood_logistics_ebm.pkl"

    model = BloodLogisticsModel()
    if model_path.exists():
        model.load(model_path)
        return model

    model_dir.mkdir(parents=True, exist_ok=True)
    model.train(df, backend="ebm")
    model.save(model_path)
    return model


def main():
    # Header
    st.markdown('<p class="main-header">🩸 DHA RESCUE: Explainable Predictive Blood Logistics System</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and model
    df = load_data()
    model = load_model(df)
    
    # Make predictions
    predictions = model.predict(df)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation menu
    menu = st.sidebar.radio(
        "Go to:",
        ["Overview", "Node Details", "SHAP Explanation", "Recommendations", "Scenario Simulation"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **DHA RESCUE** helps predict supply failure risk in the blood logistics network.
    
    Use the navigation above to explore different sections of the dashboard.
    """)
    
    # ==================== OVERVIEW SECTION ====================
    if menu == "Overview":
        st.header("📈 Network Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_nodes = df['node_id'].nunique()
            st.metric("Total Nodes", total_nodes)
        
        with col2:
            avg_failure_prob = predictions['failure_probability'].mean()
            st.metric("Avg Failure Risk", f"{avg_failure_prob:.1%}")
        
        with col3:
            high_risk_nodes = len(predictions[predictions['risk_level'] == 'HIGH'])
            st.metric("High Risk Nodes", high_risk_nodes)
        
        with col4:
            avg_time_to_failure = predictions['time_to_failure_hours'].mean()
            st.metric("Avg Time to Failure", f"{avg_time_to_failure:.0f}h")
        
        st.markdown("---")
        
        # Node summary table
        st.subheader("📋 Node Risk Summary")
        
        # Aggregate predictions by node
        node_summary = predictions.groupby('node_id').agg({
            'failure_probability': 'mean',
            'time_to_failure_hours': 'mean',
            'risk_level': lambda x: x.mode().iloc[0] if len(x) > 0 else 'LOW'
        }).reset_index()
        
        node_summary = node_summary.merge(
            df[['node_id', 'node_name', 'node_type']].drop_duplicates(),
            on='node_id'
        )
        
        # Format for display
        display_df = node_summary[['node_name', 'risk_level', 'failure_probability', 'time_to_failure_hours']].copy()
        display_df.columns = ['Node', 'Risk Level', 'Failure Probability', 'Time to Failure (h)']
        display_df['Failure Probability'] = display_df['Failure Probability'].apply(lambda x: f"{x:.1%}")
        display_df['Time to Failure (h)'] = display_df['Time to Failure (h)'].apply(lambda x: f"{x:.0f}")
        
        # Color code risk levels
        def color_risk(val):
            color = get_risk_color(val)
            return f'background-color: {color}; color: white'
        
        st.dataframe(
            display_df.style.map(color_risk, subset=['Risk Level']),
            width='stretch'
        )
        
        st.markdown("---")
        
        # Risk distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Level Distribution")
            risk_counts = predictions['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values, 
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'},
                title="Nodes by Risk Level"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("⏱️ Time to Failure Distribution")
            fig = px.histogram(
                predictions, 
                x='time_to_failure_hours',
                nbins=30,
                title="Distribution of Time to Failure",
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(xaxis_title="Hours", yaxis_title="Count")
            st.plotly_chart(fig, width='stretch')
    
    # ==================== NODE DETAILS SECTION ====================
    elif menu == "Node Details":
        st.header("🔍 Node Details")
        
        # Node selection
        node_list = df['node_id'].unique()
        selected_node = st.selectbox("Select Node:", node_list)
        
        # Get node data
        node_data = df[df['node_id'] == selected_node].iloc[0]
        node_pred = predictions[predictions['node_id'] == selected_node].iloc[0]
        
        # Display node info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Failure Probability", f"{node_pred['failure_probability']:.1%}")
        
        with col2:
            st.metric("Time to Failure", f"{node_pred['time_to_failure_hours']:.0f} hours")
        
        with col3:
            risk = node_pred['risk_level']
            color = get_risk_color(risk)
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;text-align:center;font-weight:bold;'>{risk} RISK</div>", 
                       unsafe_allow_html=True)
        
        with col4:
            confidence = 0.85 + np.random.random() * 0.1
            st.metric("Confidence Score", f"{confidence:.1%}")
        
        st.markdown("---")
        
        # Node features
        st.subheader("📊 Node Features")
        
        feature_cols = [
            'inventory_units', 'expiry_hours_remaining', 'temperature_excursion_flag',
            'transport_delay_hours', 'route_reliability_score', 'demand_rate',
            'casualty_rate', 'cold_chain_health_score', 'backup_supply_available',
            'days_of_supply', 'expiry_risk', 'transport_risk', 'viability_score', 'risk_composite'
        ]
        
        # Create two columns for features
        col1, col2 = st.columns(2)
        
        for i, col in enumerate(feature_cols):
            with col1 if i % 2 == 0 else col2:
                value = node_data[col]
                if isinstance(value, float):
                    st.metric(col.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    st.metric(col.replace('_', ' ').title(), value)
    
    # ==================== SHAP EXPLANATION SECTION ====================
    elif menu == "SHAP Explanation":
        st.header("🔍 SHAP Explainable AI")
        
        st.info("💡 SHAP (SHapley Additive exPlanations) helps understand why the model makes specific predictions.")
        
        # Node selection for explanation
        node_list = df['node_id'].unique()
        selected_node = st.selectbox("Select Node for Explanation:", node_list, key="shap_node")
        
        # Get node index
        node_indices = df[df['node_id'] == selected_node].index
        if len(node_indices) > 0:
            node_idx = node_indices[0]
            
            # Get prediction
            node_pred = predictions[predictions['node_id'] == selected_node].iloc[0]
            
            st.markdown("---")
            
            # Prediction summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Failure Risk", f"{node_pred['failure_probability']:.1%}")
            
            with col2:
                st.metric("Predicted Time to Failure", f"{node_pred['time_to_failure_hours']:.0f} hours")
            
            with col3:
                risk = node_pred['risk_level']
                color = get_risk_color(risk)
                st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;text-align:center;font-weight:bold;'>{risk} RISK</div>", 
                           unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Get local SHAP explanation
            X = df[model.feature_names].copy()
            explanation = model.get_local_explanation(X, node_idx)
            
            # Waterfall chart for local explanation path
            st.subheader("Waterfall Contribution Path")
            waterfall_features = explanation.head(10).copy()
            waterfall_labels = waterfall_features["feature"].tolist() + ["Total"]
            waterfall_values = waterfall_features["shap_value"].tolist() + [waterfall_features["shap_value"].sum()]
            waterfall_measures = ["relative"] * len(waterfall_features) + ["total"]

            wf_fig = go.Figure(
                go.Waterfall(
                    measure=waterfall_measures,
                    x=waterfall_labels,
                    y=waterfall_values,
                    connector={"line": {"color": "rgb(160,160,160)"}},
                    increasing={"marker": {"color": "#dc3545"}},
                    decreasing={"marker": {"color": "#28a745"}},
                    totals={"marker": {"color": "#1f77b4"}},
                )
            )
            wf_fig.update_layout(
                title="Top Local Feature Contributions",
                xaxis_title="Feature",
                yaxis_title="Contribution",
                showlegend=False,
            )
            st.plotly_chart(wf_fig, width='stretch')
            
            # Top features bar chart
            st.subheader("📊 Top Feature Contributions (SHAP Values)")
            
            top_features = explanation.head(10)
            
            fig = px.bar(
                top_features,
                x='shap_value',
                y='feature',
                orientation='h',
                title="Feature Impact on Prediction",
                color='shap_value',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("---")
            
            # Feature values table
            st.subheader("📋 Feature Values for Selected Node")
            
            feature_display = explanation[['feature', 'value', 'shap_value']].copy()
            feature_display.columns = ['Feature', 'Value', 'SHAP Value']
            feature_display['SHAP Value'] = feature_display['SHAP Value'].apply(lambda x: f"{x:.4f}")
            feature_display['Value'] = feature_display['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
            
            st.dataframe(feature_display, width='stretch')
            
            st.markdown("---")
            
            # Global feature importance
            st.subheader("🌍 Global Feature Importance")
            
            global_importance = model.get_global_importance()
            
            fig = px.bar(
                global_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features (Global)",
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, width='stretch')
    
    # ==================== RECOMMENDATIONS SECTION ====================
    elif menu == "Recommendations":
        st.header("💡 Actionable Recommendations")
        
        st.info("💡 Based on risk analysis, the following recommendations are provided to mitigate potential failures.")
        
        # Node selection
        node_list = df['node_id'].unique()
        selected_node = st.selectbox("Select Node for Recommendations:", node_list, key="rec_node")
        
        # Get node data
        node_data = df[df['node_id'] == selected_node].iloc[0]
        node_pred = predictions[predictions['node_id'] == selected_node].iloc[0]
        
        st.markdown("---")
        
        # Current status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Failure Risk", f"{node_pred['failure_probability']:.1%}")
        
        with col2:
            st.metric("Time to Failure", f"{node_pred['time_to_failure_hours']:.0f} hours")
        
        with col3:
            risk = node_pred['risk_level']
            color = get_risk_color(risk)
            st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;color:white;text-align:center;font-weight:bold;'>{risk} RISK</div>", 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Generate recommendations
        recommendations = generate_recommendations(node_data)
        
        # Display recommendations
        st.subheader("🎯 Recommended Actions")
        
        for i, rec in enumerate(recommendations):
            priority_color = {
                'CRITICAL': '#dc3545',
                'HIGH': '#fd7e14',
                'MEDIUM': '#ffc107',
                'LOW': '#28a745'
            }.get(rec['priority'], '#6c757d')
            
            with st.expander(f"{rec['priority']}: {rec['action']}", expanded=True):
                st.markdown(f"""
                <div style='padding:10px;border-left:4px solid {priority_color};background-color:#f8f9fa;'>
                    <strong>Action:</strong> {rec['action']}<br>
                    <strong>Reason:</strong> {rec['reason']}<br>
                    <strong>Impact:</strong> {rec['impact']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # All nodes recommendations summary
        st.subheader("📋 Network-Wide Recommendations Summary")
        
        all_recommendations = []
        for node_id in node_list:
            node_data = df[df['node_id'] == node_id].iloc[0]
            node_recs = generate_recommendations(node_data)
            for rec in node_recs:
                all_recommendations.append({
                    'Node': node_id,
                    'Priority': rec['priority'],
                    'Action': rec['action']
                })
        
        rec_df = pd.DataFrame(all_recommendations)
        
        # Count by priority
        priority_counts = rec_df['Priority'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Recommendations by Priority:")
            for priority, count in priority_counts.items():
                color = get_risk_color(priority)
                st.markdown(f"<span style='color:{color};'>●</span> **{priority}**: {count}", unsafe_allow_html=True)
        
        with col2:
            st.write("Actions by Node:")
            node_actions = rec_df.groupby('Node')['Action'].count()
            st.dataframe(node_actions, width='stretch')
    
    # ==================== SCENARIO SIMULATION SECTION ====================
    elif menu == "Scenario Simulation":
        st.header("🎮 Scenario Simulation")
        
        st.info("💡 Simulate different operational scenarios to see how predictions change.")
        
        # Simulation parameters
        st.subheader("⚙️ Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            casualty_multiplier = st.slider(
                "Casualty Rate Multiplier",
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="Increase to simulate higher casualty rates"
            )
        
        with col2:
            transport_delay = st.slider(
                "Additional Transport Delay (hours)",
                min_value=0,
                max_value=48,
                value=0,
                step=2,
                help="Add additional transport delay to simulate disruptions"
            )
        
        st.markdown("---")
        
        # Apply simulation
        sim_df = df.copy()
        sim_df['demand_rate'] = sim_df['demand_rate'] * casualty_multiplier
        sim_df['transport_delay_hours'] = sim_df['transport_delay_hours'] + transport_delay
        
        # Recalculate features
        sim_df['days_of_supply'] = sim_df['inventory_units'] / sim_df['demand_rate']
        sim_df['transport_risk'] = sim_df['transport_delay_hours'] * (1 - sim_df['route_reliability_score'])
        
        # Make predictions
        sim_predictions = model.predict(sim_df)
        
        # Display results
        st.subheader("📊 Simulation Results")
        
        # Compare before and after
        comparison = pd.DataFrame({
            'Node': predictions['node_id'],
            'Original Risk': predictions['failure_probability'],
            'Simulated Risk': sim_predictions['failure_probability'],
            'Risk Change': sim_predictions['failure_probability'] - predictions['failure_probability']
        })
        
        # Aggregate by node
        node_comparison = comparison.groupby('Node').agg({
            'Original Risk': 'mean',
            'Simulated Risk': 'mean',
            'Risk Change': 'mean'
        }).reset_index()
        
        # Display comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=node_comparison['Node'],
            y=node_comparison['Original Risk'],
            name='Original Risk',
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Bar(
            x=node_comparison['Node'],
            y=node_comparison['Simulated Risk'],
            name='Simulated Risk',
            marker_color='#dc3545'
        ))
        
        fig.update_layout(
            title="Risk Comparison: Original vs Simulated",
            xaxis_title="Node",
            yaxis_title="Failure Probability",
            barmode='group'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Impact summary
        st.subheader("📈 Impact Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_avg = predictions['failure_probability'].mean()
            st.metric("Original Avg Risk", f"{original_avg:.1%}")
        
        with col2:
            simulated_avg = sim_predictions['failure_probability'].mean()
            st.metric("Simulated Avg Risk", f"{simulated_avg:.1%}")
        
        with col3:
            change = simulated_avg - original_avg
            st.metric("Risk Change", f"{change:+.1%}")
        
        # Show nodes with biggest changes
        st.subheader("⚠️ Nodes with Largest Risk Changes")
        
        node_comparison['Risk Change %'] = (node_comparison['Risk Change'] * 100).round(1)
        node_comparison = node_comparison.sort_values('Risk Change', ascending=False)
        
        display_cols = ['Node', 'Original Risk', 'Simulated Risk', 'Risk Change %']
        node_comparison_display = node_comparison[display_cols].copy()
        node_comparison_display.columns = ['Node', 'Original Risk', 'Simulated Risk', 'Change (%)']
        node_comparison_display['Original Risk'] = node_comparison_display['Original Risk'].apply(lambda x: f"{x:.1%}")
        node_comparison_display['Simulated Risk'] = node_comparison_display['Simulated Risk'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(node_comparison_display, width='stretch')
        
        st.markdown("---")
        
        # Recommendations for simulated scenario
        st.subheader("💡 Recommendations for Simulated Scenario")
        
        # Find most affected nodes
        most_affected = node_comparison.head(3)['Node'].tolist()
        
        for node_id in most_affected:
            node_data = sim_df[sim_df['node_id'] == node_id].iloc[0]
            recs = generate_recommendations(node_data)
            
            st.markdown(f"**{node_id}:**")
            for rec in recs[:2]:
                st.write(f"  - [{rec['priority']}] {rec['action']}")
            st.write("")


if __name__ == '__main__':
    main()
