import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from scipy import stats
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def get_feature_group(feature_name):
    """Determine the group of a feature based on its suffix"""
    if feature_name == 'Y':
        return 'TARGET'
    elif feature_name.endswith('_1'):
        return 'GROUP_1'
    elif feature_name.endswith('_2'):
        return 'GROUP_2'
    elif feature_name.endswith('_3'):
        return 'GROUP_3'
    else:
        return 'EXTRA'

def create_bar_plot(data, group_name, output_path):
    """Create bar plot with optimized memory usage"""
    plt.clf()  # Clear the current figure
    fig = Figure(figsize=(15, max(8, len(data) * 0.3)))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Create bar plot
    y_pos = np.arange(len(data))
    ax.barh(y_pos, data['Weight_Percentage'], color='steelblue')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['Feature'], fontsize=8)
    ax.set_xlabel('Weight Percentage (%)')
    ax.set_title(f'Feature Importance Weights (%) - {group_name}')
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_correlation_heatmap(data, output_path):
    """Create correlation heatmap with optimized memory usage"""
    plt.clf()  # Clear the current figure
    fig = Figure(figsize=(20, 16))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Create heatmap
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Feature Correlation Heatmap (Grouped by Feature Type)')
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def analyze_features(data_path):
    """
    Analyze feature relationships with Y target variable with normalized weights.
    Features are organized by their suffix groups (_1, _2, _3, EXTRA).
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Remove rows where all values are 0
    data = data[(data != 0).any(axis=1)]
    
    # Calculate correlations with Y
    correlations = data.corr()['Y'].sort_values(ascending=False)
    correlations = correlations.drop('Y')
    
    # Get number of features
    n_features = len(correlations)
    
    # Normalize correlations to sum to 1 (convert to weights)
    abs_correlations = np.abs(correlations)
    weights = abs_correlations / abs_correlations.sum()
    
    # Create feature importance table with group information
    feature_table = pd.DataFrame({
        'Feature': correlations.index,
        'Group': [get_feature_group(feat) for feat in correlations.index],
        'Correlation': correlations.values,
        'Absolute_Correlation': abs_correlations,
        'Normalized_Weight': weights.values,
        'Weight_Percentage': (weights.values * 100).round(2)
    })
    
    # Sort by group and then by absolute correlation within each group
    feature_table = feature_table.sort_values(['Group', 'Absolute_Correlation'], ascending=[True, False])
    
    # 1. Save complete correlation table (grouped)
    feature_table.to_csv('plots/feature_correlations_grouped.csv', index=False)
    
    # 2. Create and save correlation heatmap (grouped)
    grouped_features = feature_table['Feature'].tolist()
    grouped_data = data[grouped_features + ['Y']]
    create_correlation_heatmap(grouped_data, 'plots/correlation_heatmap_grouped.png')
    
    # 3. Save feature importance scores (grouped)
    feature_table.to_csv('plots/feature_importance_grouped.csv', index=False)
    
    # 4. Create feature importance bar plot for all groups in one file
    plt.clf()  # Clear the current figure
    fig = Figure(figsize=(15, max(8, len(feature_table) * 0.3)))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Create bar plot with color related to weight percentage
    y_pos = np.arange(len(feature_table))
    colors = plt.cm.viridis(feature_table['Weight_Percentage'] / feature_table['Weight_Percentage'].max())
    bar_plot = ax.barh(y_pos, feature_table['Weight_Percentage'], color=colors)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_table['Feature'], fontsize=8)
    ax.set_xlabel('Weight Percentage (%)')
    ax.set_title('Feature Importance Weights (%) - All Groups')
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=feature_table['Weight_Percentage'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Weight Percentage (%)')
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig('plots/feature_importance_bar_all_groups.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # 5. Create interactive radar plot for all groups in one file
    fig = go.Figure()
    for group in ['GROUP_1', 'GROUP_2', 'GROUP_3', 'EXTRA']:
        group_data = feature_table[feature_table['Group'] == group]
        if not group_data.empty:
            fig.add_trace(go.Scatterpolar(
                r=group_data['Weight_Percentage'],
                theta=group_data['Feature'],
                fill='toself',
                name=f'{group} Features'
            ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(feature_table['Weight_Percentage'])])),
        showlegend=True,
        title='Feature Importance Weights (%) - All Groups'
    )
    fig.write_html('plots/feature_radar_all_groups.html')
    # 7. Create comprehensive feature scoring (grouped)
    feature_scores = pd.DataFrame({
        'Feature': correlations.index,
        'Group': [get_feature_group(feat) for feat in correlations.index],
        'Correlation': correlations.values,
        'Weight_Percentage': (weights.values * 100).round(2),
        'Abs_Correlation_Rank': abs_correlations.rank(ascending=False),
        'Statistical_Significance': [stats.pearsonr(data[feat], data['Y'])[1] for feat in correlations.index]
    }).sort_values(['Group', 'Abs_Correlation_Rank'])
    
    feature_scores.to_csv('plots/feature_scores_grouped.csv', index=False)
    
    # 8. Create detailed feature summary (grouped)
    feature_summary = pd.DataFrame({
        'Feature': correlations.index,
        'Group': [get_feature_group(feat) for feat in correlations.index],
        'Mean': [data[feat].mean() for feat in correlations.index],
        'Std': [data[feat].std() for feat in correlations.index],
        'Weight_Percentage': (weights.values * 100).round(2),
        'Correlation': correlations.values
    }).sort_values(['Group', 'Correlation'], ascending=[True, False])
    
    feature_summary.to_csv('plots/feature_summary_grouped.csv', index=False)
    
    # 9. Create styled HTML summary table for all groups
    styled_summary = feature_summary.style\
        .background_gradient(subset=['Weight_Percentage'], cmap='viridis')\
        .background_gradient(subset=['Correlation'], cmap='coolwarm')\
        .format({
            'Mean': '{:.2f}',
            'Std': '{:.2f}',
            'Weight_Percentage': '{:.2f}%',
            'Correlation': '{:.3f}'
        })
    styled_summary.to_html('plots/feature_summary_styled_all_groups.html')
    # Print summary of results
    print("\nAnalysis completed successfully!")
    print(f"\nTotal number of features analyzed: {n_features}")
    
    # Print group-wise statistics
    for group in ['GROUP_1', 'GROUP_2', 'GROUP_3', 'EXTRA']:
        group_features = feature_table[feature_table['Group'] == group]
        print(f"\n{group} Statistics:")
        print(f"Number of features: {len(group_features)}")
        print(f"Average correlation: {group_features['Correlation'].mean():.3f}")
        print(f"Total weight percentage: {group_features['Weight_Percentage'].sum():.2f}%")
    
    print("\nAnalysis files have been saved in the 'plots' directory:")
    print("1. feature_correlations_grouped.csv - Complete correlation table")
    print("2. correlation_heatmap_grouped.png - Correlation heatmap visualization")
    print("3. feature_importance_grouped.csv - Feature importance scores")
    print("4. feature_importance_bar_[GROUP].png - Bar plots for each group")
    print("5. feature_radar_[GROUP].html - Interactive radar plots for each group")
    print("7. feature_scores_grouped.csv - Comprehensive feature scoring table")
    print("8. feature_summary_grouped.csv - Detailed feature summary table")
    print("9. feature_summary_styled_[GROUP].html - Styled HTML summary tables for each group")
    
    return feature_table

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    feature_analysis = analyze_features(data_path)