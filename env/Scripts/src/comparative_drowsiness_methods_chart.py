import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Data from the provided JSON
data = {
    "methods": ["EAR-based", "CNN Deep Learning", "Traditional HOG+SVM", "Transformer (ViT)", "Multimodal EEG+Vision"],
    "accuracy": [92, 86, 78, 99, 98],
    "real_time_performance": [95, 70, 85, 60, 40],
    "hardware_requirements": [90, 60, 80, 45, 30],
    "implementation_complexity": [85, 50, 70, 40, 25],
    "cost_effectiveness": [95, 65, 80, 50, 35]
}

# Create DataFrame
df = pd.DataFrame(data)

# Abbreviate method names to fit 15 character limit
method_abbreviations = ["EAR-based", "CNN DL", "HOG+SVM", "ViT", "Multimodal"]

# Create the grouped bar chart
fig = go.Figure()

# Colors for each metric (using the 5 primary brand colors)
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']
metric_names = ['Accuracy', 'Real-time', 'Hardware Req', 'Complexity', 'Cost Effect']
metric_keys = ['accuracy', 'real_time_performance', 'hardware_requirements', 'implementation_complexity', 'cost_effectiveness']

# Add bars for each metric
for i, (metric_key, metric_name, color) in enumerate(zip(metric_keys, metric_names, colors)):
    fig.add_trace(go.Bar(
        name=metric_name,
        x=method_abbreviations,
        y=df[metric_key],
        marker_color=color,
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title='Drowsiness Detection Methods Comparison',
    xaxis_title='Method',
    yaxis_title='Score (0-100)',
    barmode='group',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Update traces for better visual appearance
fig.update_traces(cliponaxis=False)

# Update y-axis to show full range
fig.update_yaxes(range=[0, 100])

# Save the chart as both PNG and SVG
fig.write_image("drowsiness_comparison.png")
fig.write_image("drowsiness_comparison.svg", format="svg")

fig.show()