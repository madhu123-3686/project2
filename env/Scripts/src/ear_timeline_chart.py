import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {
    "frame_numbers": [0, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 135, 140, 145, 150, 160, 170, 180, 190, 200], 
    "ear_values": [0.32, 0.31, 0.33, 0.30, 0.05, 0.02, 0.28, 0.32, 0.31, 0.33, 0.29, 0.25, 0.05, 0.03, 0.26, 0.31, 0.28, 0.22, 0.08, 0.02, 0.15, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03], 
    "threshold": 0.3, 
    "annotations": [{"frame": 35, "label": "Normal Blink"}, {"frame": 95, "label": "Normal Blink"}, {"frame": 140, "label": "Drowsiness Detected"}, {"frame": 180, "label": "Deep Drowsiness"}]
}

fig = go.Figure()

# Add the main EAR line
fig.add_trace(go.Scatter(
    x=data["frame_numbers"], 
    y=data["ear_values"],
    mode='lines+markers',
    name='EAR Values',
    line=dict(color='#1FB8CD', width=2),
    marker=dict(size=4, color='#1FB8CD')
))

# Add threshold line
fig.add_trace(go.Scatter(
    x=[0, 200],
    y=[data["threshold"], data["threshold"]],
    mode='lines',
    name='Drowsy Threshold',
    line=dict(color='#DB4545', dash='dash', width=2),
))

# Add markers for key events with different colors
normal_blink_frames = [35, 95]
drowsy_frames = [140, 180]

# Normal blinks
for frame in normal_blink_frames:
    idx = data["frame_numbers"].index(frame)
    fig.add_trace(go.Scatter(
        x=[frame],
        y=[data["ear_values"][idx]],
        mode='markers',
        name='Normal Blink',
        marker=dict(size=8, color='#2E8B57', symbol='circle'),
        showlegend=True if frame == normal_blink_frames[0] else False
    ))

# Drowsiness markers
for frame in drowsy_frames:
    idx = data["frame_numbers"].index(frame)
    fig.add_trace(go.Scatter(
        x=[frame],
        y=[data["ear_values"][idx]],
        mode='markers',
        name='Drowsiness',
        marker=dict(size=8, color='#D2BA4C', symbol='diamond'),
        showlegend=True if frame == drowsy_frames[0] else False
    ))

# Update layout
fig.update_layout(
    title='EAR Values Over Time',
    xaxis_title='Frame Number',
    yaxis_title='EAR Value',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_xaxes(range=[0, 200])
fig.update_yaxes(range=[0, 0.4])

# Update traces for better visibility
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("ear_chart.png")
fig.write_image("ear_chart.svg", format="svg")