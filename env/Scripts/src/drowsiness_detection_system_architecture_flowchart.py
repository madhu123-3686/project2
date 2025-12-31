import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Define flowchart components with better spacing
components = [
    # Row 1
    {'name': 'Camera Input\nWebcam Video', 'x': 4, 'y': 10, 'type': 'input', 'color': '#1FB8CD'},
    
    # Row 2  
    {'name': 'Frame Process\nGrayscale', 'x': 4, 'y': 9, 'type': 'process', 'color': '#2E8B57'},
    
    # Row 3
    {'name': 'Face Detection\ndlib HOG', 'x': 4, 'y': 8, 'type': 'process', 'color': '#2E8B57'},
    
    # Row 4
    {'name': 'Facial Landmarks\n68-point detect', 'x': 4, 'y': 7, 'type': 'process', 'color': '#2E8B57'},
    
    # Row 5
    {'name': 'Eye Extract\nPoints 36-47', 'x': 4, 'y': 6, 'type': 'process', 'color': '#2E8B57'},
    
    # Row 6
    {'name': 'EAR Calc\nAspect Ratio', 'x': 4, 'y': 5, 'type': 'process', 'color': '#2E8B57'},
    
    # Row 7 - Decision
    {'name': 'EAR < 0.3?', 'x': 4, 'y': 4, 'type': 'decision', 'color': '#D2BA4C'},
    
    # Row 8 - Counter (right branch)
    {'name': 'Frame Counter\nCount Frames', 'x': 7, 'y': 3, 'type': 'process', 'color': '#2E8B57'},
    
    # Row 9 - Second decision
    {'name': 'Count >= 48?', 'x': 7, 'y': 2, 'type': 'decision', 'color': '#D2BA4C'},
    
    # Row 10 - Alert
    {'name': 'Alert System\nSound Alarm', 'x': 7, 'y': 1, 'type': 'output', 'color': '#DB4545'}
]

# Add rectangular background for each node
for i, comp in enumerate(components):
    # Choose shape based on type
    if comp['type'] == 'decision':
        # Diamond for decisions
        x_coords = [comp['x']-0.7, comp['x'], comp['x']+0.7, comp['x'], comp['x']-0.7]
        y_coords = [comp['y'], comp['y']+0.35, comp['y'], comp['y']-0.35, comp['y']]
    elif comp['type'] in ['input', 'output']:
        # Rounded rectangle for input/output  
        x_coords = [comp['x']-0.8, comp['x']+0.8, comp['x']+0.8, comp['x']-0.8, comp['x']-0.8]
        y_coords = [comp['y']-0.25, comp['y']-0.25, comp['y']+0.25, comp['y']+0.25, comp['y']-0.25]
    else:
        # Rectangle for processes
        x_coords = [comp['x']-0.8, comp['x']+0.8, comp['x']+0.8, comp['x']-0.8, comp['x']-0.8]
        y_coords = [comp['y']-0.25, comp['y']-0.25, comp['y']+0.25, comp['y']+0.25, comp['y']-0.25]
    
    # Add shape
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        fill='toself',
        fillcolor=comp['color'],
        line=dict(color='white', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add text
    fig.add_trace(go.Scatter(
        x=[comp['x']],
        y=[comp['y']],
        mode='text',
        text=comp['name'],
        textfont=dict(size=12, color='white'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add arrows for main flow
arrows = [
    # Main vertical flow
    {'x': 4, 'y': 9.75, 'ax': 4, 'ay': 10.25},  # Camera to Frame
    {'x': 4, 'y': 8.75, 'ax': 4, 'ay': 9.25},   # Frame to Face  
    {'x': 4, 'y': 7.75, 'ax': 4, 'ay': 8.25},   # Face to Landmarks
    {'x': 4, 'y': 6.75, 'ax': 4, 'ay': 7.25},   # Landmarks to Eye
    {'x': 4, 'y': 5.75, 'ax': 4, 'ay': 6.25},   # Eye to EAR
    {'x': 4, 'y': 4.75, 'ax': 4, 'ay': 5.25},   # EAR to Decision
    
    # Yes path to counter
    {'x': 6.2, 'y': 3.2, 'ax': 4.7, 'ay': 3.8},  # Decision Yes to Counter
    {'x': 7, 'y': 2.75, 'ax': 7, 'ay': 3.25},    # Counter to Alarm Decision
    
    # Yes path to alert
    {'x': 7, 'y': 1.75, 'ax': 7, 'ay': 2.25},    # Alarm Yes to Alert
    
    # No paths back to frame processing
    {'x': 1.5, 'y': 9, 'ax': 3.2, 'ay': 3.8},    # EAR No back to Frame
    {'x': 1.5, 'y': 9.2, 'ax': 6.2, 'ay': 1.8}   # Alarm No back to Frame
]

# Add all arrows
for arrow in arrows:
    fig.add_annotation(
        x=arrow['x'], y=arrow['y'],
        ax=arrow['ax'], ay=arrow['ay'],
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=3,
        arrowsize=2,
        arrowwidth=3,
        arrowcolor='#333333'
    )

# Add decision labels
fig.add_annotation(x=5.5, y=3.5, text="YES", showarrow=False, 
                  font=dict(size=14, color='#333333', family='Arial Black'))
fig.add_annotation(x=2.5, y=3.5, text="NO", showarrow=False, 
                  font=dict(size=14, color='#333333', family='Arial Black'))
fig.add_annotation(x=7.8, y=1.5, text="YES", showarrow=False, 
                  font=dict(size=14, color='#333333', family='Arial Black'))
fig.add_annotation(x=5, y=1.5, text="NO", showarrow=False, 
                  font=dict(size=14, color='#333333', family='Arial Black'))

# Update layout
fig.update_layout(
    title="Drowsiness Detection System",
    xaxis=dict(range=[0, 8.5], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(range=[0, 11], showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=16)
)

# Save the chart as both PNG and SVG
fig.write_image("drowsiness_detection_flowchart.png", width=1000, height=1200)
fig.write_image("drowsiness_detection_flowchart.svg", format="svg", width=1000, height=1200)

print("Flowchart created successfully!")
fig.show()