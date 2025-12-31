#  Create a comprehensive drowsiness detection system implementation
import pandas as pd
import numpy as np

# First, let's create the system requirements and dependencies
requirements_data = {
    'Library': ['OpenCV (cv2)', 'dlib', 'imutils', 'scipy', 'numpy', 'threading', 'pygame', 'argparse', 'time'],
    'Version': ['4.5.5+', '19.22+', '0.5.4+', '1.7.0+', '1.21.0+', 'Built-in', '2.1.0+', 'Built-in', 'Built-in'],
    'Purpose': [
        'Computer vision and image processing',
        'Facial landmark detection and face recognition',
        'Convenience functions for OpenCV operations',
        'Euclidean distance calculations for EAR',
        'Numerical computing and array operations',
        'Multi-threading for real-time processing',
        'Audio alarm system for drowsiness alerts',
        'Command-line argument parsing',
        'Time-based operations and delays'
    ],
    'Installation': [
        'pip install opencv-python',
        'pip install dlib',
        'pip install imutils',
        'pip install scipy',
        'pip install numpy',
        'Standard library',
        'pip install pygame',
        'Standard library',
        'Standard library'
    ]
}

requirements_df = pd.DataFrame(requirements_data)
print("=== DROWSINESS DETECTION SYSTEM REQUIREMENTS ===")
print(requirements_df.to_string(index=False))
print("\n")

# Hardware requirements
hardware_data = {
    'Component': ['Webcam/Camera', 'Processor', 'RAM', 'Storage', 'Audio Output'],
    'Minimum Specification': [
        'USB webcam, 2MP, 30fps',
        'Intel i3 or equivalent',
        '4GB RAM',
        '2GB free space',
        'Speakers/Headphones'
    ],
    'Recommended Specification': [
        'HD webcam, 5MP+, 60fps',
        'Intel i5 or equivalent',
        '8GB+ RAM',
        '5GB+ free space',
        'External speakers'
    ],
    'Purpose': [
        'Real-time video capture of driver face',
        'Real-time image processing and ML inference',
        'Buffer video frames and model operations',
        'Store models, datasets, and logs',
        'Alert system for drowsiness warnings'
    ]
}

hardware_df = pd.DataFrame(hardware_data)
print("=== HARDWARE REQUIREMENTS ===")
print(hardware_df.to_string(index=False))
print("\n")

# Create sample EAR data for different scenarios
np.random.seed(42)
frames = list(range(0, 300))

# Normal awake state with occasional blinks
normal_ear = []
for i in frames:
    if i in [30, 35, 80, 85, 150, 155, 220, 225]:  # Blink frames
        normal_ear.append(np.random.uniform(0.02, 0.08))  # Closed eyes
    else:
        normal_ear.append(np.random.uniform(0.25, 0.35))  # Open eyes

# Drowsy state - prolonged low EAR values
drowsy_ear = []
for i in frames:
    if i < 50:
        drowsy_ear.append(np.random.uniform(0.25, 0.35))  # Initially awake
    elif i < 100:
        drowsy_ear.append(np.random.uniform(0.15, 0.25))  # Getting drowsy
    else:
        drowsy_ear.append(np.random.uniform(0.05, 0.15))  # Drowsy state

# Create comprehensive dataset
drowsiness_data = {
    'Frame_Number': frames + frames,  # Two scenarios
    'EAR_Value': normal_ear + drowsy_ear,
    'State': ['Normal'] * len(frames) + ['Drowsy'] * len(frames),
    'Alert_Triggered': [0] * len(frames) + [1 if ear < 0.3 else 0 for ear in drowsy_ear]
}

drowsiness_df = pd.DataFrame(drowsiness_data)
print("=== SAMPLE EAR DATA OVERVIEW ===")
print(f"Total data points: {len(drowsiness_df)}")
print(f"Normal state samples: {len(drowsiness_df[drowsiness_df['State'] == 'Normal'])}")
print(f"Drowsy state samples: {len(drowsiness_df[drowsiness_df['State'] == 'Drowsy'])}")
print(f"Alerts triggered: {drowsiness_df['Alert_Triggered'].sum()}")
print("\nSample data:")
print(drowsiness_df.head(10).to_string(index=False))
print("\n")

# Save the comprehensive dataset
drowsiness_df.to_csv('drowsiness_detection_sample_data.csv', index=False)
requirements_df.to_csv('system_requirements.csv', index=False)
hardware_df.to_csv('hardware_requirements.csv', index=False)

print("=== FILES CREATED ===")
print("1. drowsiness_detection_sample_data.csv - Sample EAR data for testing")
print("2. system_requirements.csv - Software dependencies and requirements")
print("3. hardware_requirements.csv - Hardware specifications and requirements")