# Create additional utility files for the drowsiness detection project

# 1. Setup and installation script
setup_script = '''#!/usr/bin/env python3
"""
Drowsiness Detection System Setup Script
Automatically installs dependencies and downloads required models
"""

import subprocess
import sys
import os
import urllib.request
import bz2
import shutil

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def download_file(url, filename):
    """Download a file from URL"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def extract_bz2(source, target):
    """Extract bz2 compressed file"""
    try:
        with bz2.BZ2File(source, 'rb') as f_in:
            with open(target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)  # Remove compressed file
        print(f"✓ Extracted {target}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract {source}: {e}")
        return False

def main():
    print("=== Drowsiness Detection System Setup ===")
    print("This script will install all required dependencies and models.\\n")
    
    # Required packages
    packages = [
        'opencv-python',
        'dlib',
        'imutils', 
        'scipy',
        'numpy',
        'pygame'
    ]
    
    print("Installing Python packages...")
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("\\n✓ All packages installed successfully!")
    
    # Download facial landmark predictor
    print("\\nDownloading facial landmark predictor...")
    predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    predictor_file = "shape_predictor_68_face_landmarks.dat.bz2"
    predictor_extracted = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_extracted):
        if download_file(predictor_url, predictor_file):
            extract_bz2(predictor_file, predictor_extracted)
    else:
        print(f"✓ {predictor_extracted} already exists")
    
    # Create sample alarm sound (simple beep)
    print("\\nCreating sample alarm sound...")
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Generate a simple beep sound
        import numpy as np
        duration = 1.0  # seconds
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 800  # Hz
        wave = np.sin(frequency * 2 * np.pi * t)
        
        # Convert to pygame sound format
        sound = np.array([wave, wave]).T
        sound = (sound * 32767).astype(np.int16)
        
        sound_surface = pygame.sndarray.make_sound(sound)
        pygame.mixer.Sound.stop(sound_surface)
        
        # Save as WAV file
        pygame.mixer.music.stop()
        pygame.mixer.init()
        pygame.mixer.Sound(sound_surface)
        
        print("✓ Sample alarm sound created")
        
    except Exception as e:
        print(f"⚠️  Could not create alarm sound: {e}")
        print("You can use any .wav file as an alarm sound")
    
    print("\\n=== Setup Complete ===")
    print("\\nTo run the drowsiness detection system:")
    print("python drowsiness_detection_system.py -p shape_predictor_68_face_landmarks.dat")
    print("\\nOptional parameters:")
    print("  -a alarm.wav          # Path to alarm sound file")
    print("  -w 0                  # Webcam index (default: 0)")
    print("  -t 0.3                # EAR threshold (default: 0.3)")
    print("  -f 48                 # Frame count threshold (default: 48)")

if __name__ == "__main__":
    main()
'''

# 2. Data analysis script
analysis_script = '''#!/usr/bin/env python3
"""
Drowsiness Detection Data Analysis Script
Analyzes session data and generates reports
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

class DrowsinessAnalyzer:
    def __init__(self):
        self.session_files = []
        self.combined_data = []
    
    def load_session_files(self, pattern="drowsiness_session_*.json"):
        """Load all session files matching the pattern"""
        self.session_files = glob.glob(pattern)
        print(f"Found {len(self.session_files)} session files")
        
        for file in self.session_files:
            try:
                with open(file, 'r') as f:
                    session_data = json.load(f)
                    self.combined_data.extend(session_data['detection_events'])
                print(f"✓ Loaded {file}")
            except Exception as e:
                print(f"✗ Failed to load {file}: {e}")
    
    def analyze_sessions(self):
        """Perform comprehensive analysis of session data"""
        if not self.combined_data:
            print("No data to analyze. Run load_session_files() first.")
            return
        
        df = pd.DataFrame(self.combined_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print("\\n=== DROWSINESS DETECTION ANALYSIS ===")
        print(f"Total detection events: {len(df)}")
        print(f"Total alerts triggered: {df['alert_triggered'].sum()}")
        print(f"Alert rate: {df['alert_triggered'].mean():.2%}")
        
        print("\\n=== EAR STATISTICS ===")
        print(f"Mean EAR: {df['ear_value'].mean():.3f}")
        print(f"Median EAR: {df['ear_value'].median():.3f}")
        print(f"Min EAR: {df['ear_value'].min():.3f}")
        print(f"Max EAR: {df['ear_value'].max():.3f}")
        print(f"Std EAR: {df['ear_value'].std():.3f}")
        
        # EAR distribution by alert status
        alert_ear = df[df['alert_triggered']]['ear_value']
        normal_ear = df[~df['alert_triggered']]['ear_value']
        
        print("\\n=== EAR BY ALERT STATUS ===")
        print(f"Alert EAR - Mean: {alert_ear.mean():.3f}, Std: {alert_ear.std():.3f}")
        print(f"Normal EAR - Mean: {normal_ear.mean():.3f}, Std: {normal_ear.std():.3f}")
        
        return df
    
    def generate_plots(self, df):
        """Generate analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: EAR over time
        axes[0, 0].plot(df.index, df['ear_value'], alpha=0.7)
        axes[0, 0].axhline(y=0.3, color='r', linestyle='--', label='Threshold (0.3)')
        axes[0, 0].set_title('Eye Aspect Ratio Over Time')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('EAR Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: EAR distribution
        axes[0, 1].hist(df['ear_value'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0.3, color='r', linestyle='--', label='Threshold (0.3)')
        axes[0, 1].set_title('EAR Value Distribution')
        axes[0, 1].set_xlabel('EAR Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Alert events over time
        alert_frames = df[df['alert_triggered']].index
        axes[1, 0].scatter(alert_frames, [1] * len(alert_frames), 
                          c='red', alpha=0.6, s=10)
        axes[1, 0].set_title('Alert Events Over Time')
        axes[1, 0].set_xlabel('Frame Number')
        axes[1, 0].set_ylabel('Alert Status')
        axes[1, 0].set_ylim(0.5, 1.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Consecutive frames distribution
        axes[1, 1].hist(df['consecutive_frames'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Consecutive Frames Distribution')
        axes[1, 1].set_xlabel('Consecutive Frames')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('drowsiness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\\n✓ Analysis plots saved as 'drowsiness_analysis.png'")
    
    def generate_report(self, df):
        """Generate comprehensive analysis report"""
        report = f"""
DROWSINESS DETECTION SYSTEM - ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SESSION SUMMARY ===
Total Sessions Analyzed: {len(self.session_files)}
Total Detection Events: {len(df)}
Total Alerts Triggered: {df['alert_triggered'].sum()}
Alert Rate: {df['alert_triggered'].mean():.2%}

=== EAR STATISTICS ===
Mean EAR: {df['ear_value'].mean():.3f}
Median EAR: {df['ear_value'].median():.3f}
Standard Deviation: {df['ear_value'].std():.3f}
Minimum EAR: {df['ear_value'].min():.3f}
Maximum EAR: {df['ear_value'].max():.3f}

=== PERFORMANCE METRICS ===
- EAR values below threshold (0.3): {(df['ear_value'] < 0.3).sum()} events ({(df['ear_value'] < 0.3).mean():.1%})
- Average consecutive frames during alerts: {df[df['alert_triggered']]['consecutive_frames'].mean():.1f}
- Maximum consecutive frames: {df['consecutive_frames'].max()}

=== RECOMMENDATIONS ===
Based on the analysis:
1. Current EAR threshold (0.3) appears {'appropriate' if df['alert_triggered'].mean() < 0.1 else 'too sensitive'}
2. Frame count threshold should be {'maintained' if df[df['alert_triggered']]['consecutive_frames'].mean() > 40 else 'adjusted'}
3. System {'performed well' if df['alert_triggered'].mean() < 0.05 else 'may need calibration'}

=== DATA QUALITY ===
- No missing values: {df.isnull().sum().sum() == 0}
- EAR range check: All values between 0 and 1: {(df['ear_value'] >= 0).all() and (df['ear_value'] <= 1).all()}
- Timestamp continuity: {len(df)} events processed
        """
        
        with open('drowsiness_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("\\n✓ Analysis report saved as 'drowsiness_analysis_report.txt'")
        return report

def main():
    analyzer = DrowsinessAnalyzer()
    analyzer.load_session_files()
    
    if analyzer.combined_data:
        df = analyzer.analyze_sessions()
        analyzer.generate_plots(df)
        analyzer.generate_report(df)
    else:
        print("No session data found. Run the drowsiness detection system first.")

if __name__ == "__main__":
    main()
'''

# 3. Configuration file
config_content = '''# Drowsiness Detection System Configuration

[DEFAULT]
# EAR threshold for drowsiness detection (0.2-0.4 typical range)
ear_threshold = 0.3

# Number of consecutive frames below threshold to trigger alarm
consecutive_frames = 48

# Frame width for processing (smaller = faster, larger = more accurate)
frame_width = 450

# Webcam settings
webcam_index = 0
webcam_fps = 30

[PATHS]
# Path to dlib facial landmark predictor
shape_predictor = shape_predictor_68_face_landmarks.dat

# Path to alarm sound file (leave empty for no sound)
alarm_sound = alarm.wav

# Directory for session logs
log_directory = logs/

[DETECTION]
# Sensitivity settings
high_sensitivity_ear = 0.25
high_sensitivity_frames = 30

medium_sensitivity_ear = 0.3
medium_sensitivity_frames = 48

low_sensitivity_ear = 0.35  
low_sensitivity_frames = 60

# Enable/disable features
enable_audio_alert = true
enable_visual_alert = true
enable_data_logging = true
enable_realtime_plot = false

[DISPLAY]
# Display settings
show_eye_contours = true
show_ear_value = true
show_frame_counter = true
show_status_text = true

# Colors (BGR format)
alert_color = 0,0,255
normal_color = 0,255,0
eye_contour_color = 0,255,0
'''

# Save all utility files
with open('setup_drowsiness_detection.py', 'w',encoding='utf-8') as f:
    f.write(setup_script)

with open('analyze_drowsiness_data.py', 'w',encoding='utf-8') as f:
    f.write(analysis_script)

with open('config.ini', 'w',encoding='utf-8') as f:
    f.write(config_content)

# Create requirements.txt
requirements_txt = '''opencv-python>=4.5.5
dlib>=19.22.0
imutils>=0.5.4
scipy>=1.7.0
numpy>=1.21.0
pygame>=2.1.0
pandas>=1.3.0
matplotlib>=3.5.0
configparser>=5.2.0
'''

with open('requirements.txt', 'w',encoding='utf-8') as f:
    f.write(requirements_txt)

# Create README
readme_content = '''# Real-Time Driver Drowsiness Detection System

A comprehensive computer vision system that detects driver drowsiness in real-time using facial landmarks and Eye Aspect Ratio (EAR) analysis.

## Features

- ✅ Real-time face and eye detection using dlib
- ✅ Eye Aspect Ratio (EAR) calculation and monitoring  
- ✅ Configurable thresholds and sensitivity settings
- ✅ Audio and visual alerts for drowsiness detection
- ✅ Multi-threading for smooth real-time operation
- ✅ Comprehensive session data logging and analysis
- ✅ Command-line interface with multiple options
- ✅ Data visualization and performance analysis tools

## Quick Start

1. **Install dependencies:**
   ```bash
   python setup_drowsiness_detection.py
   ```

2. **Run the system:**
   ```bash
   python drowsiness_detection_system.py -p shape_predictor_68_face_landmarks.dat
   ```

3. **Analyze results:**
   ```bash
   python analyze_drowsiness_data.py
   ```

## System Requirements

### Hardware
- Webcam (minimum 2MP, recommended 5MP+)
- Intel i3+ processor (i5+ recommended)
- 4GB+ RAM (8GB+ recommended)
- Speakers or headphones for audio alerts

### Software
- Python 3.7+
- OpenCV 4.5.5+
- dlib 19.22+
- See `requirements.txt` for complete list

## Usage

### Basic Usage
```bash
python drowsiness_detection_system.py -p shape_predictor_68_face_landmarks.dat
```

### Advanced Usage
```bash
python drowsiness_detection_system.py \\
    -p shape_predictor_68_face_landmarks.dat \\
    -a alarm.wav \\
    -w 0 \\
    -t 0.3 \\
    -f 48
```

### Parameters
- `-p, --shape-predictor`: Path to facial landmark predictor (required)
- `-a, --alarm`: Path to alarm sound file (optional)
- `-w, --webcam`: Webcam index (default: 0)
- `-t, --threshold`: EAR threshold for detection (default: 0.3)
- `-f, --frames`: Consecutive frames threshold (default: 48)

## How It Works

1. **Face Detection**: Uses dlib's HOG-based face detector
2. **Landmark Detection**: Identifies 68 facial landmarks
3. **Eye Extraction**: Extracts eye regions from landmarks
4. **EAR Calculation**: Computes Eye Aspect Ratio using the formula:
   ```
   EAR = (||p2-p6|| + ||p3-p5||) / (2.0 * ||p1-p4||)
   ```
5. **Drowsiness Detection**: Monitors EAR values and frame counts
6. **Alert System**: Triggers audio/visual alerts when drowsiness detected

## Configuration

Edit `config.ini` to customize system behavior:
- Detection thresholds
- Display options  
- File paths
- Alert settings

## Data Analysis

The system logs all detection events. Use the analysis script to:
- Generate performance reports
- Visualize EAR patterns
- Analyze detection accuracy
- Optimize system parameters

## Files Structure

```
├── drowsiness_detection_system.py    # Main detection system
├── setup_drowsiness_detection.py     # Setup and installation script
├── analyze_drowsiness_data.py        # Data analysis and reporting
├── config.ini                        # Configuration file
├── requirements.txt                  # Python dependencies
├── drowsiness_detection_sample_data.csv  # Sample data for testing
├── system_requirements.csv           # System requirements table
└── hardware_requirements.csv         # Hardware specifications
```

## Safety Notice

⚠️ **Important**: This system is for research and development purposes. It should not be used as the sole safety system in vehicles. Always practice safe driving habits and take regular breaks during long drives.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- dlib library for facial landmark detection
- OpenCV community for computer vision tools  
- Research papers on Eye Aspect Ratio by Soukupová and Čech
- PyImageSearch tutorials and community

## Troubleshooting

### Common Issues

1. **dlib installation fails**: 
   - Install cmake: `pip install cmake`
   - For Windows: Install Visual Studio Build Tools

2. **Camera not detected**:
   - Check webcam index with `-w` parameter
   - Ensure camera permissions are granted

3. **Poor detection accuracy**:
   - Adjust EAR threshold with `-t` parameter
   - Ensure good lighting conditions
   - Position camera properly

4. **No sound alerts**:
   - Check audio file path with `-a` parameter
   - Verify pygame installation
   - Test system audio

For more help, check the documentation or open an issue.
'''

with open('README.md', 'w',encoding='utf-8') as f:
    f.write(readme_content)

print("=== COMPLETE PROJECT STRUCTURE CREATED ===")
print("\nFiles created:")
print("✓ setup_drowsiness_detection.py - Setup and installation script")
print("✓ analyze_drowsiness_data.py - Data analysis and reporting")
print("✓ config.ini - Configuration file")
print("✓ requirements.txt - Python dependencies")
print("✓ README.md - Complete project documentation")
print("\nProject is now ready for deployment!")
print("\nTo get started:")
print("1. Run: python setup_drowsiness_detection.py")
print("2. Run: python drowsiness_detection_system.py -p shape_predictor_68_face_landmarks.dat")
print("3. Analyze: python analyze_drowsiness_data.py")