# Real-Time Driver Drowsiness Detection System

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
python drowsiness_detection_system.py \
    -p shape_predictor_68_face_landmarks.dat \
    -a alarm.wav \
    -w 0 \
    -t 0.3 \
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
