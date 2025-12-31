#!/usr/bin/env python3
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
    print("This script will install all required dependencies and models.\n")
    
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
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("\n✓ All packages installed successfully!")
    
    # Download facial landmark predictor
    print("\nDownloading facial landmark predictor...")
    predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    predictor_file = "shape_predictor_68_face_landmarks.dat.bz2"
    predictor_extracted = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_extracted):
        if download_file(predictor_url, predictor_file):
            extract_bz2(predictor_file, predictor_extracted)
    else:
        print(f"✓ {predictor_extracted} already exists")
    
    # Create sample alarm sound (simple beep)
    print("\nCreating sample alarm sound...")
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
        # Add this line to ensure the array is C-contiguous:
        sound = np.ascontiguousarray(sound)
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
    
    print("\n=== Setup Complete ===")
    print("\nTo run the drowsiness detection system:")
    print("python drowsiness_detection_system.py -p shape_predictor_68_face_landmarks.dat")
    print("\nOptional parameters:")
    print("  -a alarm.wav          # Path to alarm sound file")
    print("  -w 0                  # Webcam index (default: 0)")
    print("  -t 0.3                # EAR threshold (default: 0.3)")
    print("  -f 48                 # Frame count threshold (default: 48)")

if __name__ == "__main__":
    main()
