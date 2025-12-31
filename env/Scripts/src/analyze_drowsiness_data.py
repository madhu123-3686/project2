#!/usr/bin/env python3
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
        
        print("\n=== DROWSINESS DETECTION ANALYSIS ===")
        print(f"Total detection events: {len(df)}")
        print(f"Total alerts triggered: {df['alert_triggered'].sum()}")
        print(f"Alert rate: {df['alert_triggered'].mean():.2%}")
        
        print("\n=== EAR STATISTICS ===")
        print(f"Mean EAR: {df['ear_value'].mean():.3f}")
        print(f"Median EAR: {df['ear_value'].median():.3f}")
        print(f"Min EAR: {df['ear_value'].min():.3f}")
        print(f"Max EAR: {df['ear_value'].max():.3f}")
        print(f"Std EAR: {df['ear_value'].std():.3f}")
        
        # EAR distribution by alert status
        alert_ear = df[df['alert_triggered']]['ear_value']
        normal_ear = df[~df['alert_triggered']]['ear_value']
        
        print("\n=== EAR BY ALERT STATUS ===")
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
        
        print("\n✓ Analysis plots saved as 'drowsiness_analysis.png'")
    
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
        
        print("\n✓ Analysis report saved as 'drowsiness_analysis_report.txt'")
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
