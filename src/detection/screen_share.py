"""Screen sharing detection for Windows and macOS"""

import platform
import subprocess
import psutil
from typing import List


class ScreenShareDetector:
    """Detects if screen sharing is active"""
    
    # Common screen recording/sharing processes
    SCREEN_SHARING_PROCESSES = [
        'zoom', 'Zoom',
        'teams', 'Teams', 'ms-teams',
        'chrome', 'Google Chrome',  # For Google Meet
        'firefox', 'Firefox',
        'obs', 'OBS Studio',
        'obs64', 'obs32',
        'streamlabs', 'Streamlabs',
        'xsplit', 'XSplit',
        'nvidia', 'NVIDIA Share',  # ShadowPlay
        'dxtory',
        'bandicam',
        'fraps',
        'quicktime', 'QuickTime Player',
        'screenflow',
        'camtasia',
        'screencast',
        'screenrecorder',
        'sharex',
        'greenshot',
        'snagit'
    ]
    
    def __init__(self):
        """Initialize screen share detector"""
        self.system = platform.system()
    
    def is_screen_sharing(self) -> bool:
        """
        Check if screen sharing is currently active
        
        Returns:
            True if screen sharing is detected, False otherwise
        """
        if self.system == "Windows":
            return self._check_windows()
        elif self.system == "Darwin":  # macOS
            return self._check_macos()
        else:
            # Linux - similar to Windows approach
            return self._check_linux()
    
    def _check_windows(self) -> bool:
        """Check for screen sharing on Windows"""
        try:
            # Get all running processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    # Check if any screen sharing process is running
                    for sharing_process in self.SCREEN_SHARING_PROCESSES:
                        if sharing_process.lower() in proc_name:
                            # Additional check: verify it's actually sharing
                            # For Zoom, Teams, etc., we assume if the process is running,
                            # screen sharing might be active
                            # This is a heuristic - not perfect but reasonable
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for screen recording via Windows API (more advanced)
            # This would require additional Windows-specific libraries
            # For now, we rely on process detection
            
            return False
        
        except Exception as e:
            print(f"Error checking Windows screen sharing: {e}")
            return False
    
    def _check_macos(self) -> bool:
        """Check for screen sharing on macOS"""
        try:
            # Method 1: Check for screen recording processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    for sharing_process in self.SCREEN_SHARING_PROCESSES:
                        if sharing_process.lower() in proc_name:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Method 2: Check screen recording status via system_profiler
            # macOS has a system-level indicator for screen recording
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPApplicationsDataType'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                # This is a basic check - more sophisticated detection would
                # require Objective-C bindings or checking system preferences
            except:
                pass
            
            # Method 3: Check for specific macOS screen sharing indicators
            # Check if Screen Recording permission is being used
            # This is harder to detect without native code
            
            return False
        
        except Exception as e:
            print(f"Error checking macOS screen sharing: {e}")
            return False
    
    def _check_linux(self) -> bool:
        """Check for screen sharing on Linux"""
        try:
            # Similar to Windows approach
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    for sharing_process in self.SCREEN_SHARING_PROCESSES:
                        if sharing_process.lower() in proc_name:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
        
        except Exception as e:
            print(f"Error checking Linux screen sharing: {e}")
            return False
    
    def get_active_sharing_processes(self) -> List[str]:
        """
        Get list of active screen sharing processes
        
        Returns:
            List of process names that might be sharing screen
        """
        active_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name']
                    for sharing_process in self.SCREEN_SHARING_PROCESSES:
                        if sharing_process.lower() in proc_name.lower():
                            active_processes.append(proc_name)
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error getting active processes: {e}")
        
        return list(set(active_processes))  # Remove duplicates

