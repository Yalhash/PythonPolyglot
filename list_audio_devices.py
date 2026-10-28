"""Utility script to list available audio devices"""

import sounddevice as sd
import platform

def list_devices():
    """List all available audio devices"""
    print("=" * 80)
    print("Available Audio Devices")
    print("=" * 80)
    print(f"Platform: {platform.system()}\n")
    
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        print(f"\nDevice {i}: {device['name']}")
        print(f"  Host API: {device.get('hostapi_name', 'Unknown')}")
        print(f"  Max Input Channels: {device.get('max_input_channels', 0)}")
        print(f"  Max Output Channels: {device.get('max_output_channels', 0)}")
        print(f"  Default Sample Rate: {device.get('default_samplerate', 'Unknown')} Hz")
        
        # Highlight loopback devices
        if 'loopback' in device['name'].lower():
            print("  *** LOOPBACK DEVICE (good for system audio capture) ***")
        
        # Highlight virtual audio devices
        name_lower = device['name'].lower()
        if any(keyword in name_lower for keyword in ['vb-audio', 'virtual', 'blackhole', 'soundflower', 'obs']):
            print("  *** VIRTUAL AUDIO DEVICE (good for system audio capture) ***")
    
    print("\n" + "=" * 80)
    print("Tips for Windows:")
    print("- Look for devices with 'loopback' in the name")
    print("- Or install a virtual audio device like:")
    print("  - VB-Audio Cable (free)")
    print("  - OBS Virtual Audio")
    print("  - Voicemeeter")
    print("\nThen route your call audio through that device and select it in Polyglot.")
    print("=" * 80)

if __name__ == "__main__":
    try:
        list_devices()
    except Exception as e:
        print(f"Error listing devices: {e}")

