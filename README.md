
# Peters Voice Control

Voice-Controlled Bluetooth Audio Amplifier Using ESP32 and Machine Learning

## Project Description
An Android application that recognises voice commands
offline using a custom-trained CNN model and controls
an ESP32 Bluetooth audio amplifier.

## Hardware
- ESP32-WROOM
- LM386 Audio Amplifier Module
- 9 Ohm, 2 W Speaker
- Android Smartphone

## Machine Learning
- 1D Convolutional Neural Network (CNN)
- TensorFlow Lite
- MFCC Audio Features
- 150 Voice Recordings
- Five Classification Categories

## Voice Commands
1. PLAY
2. PAUSE
3. NEXT
4. STOP
5. UNKNOWN

## Android Application
Application Name: Peters Voice Control

The Android application will capture microphone audio,
extract MFCC features, and classify voice commands
offline using a TensorFlow Lite model.

## Bluetooth Audio
ESP32 Bluetooth Name: Nzuzo Bluetooth Amp

## Development
Android APK will be built using GitHub Actions.
