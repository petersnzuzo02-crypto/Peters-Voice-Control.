# Peters Voice Control — Android prototype

A **local-only recognition prototype** for the ESP32 Bluetooth audio amplifier. It uses the supplied `Voice_Command_CNN.tflite` model (14.33 KB) with the five labels in **exact order**: NEXT, PAUSE, PLAY, STOP, UNKNOWN.

**Status:** source project prepared; GitHub Actions APK build and real-phone inference **not yet verified**. The model's earlier validation accuracy is **68.18%** (15/22), with UNKNOWN recall **40%** (2/5). **No Bluetooth playback commands are sent** in this build.

## Build without Android Studio

1. Keep this entire project's files and folders (including `.github/workflows`) at the root of your GitHub repository.
2. On GitHub, choose **Actions → Build Android APK → Run workflow**, or push a commit to `main`.
3. Wait for the build to finish; open its run and download the **Peters-Voice-Control-debug-apk** artifact. The artifact is a ZIP containing `app-debug.apk`.
4. Install the debug APK on your own Android phone, grant microphone permission, and test predictions. Android may ask you to permit installation from your browser/files app.
5. If the build fails, share the first red error from the build logs for a targeted fix.

## What the app does

- Records a fixed 3.5-second window of 16 kHz mono PCM audio when you tap the button.
- Trims leading/trailing low-energy audio using a 20 ms RMS frame, 10 ms hop, 10% peak threshold and 150 ms padding.
- Computes 13 librosa-style MFCCs (400 FFT, 160 hop, 128 Slaney mel filters, power-to-dB, orthonormal DCT-II), pads/truncates to 160 frames and applies the saved 13 training-set means and standard deviations.
- Runs the provided float32 `[1,160,13]` model with a local LiteRT Interpreter and prints the predicted label and uncalibrated scores.

## Caveats before connecting to ESP32

1. **MFCC parity is not yet verified:** Kotlin processing was written to approximate the Colab `librosa` pipeline. Before relying on results, compare Android and Colab MFCCs for the *same actual recorded WAV* and adjust if needed. Differences in microphone recording conditions, audio framing and spectral implementation can affect predictions.
2. A fixed 160-frame input may truncate longer speech (including phrases); the app displays a warning when a trimmed clip may be too long. The same limitation was present during training.
3. Test the app with known recorded phrases and unrelated speech. Do not enable automatic playback until rejection behaviour and real-phone reliability are validated.
4. The audio amplifier uses A2DP. Bluetooth command integration (potentially AVRCP) is a separate unfinished stage; this prototype intentionally leaves the current ESP32 sketch untouched.
5. **Privacy:** this first APK does not require internet permission and does not upload audio. It is a debug build, not a Play Store release.

## Development files

- `app/src/main/assets/Voice_Command_CNN.tflite`: supplied trained model.
- `app/src/main/assets/Voice_Command_Labels.txt`: supplied label order.
- `app/src/main/assets/normalization.json`: numerically extracted from supplied `CNN_Normalisation.npz`.
- `app/src/main/java/com/peters/voicecontrol/AudioFeatures.kt`: audio preprocessing.
- `app/src/main/java/com/peters/voicecontrol/MainActivity.kt`: local microphone UI and LiteRT inference.
- `.github/workflows/build-apk.yml`: cloud APK build.
