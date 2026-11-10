package com.peters.voicecontrol

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.net.Uri
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.AudioManager
import android.view.KeyEvent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.widget.Button
import android.widget.Spinner
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.PI
import java.util.Locale

class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var result: TextView
    private lateinit var recordButton: Button
    private lateinit var offlineButton: Button
    private var speechRecognizer: SpeechRecognizer? = null
    private var offlineListening = false
    private var pendingOfflinePermission = false
    private lateinit var diagnosticButton: Button
    private lateinit var datasetButton: Button
    private lateinit var exportButton: Button
    private lateinit var classSpinner: Spinner
    private lateinit var countsView: TextView
    // Legacy CNN scores remain diagnostic only. New offline mode uses recognised words.
    private var pendingDatasetPermission = false
    private val datasetClasses = listOf("PLAY", "PAUSE", "NEXT", "STOP", "UNKNOWN")
    private val createZipRequest = 301
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var busy = false
    private val labels = listOf("NEXT", "PAUSE", "PLAY", "STOP", "UNKNOWN")
    private lateinit var normalMean: FloatArray
    private lateinit var normalStd: FloatArray
    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }
        val heading = TextView(this).apply {
            text = "Peters Voice Control"
            textSize = 25f
            gravity = Gravity.CENTER
        }
        status = TextView(this).apply {
            text = "Loading model..."
            textSize = 17f
            setPadding(0, 30, 0, 30)
        }
        recordButton = Button(this).apply {
            text = "Record command (3.5 seconds)"
            isEnabled = false
            setOnClickListener { startRecording() }
        }
        offlineButton = Button(this).apply {
            text = "VOICE COMMAND — ON-DEVICE RECOGNITION"
            isEnabled = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
                SpeechRecognizer.isOnDeviceRecognitionAvailable(this@MainActivity)
            setOnClickListener { startOfflineRecognition() }
        }
        diagnosticButton = Button(this).apply {
            text = "Check MFCC parity (no microphone)"
            isEnabled = false
            setOnClickListener { runMfccDiagnostic() }
        }
        val mediaHeading = TextView(this).apply {
            text = "MUSIC PLAYBACK CONTROLS"
            textSize = 18f
            setPadding(0, 25, 0, 8)
        }
        val mediaHelp = TextView(this).apply {
            text = "Connect your phone to Nzuzo Bluetooth Amp and start music in your phone's player. " +
                "Manual buttons test playback. On-device voice mode uses spoken words; the old CNN is for testing only."
            textSize = 14f
        }
        val manualControls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        for (command in listOf("PLAY", "PAUSE", "NEXT", "STOP")) {
            val button = Button(this).apply {
                text = "Manual $command"
                setOnClickListener { sendMediaCommand(command) }
            }
            manualControls.addView(button)
        }
        val datasetHeading = TextView(this).apply {
            text = "DATASET RECORDING MODE"
            textSize = 18f
            setPadding(0, 34, 0, 8)
        }
        val datasetHelp = TextView(this).apply {
            text = "Select the intended command, then record one phrase. " +
                "The original 16 kHz WAV is saved privately on this phone. " +
                "Export ZIP before uninstalling the app."
            textSize = 15f
        }
        classSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                datasetClasses
            )
        }
        datasetButton = Button(this).apply {
            text = "Save labelled recording (3.5 seconds)"
            setOnClickListener { startDatasetRecording() }
        }
        exportButton = Button(this).apply {
            text = "Export recorded WAV files as ZIP"
            setOnClickListener { requestZipExport() }
        }
        countsView = TextView(this).apply {
            textSize = 15f
            setPadding(0, 12, 0, 16)
        }
        refreshDatasetCounts()
        result = TextView(this).apply {
            textSize = 18f
            text = "Results appear here. Use on-device voice recognition for commands; CNN mode is diagnostic."
            setPadding(0, 30, 0, 30)
        }
        val note = TextView(this).apply {
            text = "CNN v2 — EXPERIMENTAL VOICE RECOGNITION\n" +
                "Validation accuracy was 80.43% on 46 examples, but phone recognition remains unreliable. " +
                "CNN is retained for comparison; its scores are not calibrated probabilities. " +
                "No audio is uploaded; processing runs on this phone."
            textSize = 14f
        }
        container.addView(heading)
        container.addView(status)
        container.addView(offlineButton)
        container.addView(recordButton)
        container.addView(diagnosticButton)
        container.addView(result)
        container.addView(mediaHeading)
        container.addView(mediaHelp)
        container.addView(manualControls)
        container.addView(datasetHeading)
        container.addView(datasetHelp)
        container.addView(classSpinner)
        container.addView(datasetButton)
        container.addView(exportButton)
        container.addView(countsView)
        container.addView(note)
        setContentView(ScrollView(this).apply { addView(container) })
        if (!offlineButton.isEnabled) {
            status.text = "On-device speech recognition unavailable on this phone (requires Android 12+ and installed offline recognition service)."
        }
        try {
            val norm = JSONObject(assets.open("normalization.json").bufferedReader().use { it.readText() })
            val m = norm.getJSONArray("mean")
            val s = norm.getJSONArray("std")
            normalMean = FloatArray(m.length()) { m.getDouble(it).toFloat() }
            normalStd = FloatArray(s.length()) { s.getDouble(it).toFloat() }
            val rawModel = assets.open("Voice_Command_CNN.tflite").use { it.readBytes() }
            val modelBuffer = ByteBuffer.allocateDirect(rawModel.size).order(ByteOrder.nativeOrder())
            modelBuffer.put(rawModel)
            modelBuffer.rewind()
            interpreter = Interpreter(modelBuffer)
            status.text = "CNN loaded. On-device recogniser: " + if (offlineButton.isEnabled) "available" else "unavailable"
            recordButton.isEnabled = true
            diagnosticButton.isEnabled = true
        } catch (e: Exception) {
            status.text = "CNN failed to load: ${e.message}. On-device voice mode may still work."
        }
    }

    /** The offline service is requested explicitly; never silently fall back to cloud recognition. */
    private fun startOfflineRecognition() {
        if (busy || offlineListening) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingOfflinePermission = true
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 100)
            return
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S ||
            !SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) {
            status.text = "On-device speech recognition is unavailable; nothing was sent."
            return
        }
        try {
            speechRecognizer?.destroy()
            speechRecognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(this)
            val listener = object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) { status.text = "Listening: say a music command." }
                override fun onBeginningOfSpeech() { status.text = "Speech detected..." }
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() { status.text = "Recognising words offline..." }
                override fun onError(error: Int) {
                    finishOfflineRecognition()
                    status.text = "Offline recognition error $error. Check language pack and microphone."
                    result.text = "No command was sent."
                }
                override fun onResults(results: Bundle?) {
                    val hypotheses = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?: arrayListOf()
                    // Accept the primary transcription only, not an arbitrary alternative hypothesis.
                    val phrase = hypotheses.firstOrNull().orEmpty()
                    val command = matchSpokenCommand(phrase)
                    finishOfflineRecognition()
                    result.text = "Recognised words: ${if (phrase.isBlank()) "(none)" else phrase}\n" +
                        "Command: ${command ?: "UNKNOWN"}\n" +
                        (if (command == null) "No matching command; no playback action." else
                            "Exact supported phrase matched. Sending media command.")
                    if (command != null) sendMediaCommand(command)
                    else status.text = "UNKNOWN — no playback action."
                }
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            }
            speechRecognizer?.setRecognitionListener(listener)
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            }
            offlineListening = true
            offlineButton.isEnabled = false
            result.text = "Listening for spoken command..."
            speechRecognizer?.startListening(intent)
        } catch (e: Exception) {
            finishOfflineRecognition()
            status.text = "Could not start on-device recognition: ${e.message}"
        }
    }

    private fun finishOfflineRecognition() {
        offlineListening = false
        speechRecognizer?.destroy()
        speechRecognizer = null
        offlineButton.isEnabled = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            SpeechRecognizer.isOnDeviceRecognitionAvailable(this)
    }

    private fun matchSpokenCommand(raw: String): String? {
        val phrase = raw.lowercase(Locale.ROOT)
            .replace(Regex("[^a-z0-9 ]"), " ")
            .trim().replace(Regex("\\s+"), " ")
        return when (phrase) {
            "play", "play music", "play the music", "start music", "start the music", "resume music", "resume the music" -> "PLAY"
            "pause", "pause music", "pause the music" -> "PAUSE"
            "next", "next song", "next track", "skip song", "skip the song", "skip track", "skip to next song" -> "NEXT"
            "stop", "stop music", "stop the music", "stop playing", "stop playback" -> "STOP"
            else -> null
        }
    }

    private fun runMfccDiagnostic() {
        if (busy) return
        busy = true
        diagnosticButton.isEnabled = false
        recordButton.isEnabled = false
        status.text = "Comparing deterministic MFCC features..."
        executor.execute {
            try {
                // Exactly 2.0 seconds: 0.4 s silence, 1.2 s tones, 0.4 s silence.
                // Same definition must be used in the Colab reference cell.
                val signal = FloatArray(32000)
                for (i in 6400 until 25600) {
                    val t = (i - 6400) / 16000.0
                    signal[i] = (0.25 * sin(2.0 * PI * 220.0 * t) +
                        0.10 * sin(2.0 * PI * 440.0 * t)).toFloat()
                }
                val trimmed = AudioFeatures.trim(signal)
                val feats = AudioFeatures.extract(signal, normalMean, normalStd)[0]
                val report = buildString {
                    append("MFCC PARITY DIAGNOSTIC (synthetic audio)\n")
                    append("Trimmed samples: ${trimmed.size}\n")
                    append("Shape: 160 x 13\n")
                    append("Values: frame, coefficient, normalized MFCC\n")
                    for (frame in listOf(0, 20, 50, 100)) {
                        for (coeff in 0..3) {
                            append("$frame,$coeff,${String.format(Locale.US, "%.6f", feats[frame][coeff])}\n")
                        }
                    }
                    append("\nNo audio recorded or transmitted.")
                }
                mainHandler.post { result.text = report; status.text = "MFCC diagnostic complete." }
            } catch (e: Exception) {
                mainHandler.post { status.text = "Diagnostic error: ${e.message}" }
            } finally {
                mainHandler.post {
                    busy = false
                    diagnosticButton.isEnabled = true
                    recordButton.isEnabled = true
                }
            }
        }
    }

    private fun datasetDirectory(): File = File(filesDir, "voice_dataset")

    private fun refreshDatasetCounts() {
        val root = datasetDirectory()
        val counts = datasetClasses.joinToString("  |  ") { category ->
            val count = File(root, category).listFiles()?.count {
                it.isFile && it.extension.equals("wav", ignoreCase = true)
            } ?: 0
            "$category: $count"
        }
        countsView.text = "Saved on phone: $counts"
    }

    private fun setRecordingControls(enabled: Boolean) {
        recordButton.isEnabled = enabled
        diagnosticButton.isEnabled = enabled
        datasetButton.isEnabled = enabled
        exportButton.isEnabled = enabled
    }

    private fun startDatasetRecording() {
        if (busy) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pendingDatasetPermission = true
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 100)
            return
        }
        val category = datasetClasses[classSpinner.selectedItemPosition]
        busy = true
        setRecordingControls(false)
        status.text = "Recording $category sample..."
        executor.execute {
            try {
                val audio = captureAudio()
                val folder = File(datasetDirectory(), category)
                if (!folder.exists() && !folder.mkdirs()) error("Could not create dataset directory")
                val file = File(folder, "${category}_${System.currentTimeMillis()}.wav")
                FileOutputStream(file).use { output -> writeWavPcm16(audio, output) }
                mainHandler.post {
                    status.text = "Saved labelled $category recording"
                    result.text = "Saved ${file.name} (${String.format(Locale.US, "%.2f", audio.size / 16000.0)} s). " +
                        "This sample was NOT used for a prediction and was NOT sent to the ESP32."
                    refreshDatasetCounts()
                }
            } catch (e: Exception) {
                mainHandler.post { status.text = "Dataset recording error: ${e.message}" }
            } finally {
                mainHandler.post { busy = false; setRecordingControls(true) }
            }
        }
    }

    private fun writeWavPcm16(audio: FloatArray, output: OutputStream) {
        val dataBytes = audio.size * 2
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + dataBytes)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1.toShort()) // PCM
        header.putShort(1.toShort()) // mono
        header.putInt(16000)
        header.putInt(16000 * 2)
        header.putShort(2.toShort())
        header.putShort(16.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataBytes)
        output.write(header.array())
        val bytes = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in audio) {
            if (bytes.remaining() < 2) {
                output.write(bytes.array(), 0, bytes.position())
                bytes.clear()
            }
            val pcm = (sample.coerceIn(-1f, 0.9999695f) * 32768f).toInt()
                .coerceIn(-32768, 32767)
            bytes.putShort(pcm.toShort())
        }
        if (bytes.position() > 0) output.write(bytes.array(), 0, bytes.position())
    }

    private fun requestZipExport() {
        if (busy) return
        val count = datasetClasses.sumOf { category ->
            File(datasetDirectory(), category).listFiles()?.count { it.extension.equals("wav", true) } ?: 0
        }
        if (count == 0) {
            status.text = "No dataset recordings to export yet."
            return
        }
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/zip"
            putExtra(Intent.EXTRA_TITLE, "Peters_Voice_Dataset.zip")
        }
        startActivityForResult(intent, createZipRequest)
    }

    @Deprecated("Uses legacy Activity callback for API 26 compatibility")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != createZipRequest || resultCode != RESULT_OK) return
        val destination: Uri = data?.data ?: return
        if (busy) return
        busy = true
        setRecordingControls(false)
        status.text = "Exporting WAV dataset..."
        executor.execute {
            try {
                var exported = 0
                val out = contentResolver.openOutputStream(destination)
                    ?: error("Cannot open ZIP destination")
                ZipOutputStream(out.buffered()).use { zip ->
                    for (category in datasetClasses) {
                        val folder = File(datasetDirectory(), category)
                        val wavs = folder.listFiles()?.filter {
                            it.isFile && it.extension.equals("wav", true)
                        }?.sortedBy { it.name } ?: emptyList()
                        for (file in wavs) {
                            zip.putNextEntry(ZipEntry("$category/${file.name}"))
                            file.inputStream().use { input -> input.copyTo(zip) }
                            zip.closeEntry()
                            exported++
                        }
                    }
                }
                mainHandler.post {
                    status.text = "ZIP exported successfully ($exported WAV files)."
                    result.text = "Dataset ZIP saved to the location you selected. " +
                        "The original recordings remain privately on this phone. " +
                        "Check the ZIP before deleting any app data."
                }
            } catch (e: Exception) {
                mainHandler.post { status.text = "ZIP export failed: ${e.message}" }
            } finally {
                mainHandler.post { busy = false; setRecordingControls(true) }
            }
        }
    }

    private fun sendMediaCommand(command: String) {
        val keyCode = when (command) {
            "PLAY" -> KeyEvent.KEYCODE_MEDIA_PLAY
            "PAUSE" -> KeyEvent.KEYCODE_MEDIA_PAUSE
            "NEXT" -> KeyEvent.KEYCODE_MEDIA_NEXT
            "STOP" -> KeyEvent.KEYCODE_MEDIA_STOP
            else -> return
        }
        try {
            val audio = getSystemService(AUDIO_SERVICE) as AudioManager
            audio.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, keyCode))
            audio.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_UP, keyCode))
            status.text = "Sent $command media-button event to Android. Verify the player responded."
        } catch (e: Exception) {
            status.text = "Could not send $command: ${e.message}"
        }
    }

    private fun startRecording() {
        if (busy) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 100)
            return
        }
        busy = true
        recordButton.isEnabled = false
        status.text = "Recording..."
        result.text = "Listening for 3.5 seconds."
        executor.execute {
            try {
                val raw = captureAudio()
                val trimmed = AudioFeatures.trim(raw)
                val features = AudioFeatures.extract(raw, normalMean, normalStd)
                val input = ByteBuffer.allocateDirect(160 * 13 * 4).order(ByteOrder.nativeOrder())
                for (frame in features[0]) for (coefficient in frame) input.putFloat(coefficient)
                input.rewind()
                val output = Array(1) { FloatArray(5) }
                val engine = interpreter ?: error("Model was not initialised")
                synchronized(engine) { engine.run(input, output) }
                val scores = output[0]
                val winning = scores.indices.maxByOrNull { scores[it] } ?: error("No model output")
                val winningScore = scores[winning]
                val secondScore = scores.indices.filter { it != winning }
                    .maxOf { scores[it] }
                val margin = winningScore - secondScore
                val candidate = labels[winning]
                // CNN is diagnostic only: do not reject correct predictions by arbitrary score
                // thresholds or trigger media actions from unreliable model predictions.
                val report = buildString {
                    append("Predicted: $candidate (experimental)\n")
                    append("Trimmed audio: ${"%.2f".format(Locale.US, trimmed.size / 16000.0)} s\n")
                    append("Top score: ${"%.3f".format(Locale.US, winningScore)}, margin: ${"%.3f".format(Locale.US, margin)}\n")
                    if (trimmed.size > 25440) append("Warning: speech may exceed the 160-frame limit.\n")
                    append("\nModel scores (not calibrated):\n")
                    for (i in labels.indices) append("${labels[i]}: ${"%.3f".format(Locale.US, scores[i])}\n")
                    append("\nCNN diagnostic only — no media action. Use on-device voice mode.")
                }
                mainHandler.post {
                    result.text = report
                    status.text = "CNN result displayed for comparison (no action sent)."
                }
            } catch (e: Exception) {
                mainHandler.post { status.text = "Error: ${e.message}" }
            } finally {
                mainHandler.post { busy = false; recordButton.isEnabled = true }
            }
        }
    }

    @Suppress("MissingPermission")
    private fun captureAudio(): FloatArray {
        val sampleRate = 16000
        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        require(minBuffer > 0) { "16 kHz audio capture is unavailable" }
        val bufferBytes = max(minBuffer, 8192)
        val audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC, sampleRate,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferBytes
        )
        require(audioRecord.state == AudioRecord.STATE_INITIALIZED) { "Microphone could not be initialised" }
        val targetSamples = sampleRate * 35 / 10
        val capture = FloatArray(targetSamples)
        val scratch = ShortArray(2048)
        var count = 0
        try {
            audioRecord.startRecording()
            while (count < targetSamples) {
                val n = audioRecord.read(scratch, 0, minOf(scratch.size, targetSamples - count))
                if (n <= 0) error("AudioRecord read failed: $n")
                for (i in 0 until n) capture[count++] = scratch[i].toFloat() / 32768f
            }
        } finally {
            try { audioRecord.stop() } catch (_: Exception) {}
            audioRecord.release()
        }
        return capture
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            val granted = grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
            val forDataset = pendingDatasetPermission
            val forOffline = pendingOfflinePermission
            pendingDatasetPermission = false
            pendingOfflinePermission = false
            if (granted) {
                if (forDataset) startDatasetRecording()
                else if (forOffline) startOfflineRecognition()
                else startRecording()
            } else status.text = "Microphone permission is required to record commands."
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer?.destroy()
        executor.shutdown()
        interpreter?.close()
    }
}
