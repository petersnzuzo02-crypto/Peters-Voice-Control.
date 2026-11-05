package com.peters.voicecontrol

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var result: TextView
    private lateinit var recordButton: Button
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
        result = TextView(this).apply {
            textSize = 18f
            text = "Results appear here. No Bluetooth commands will be sent."
            setPadding(0, 30, 0, 30)
        }
        val note = TextView(this).apply {
            text = "EXPERIMENTAL RECOGNITION ONLY\n" +
                "The CNN achieved 68.18% validation accuracy on 22 recordings. " +
                "UNKNOWN recall was only 40%. Predictions are not safe for automatic playback control. " +
                "No audio is uploaded; processing runs on this phone."
            textSize = 14f
        }
        container.addView(heading)
        container.addView(status)
        container.addView(recordButton)
        container.addView(result)
        container.addView(note)
        setContentView(ScrollView(this).apply { addView(container) })
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
            status.text = "Offline model loaded. Tap to record."
            recordButton.isEnabled = true
        } catch (e: Exception) {
            status.text = "Model failed to load: ${e.message}"
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
                val report = buildString {
                    append("Predicted: ${labels[winning]} (experimental)\n")
                    append("Trimmed audio: ${"%.2f".format(trimmed.size / 16000.0)} s\n")
                    if (trimmed.size > 25440) append("Warning: audio may exceed 160 MFCC frames; end was truncated.\n")
                    append("\nModel scores (not calibrated):\n")
                    for (i in labels.indices) append("${labels[i]}: ${"%.3f".format(scores[i])}\n")
                    append("\nNothing was sent to ESP32.")
                }
                mainHandler.post { result.text = report; status.text = "Recording processed offline." }
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
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) startRecording()
            else status.text = "Microphone permission is required to record commands."
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        interpreter?.close()
    }
}
