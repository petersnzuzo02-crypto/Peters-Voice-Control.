
package com.peters.voicecontrol

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.net.Uri
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
    private lateinit var diagnosticButton: Button
    private lateinit var datasetButton: Button
    private lateinit var exportButton: Button
    private lateinit var classSpinner: Spinner
    private lateinit var countsView: TextView
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
0%
