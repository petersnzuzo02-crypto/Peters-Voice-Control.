package com.peters.voicecontrol

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioManager
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.view.KeyEvent
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import java.io.File
import java.util.concurrent.Executors

/**
 * Offline hands-free prototype for Android 13.
 * One microphone owner (Vosk) detects both the wake phrase and the music command.
 * No cloud speech recognition and no recording files are created.
 */
class PetersHandsFreeService : Service(), RecognitionListener {
    companion object {
        const val ACTION_START = "com.peters.voicecontrol.START_HANDS_FREE"
        const val ACTION_STOP = "com.peters.voicecontrol.STOP_HANDS_FREE"
        private const val CHANNEL_ID = "peters_hands_free"
        private const val NOTIFICATION_ID = 1401
        private const val COMMAND_WINDOW_MS = 8000L
    }

    private val loader = Executors.newSingleThreadExecutor()
    private var model: Model? = null
    private var recognizer: Recognizer? = null
    private var speechService: SpeechService? = null
    @Volatile private var stopped = false
    private var started = false
    private var armedUntil = 0L
    private var lastWakeAt = 0L
    private var lastCommandAt = 0L
    private lateinit var notifications: NotificationManager

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        notifications = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        notifications.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Peters hands-free control", NotificationManager.IMPORTANCE_LOW)
                .apply { description = "Shows offline listening status and provides Stop" }
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (started) return START_NOT_STICKY
        started = true
        stopped = false
        val notification = makeNotification("Loading offline voice model...")
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (e: Exception) {
            stopSelf()
            return START_NOT_STICKY
        }

        loader.execute {
            try {
                // Assets are extracted to private app storage because Vosk requires a file-system path.
                val modelFolder = File(filesDir, "vosk-model")
                if (!File(modelFolder, "am/final.mdl").exists() ||
                    !File(modelFolder, "conf/model.conf").exists()) {
                    copyAssetDirectory("vosk-model", modelFolder)
                }
                if (stopped) return@execute
                val loaded = Model(modelFolder.absolutePath)
                if (stopped) { loaded.close(); return@execute }
                val rec = Recognizer(loaded, 16000.0f)
                if (stopped) { rec.close(); loaded.close(); return@execute }
                runOnMain {
                    if (stopped) { rec.close(); loaded.close(); return@runOnMain }
                    try {
                        val microphone = SpeechService(rec, 16000.0f)
                        model = loaded
                        recognizer = rec
                        speechService = microphone
                        if (microphone.startListening(this)) {
                            showStatus("Listening for 'Hey Peters' (offline)")
                        } else {
                            showStatus("Could not start wake-phrase listening")
                        }
                    } catch (e: Exception) {
                        rec.close(); loaded.close()
                        showStatus("Microphone unavailable: ${e.message}")
                    }
                }
            } catch (e: Exception) {
                runOnMain { if (!stopped) showStatus("Vosk setup error: ${e.message}") }
            }
        }
        return START_NOT_STICKY
    }

    private fun copyAssetDirectory(assetPath: String, dest: File) {
        val children = assets.list(assetPath) ?: emptyArray()
        if (children.isEmpty()) {
            dest.parentFile?.mkdirs()
            assets.open(assetPath).use { input ->
                dest.outputStream().use { output -> input.copyTo(output) }
            }
        } else {
            if (!dest.exists() && !dest.mkdirs()) error("Cannot create ${dest.name}")
            for (child in children) copyAssetDirectory("$assetPath/$child", File(dest, child))
        }
    }

    private fun runOnMain(action: () -> Unit) = android.os.Handler(mainLooper).post(action)

    private fun words(json: String?, key: String): String = try {
        JSONObject(json ?: "{}").optString(key, "")
            .lowercase(java.util.Locale.ROOT)
            .replace(Regex("[^a-z0-9 ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    } catch (_: Exception) { "" }

    private fun containsWake(phrase: String): Boolean =
        Regex("\\bhey (peters|peter)\\b").containsMatchIn(phrase)

    private fun stripWake(phrase: String): String =
        phrase.replace(Regex(".*?\\bhey (peters|peter)\\b"), "").trim()

    /** Only direct, unambiguous music-control phrases are accepted after wake-up. */
    private fun musicCommand(phrase: String): String? {
        if (phrase.isBlank() || Regex("\\b(dont|do not|never|not|without)\\b").containsMatchIn(phrase)) return null
        val p = phrase.removePrefix("please ").trim()
        val matches = mutableListOf<String>()
        if (Regex("\\b(play|start|resume)\\b").containsMatchIn(p) &&
            Regex("\\b(music|song|audio|playback)\\b").containsMatchIn(p)) matches += "PLAY"
        if (Regex("\\bpause\\b").containsMatchIn(p)) matches += "PAUSE"
        if (Regex("\\b(next|skip)\\b").containsMatchIn(p)) matches += "NEXT"
        if (Regex("\\bstop\\b").containsMatchIn(p)) matches += "STOP"
        return matches.singleOrNull()
    }

    private fun inspect(text: String, isFinal: Boolean) {
        if (stopped || text.isBlank()) return
        val now = SystemClock.elapsedRealtime()
        if (armedUntil != 0L && now > armedUntil) {
            armedUntil = 0L
            showStatus("Listening for 'Hey Peters' (offline)")
        }
        val wake = containsWake(text)
        if (wake && armedUntil == 0L && now - lastWakeAt > 1500L) {
            armedUntil = now + COMMAND_WINDOW_MS
            lastWakeAt = now
            showStatus("Wake phrase detected — say a music command")
        }
        if (armedUntil == 0L || !isFinal) return
        val phrase = if (wake) stripWake(text) else text
        val command = musicCommand(phrase)
        if (command != null && now - lastCommandAt > 1300L) {
            lastCommandAt = now
            armedUntil = 0L
            dispatch(command)
            showStatus("$command sent — listening for 'Hey Peters'")
        }
    }

    override fun onPartialResult(hypothesis: String?) = inspect(words(hypothesis, "partial"), false)
    override fun onResult(hypothesis: String?) = inspect(words(hypothesis, "text"), true)
    override fun onFinalResult(hypothesis: String?) = inspect(words(hypothesis, "text"), true)
    override fun onError(exception: Exception?) {
        if (!stopped) showStatus("Offline listening error: ${exception?.message}")
    }
    override fun onTimeout() { if (!stopped) showStatus("Wake-word listener timed out") }

    private fun dispatch(command: String) {
        val key = when (command) {
            "PLAY" -> KeyEvent.KEYCODE_MEDIA_PLAY
            "PAUSE" -> KeyEvent.KEYCODE_MEDIA_PAUSE
            "NEXT" -> KeyEvent.KEYCODE_MEDIA_NEXT
            "STOP" -> KeyEvent.KEYCODE_MEDIA_STOP
            else -> return
        }
        val manager = getSystemService(AUDIO_SERVICE) as AudioManager
        manager.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, key))
        manager.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_UP, key))
    }

    private fun makeNotification(message: String): Notification {
        val stopIntent = Intent(this, PetersHandsFreeService::class.java).apply { action = ACTION_STOP }
        val stopAction = PendingIntent.getService(
            this, 1, stopIntent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val openApp = PendingIntent.getActivity(
            this, 2, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Peters Voice Control")
            .setContentText(message)
            .setContentIntent(openApp)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(Notification.Action.Builder(
                android.R.drawable.ic_menu_close_clear_cancel, "Stop", stopAction
            ).build())
            .build()
    }

    private fun showStatus(message: String) {
        if (!stopped) notifications.notify(NOTIFICATION_ID, makeNotification(message))
    }

    override fun onDestroy() {
        stopped = true
        try { speechService?.cancel() } catch (_: Exception) { }
        try { speechService?.shutdown() } catch (_: Exception) { }
        speechService = null
        try { recognizer?.close() } catch (_: Exception) { }
        recognizer = null
        try { model?.close() } catch (_: Exception) { }
        model = null
        loader.shutdown()
        super.onDestroy()
    }
}
