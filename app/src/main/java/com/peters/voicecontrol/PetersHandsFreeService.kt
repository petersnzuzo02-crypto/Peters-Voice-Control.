package com.peters.voicecontrol

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder

/**
 * Stage 1 of hands-free mode: a user-activated, stoppable microphone foreground
 * service. A wake-word engine will be added in the next stage.
 * No microphone capture or speech recognition occurs in this version.
 */
class PetersHandsFreeService : Service() {
    companion object {
        const val ACTION_START = "com.peters.voicecontrol.START_HANDS_FREE"
        const val ACTION_STOP = "com.peters.voicecontrol.STOP_HANDS_FREE"
        private const val CHANNEL_ID = "peters_hands_free"
        private const val NOTIFICATION_ID = 1401
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Peters hands-free control",
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Shows whether hands-free mode is enabled" }
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val stopIntent = Intent(this, PetersHandsFreeService::class.java).apply {
            action = ACTION_STOP
        }
        val stopAction = PendingIntent.getService(
            this, 1, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val openApp = PendingIntent.getActivity(
            this, 2, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val notification = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Peters Voice Control active")
            .setContentText("Hands-free service ready — wake-word engine not connected yet")
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(Notification.Action.Builder(
                android.R.drawable.ic_menu_close_clear_cancel,
                "Stop",
                stopAction
            ).build())
            .build()

        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        return START_NOT_STICKY
    }
}
