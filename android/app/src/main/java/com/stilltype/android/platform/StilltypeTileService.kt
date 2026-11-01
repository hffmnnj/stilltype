package com.stilltype.android.platform

import android.app.PendingIntent
import android.os.Build
import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import com.stilltype.android.StilltypeApplication
import com.stilltype.android.core.ModelStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Control-center shortcut that opens quick capture over the previous app.
 * Never starts the camera, the recognizer, or model preparation: listening
 * only collects the repository status flow, and clicks only launch the
 * UI-owned capture activity. Declared in the manifest with
 * BIND_QUICK_SETTINGS_TILE; no other permissions are used.
 */
class StilltypeTileService : TileService() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var statusJob: Job? = null

    override fun onStartListening() {
        super.onStartListening()
        statusJob?.cancel()
        statusJob = scope.launch {
            try {
                // Read-only: collecting status must not copy, download, or load models.
                val models = (application as? StilltypeApplication)?.models ?: return@launch
                models.status.collect { updateTile(it) }
            } catch (_: Exception) {
                // SystemUI must never crash because of this tile.
            }
        }
    }

    override fun onStopListening() {
        statusJob?.cancel()
        statusJob = null
        super.onStopListening()
    }

    override fun onDestroy() {
        statusJob?.cancel()
        statusJob = null
        scope.cancel()
        super.onDestroy()
    }

    override fun onClick() {
        super.onClick()
        if (isLocked) {
            try {
                unlockAndRun { launchCapture() }
            } catch (_: Exception) {
                // Stay silent; the user can open the app manually.
            }
        } else {
            launchCapture()
        }
    }

    @Suppress("DEPRECATION") // Intent overload: only used below API 34.
    private fun launchCapture() {
        try {
            val intent = captureIntent(this)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                val pending = PendingIntent.getActivity(
                    this,
                    0,
                    intent,
                    PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
                )
                startActivityAndCollapse(pending)
            } else {
                startActivityAndCollapse(intent)
            }
        } catch (_: Exception) {
            // SystemUI must never crash because of this tile.
        }
    }

    private fun updateTile(status: ModelStatus) {
        try {
            val tile = qsTile ?: return
            val ready = status.installed && status.error == null
            tile.label = TILE_LABEL
            tile.subtitle = subtitleFor(status)
            tile.state = if (ready) Tile.STATE_ACTIVE else Tile.STATE_INACTIVE
            tile.contentDescription = "$TILE_LABEL: ${tile.subtitle}"
            tile.updateTile()
        } catch (_: Exception) {
            // Keep the manifest label/state rather than crashing SystemUI.
        }
    }

    private fun subtitleFor(status: ModelStatus): String = when {
        status.error != null -> "Model unavailable"
        status.preparing ->
            if (status.progress > 0f) "Preparing ${(status.progress * 100).toInt()}%"
            else status.message.ifBlank { "Preparing model" }
        status.installed -> "Ready to dictate"
        status.bundled -> "Model bundled"
        else -> status.message.ifBlank { "Model not installed" }
    }

    private companion object {
        const val TILE_LABEL = "Stilltype"
    }
}
