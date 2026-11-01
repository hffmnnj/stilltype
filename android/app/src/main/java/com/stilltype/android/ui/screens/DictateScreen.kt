package com.stilltype.android.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.VideocamOff
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.size
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.stilltype.android.core.MAX_SESSION_SECONDS
import com.stilltype.android.core.RecognitionMode
import com.stilltype.android.ui.AppViewModel
import com.stilltype.android.ui.Sharing
import com.stilltype.android.ui.capture.CapturePhase
import com.stilltype.android.ui.capture.CaptureStage
import com.stilltype.android.ui.capture.CaptureViewModel
import com.stilltype.android.ui.capture.FailureKind
import com.stilltype.android.ui.capture.FailureNotice
import com.stilltype.android.ui.capture.RecordButton
import com.stilltype.android.ui.capture.TranscriptPanel
import com.stilltype.android.ui.capture.formatClock
import com.stilltype.android.ui.capture.rememberCameraAccess
import com.stilltype.android.ui.capture.ProcessingStatus
import com.stilltype.android.ui.components.Panel
import com.stilltype.android.ui.components.PrimaryButton
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.StatusChip
import com.stilltype.android.ui.theme.StilltypeTheme

private enum class Pending { RECORD, PREVIEW }

@Composable
fun DictateScreen(
    padding: PaddingValues,
    app: AppViewModel,
    capture: CaptureViewModel,
    resumed: Boolean,
    onOpenModels: () -> Unit,
    notify: (String) -> Unit,
) {
    val context = LocalContext.current
    val colors = StilltypeTheme.colors
    val state by capture.state.collectAsStateWithLifecycle()
    val live by capture.live.collectAsStateWithLifecycle()
    val data by app.data.collectAsStateWithLifecycle()
    val model by app.modelStatus.collectAsStateWithLifecycle()
    val quality by app.qualityModelStatus.collectAsStateWithLifecycle()
    val loading by app.loading.collectAsStateWithLifecycle()
    var pending by remember { mutableStateOf(Pending.RECORD) }
    val access = rememberCameraAccess {
        if (pending == Pending.RECORD) capture.start() else capture.setPreview(true)
    }

    val busy = state.phase != CapturePhase.IDLE
    // A running session keeps the mode it started with, whatever Settings says now.
    val mode = if (busy) state.mode else data.settings.recognitionMode
    val readiness = captureReadiness(mode, model, quality, loading)
    val ready = readiness.ready
    val cameraActive = state.cameraOn && resumed
    val wide = isWide()
    val maxSecs = data.settings.maxClipSeconds.coerceIn(3, MAX_SESSION_SECONDS)
    val limitLabel = if (maxSecs >= MAX_SESSION_SECONDS) "Up to 2:00" else "Up to $maxSecs seconds"

    fun record() {
        pending = Pending.RECORD
        if (access.granted) capture.start() else access.request()
    }

    fun preview() {
        pending = Pending.PREVIEW
        if (access.granted) capture.setPreview(true) else access.request()
    }

    val stage: @Composable () -> Unit = {
        CaptureStage(
            cameraActive = cameraActive,
            mirrored = data.settings.mirrored,
            sink = capture,
            state = state,
            live = live,
        ) {
            when {
                !ready -> StageMessage(
                    if (loading) "Loading" else readiness.headline,
                    if (readiness.offerFast) readiness.detail
                    else "Recording stays off until ${mode.label()} recognition is ready on this device.",
                )
                !access.granted && (access.denied || access.blocked) -> {
                    StageMessage("Camera access is off", "Stilltype reads lip movement from the camera. Nothing is stored as video.")
                    StageButton(if (access.blocked) "Open app settings" else "Allow camera") { access.request() }
                }
                else -> {
                    StageMessage("Camera is off", "It opens only when you record or check framing.")
                    StageButton("Check framing") { preview() }
                }
            }
        }
    }

    val controls: @Composable () -> Unit = {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                StatusChip(
                    if (ready) "${mode.label()} ready" else readiness.headline,
                    tone = if (ready) colors.success else colors.muted,
                )
                StatusChip(if (cameraActive) "Camera on" else "Camera off")
            }
            if (!ready && readiness.offerFast) {
                PrimaryButton(
                    "Use Fast",
                    { app.updateSettings(notify) { it.copy(recognitionMode = RecognitionMode.FAST) } },
                    modifier = Modifier.fillMaxWidth(),
                )
            } else if (!ready) {
                PrimaryButton(
                    if (loading) "Loading" else "Open model setup",
                    onOpenModels,
                    enabled = !loading,
                    modifier = Modifier.fillMaxWidth(),
                )
            } else {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    RecordButton(state.phase, onRecord = ::record, onStop = capture::stop)
                    Column(Modifier.weight(1f)) {
                        Text(
                            when (state.phase) {
                                CapturePhase.IDLE -> "Record"
                                CapturePhase.STARTING -> "Opening camera"
                                CapturePhase.RECORDING -> "Recording"
                                CapturePhase.PROCESSING -> "Reading lip movement"
                            },
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Text(
                            if (state.phase == CapturePhase.RECORDING) "${formatClock(state.elapsedMs)} of ${formatClock(state.limitMs)}"
                            else "${mode.label()} · $limitLabel",
                            style = MaterialTheme.typography.bodySmall,
                            color = colors.muted,
                        )
                    }
                    if (state.cameraOn && !busy) QuietButton("Camera off", { capture.setPreview(false) })
                    if (busy && state.phase != CapturePhase.PROCESSING) QuietButton("Cancel", capture::cancel)
                }
            }
            if (state.phase == CapturePhase.PROCESSING) ProcessingStatus(state, onCancel = capture::cancel)
            state.failure?.let { failure ->
                FailureNotice(failure, onDismiss = capture::clearFailure)
                if (failure.kind == FailureKind.MODEL_MISSING) PrimaryButton("Open model setup", onOpenModels)
            }
        }
    }

    val transcript: @Composable () -> Unit = {
        val result = state.result
        if (result != null) {
            TranscriptPanel(
                result,
                onEdit = capture::editResult,
                onCopy = { notify(if (capture.copyResult()) "Copied" else "Could not copy") },
                onShare = { Sharing.shareText(context, result.text).also { if (!it) notify("No app can share this") } },
            )
        } else {
            Panel(Modifier.fillMaxWidth()) {
                Text(
                    "Your transcript appears here after a recording.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.muted,
                    modifier = Modifier.padding(16.dp),
                )
            }
        }
    }

    ScreenColumn(padding, maxWidth = if (wide) 1040 else 680) {
        if (wide) {
            Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                Column(Modifier.weight(1.1f)) { stage() }
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    controls()
                    transcript()
                }
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                stage()
                controls()
                transcript()
            }
        }
    }
}

@Composable
internal fun StageMessage(title: String, detail: String) {
    val colors = StilltypeTheme.colors
    Icon(Icons.Outlined.VideocamOff, contentDescription = null, tint = colors.onStage.copy(alpha = 0.7f), modifier = Modifier.size(24.dp))
    Text(title, color = colors.onStage, style = MaterialTheme.typography.titleSmall, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 8.dp))
    Text(detail, color = colors.onStage.copy(alpha = 0.72f), style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 2.dp, bottom = 8.dp))
}

@Composable
internal fun StageButton(text: String, onClick: () -> Unit) {
    val colors = StilltypeTheme.colors
    OutlinedButton(
        onClick = onClick,
        modifier = Modifier.heightIn(min = 48.dp),
        shape = MaterialTheme.shapes.small,
        border = BorderStroke(1.dp, colors.onStage.copy(alpha = 0.5f)),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = colors.onStage),
    ) { Text(text, style = MaterialTheme.typography.labelLarge) }
}
