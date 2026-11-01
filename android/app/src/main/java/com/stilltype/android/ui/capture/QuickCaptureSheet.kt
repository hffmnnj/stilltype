package com.stilltype.android.ui.capture

import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.stilltype.android.core.MAX_SESSION_SECONDS
import com.stilltype.android.ui.AppViewModel
import com.stilltype.android.ui.Sharing
import com.stilltype.android.ui.components.BusyRow
import com.stilltype.android.ui.components.PrimaryButton
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.SeamMark
import com.stilltype.android.ui.components.SecondaryButton
import com.stilltype.android.ui.screens.StageButton
import com.stilltype.android.ui.screens.StageMessage
import com.stilltype.android.ui.screens.isChecking
import com.stilltype.android.ui.screens.modelHeadline
import com.stilltype.android.ui.theme.StilltypeTheme
import com.stilltype.android.ui.theme.WordmarkStyle

/**
 * Compact capture sheet for the Quick Settings tile and pinned shortcut.
 * Opening the sheet never starts the camera; recording requires a tap.
 */
@Composable
fun QuickCaptureSheet(
    app: AppViewModel,
    capture: CaptureViewModel,
    onFinish: () -> Unit,
    onOpenModels: () -> Unit,
) {
    val context = LocalContext.current
    val colors = StilltypeTheme.colors
    val state by capture.state.collectAsStateWithLifecycle()
    val live by capture.live.collectAsStateWithLifecycle()
    val data by app.data.collectAsStateWithLifecycle()
    val model by app.modelStatus.collectAsStateWithLifecycle()
    val loading by app.loading.collectAsStateWithLifecycle()
    var resumed by remember { mutableStateOf(false) }
    val access = rememberCameraAccess { capture.start() }

    // ON_PAUSE releases the camera so it is never shared with the screen behind the sheet.
    LifecycleResumeEffect(capture) {
        resumed = true
        onPauseOrDispose {
            resumed = false
            capture.onPaused()
        }
    }

    fun close() {
        capture.cancel()
        onFinish()
    }
    BackHandler { close() }

    val busy = state.phase != CapturePhase.IDLE
    val ready = model.installed && !loading
    val result = state.result
    val maxSecs = data.settings.maxClipSeconds.coerceIn(3, MAX_SESSION_SECONDS)
    val limitLabel = if (maxSecs >= MAX_SESSION_SECONDS) "Up to 2:00" else "Up to $maxSecs seconds"

    Box(Modifier.fillMaxSize()) {
        Box(
            Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.32f))
                .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null, onClickLabel = "Close", role = Role.Button) { close() },
        )
        Surface(
            modifier = Modifier.align(Alignment.BottomCenter).widthIn(max = 560.dp).fillMaxWidth(),
            shape = RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp),
            color = MaterialTheme.colorScheme.background,
            shadowElevation = 8.dp,
        ) {
            Column(
                Modifier.navigationBarsPadding().imePadding().padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    SeamMark(size = 24.dp)
                    Text("Stilltype", style = WordmarkStyle.copy(fontSize = 20.sp), modifier = Modifier.padding(start = 8.dp).weight(1f))
                    IconButton(onClick = ::close) { Icon(Icons.Outlined.Close, contentDescription = "Close") }
                }

                when {
                    loading || isChecking(model) -> BusyRow("Checking the model")
                    !ready -> {
                        val (title, detail) = modelHeadline(model)
                        Text(title, style = MaterialTheme.typography.titleMedium)
                        Text(detail, style = MaterialTheme.typography.bodyMedium, color = colors.muted)
                        if (model.bundled) PrimaryButton("Open model setup", onOpenModels, modifier = Modifier.fillMaxWidth())
                    }
                    result != null && !busy -> {
                        TranscriptPanel(
                            result,
                            onEdit = capture::editResult,
                            onCopy = { capture.copyResult(); capture.flushEdit(); Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show(); onFinish() },
                            onShare = {
                                capture.flushEdit()
                                if (Sharing.shareText(context, result.text)) onFinish() else Toast.makeText(context, "No app can share this", Toast.LENGTH_SHORT).show()
                            },
                            minLines = 2,
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                            PrimaryButton("Done", { capture.flushEdit(); onFinish() }, modifier = Modifier.weight(1f))
                            SecondaryButton("Record again", { capture.dismissResult(); capture.start() }, modifier = Modifier.weight(1f))
                        }
                    }
                    else -> {
                        CaptureStage(
                            cameraActive = state.cameraOn && resumed,
                            mirrored = data.settings.mirrored,
                            sink = capture,
                            state = state,
                            live = live,
                        ) {
                            if (!access.granted) {
                                StageMessage("Camera access needed", "Stilltype reads lip movement from the camera. Nothing is stored as video.")
                                StageButton(if (access.blocked) "Open app settings" else "Allow camera") { access.request() }
                            } else {
                                StageMessage("Ready", "Press record, then mouth your words.")
                            }
                        }
                        state.failure?.let { FailureNotice(it, onDismiss = capture::clearFailure) }
                        if (state.phase == CapturePhase.PROCESSING) BusyRow("Processing on this device")
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            RecordButton(
                                state.phase,
                                onRecord = { if (access.granted) capture.start() else access.request() },
                                onStop = capture::stop,
                            )
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
                                    else limitLabel,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = colors.muted,
                                )
                            }
                            if (busy && state.phase != CapturePhase.PROCESSING) QuietButton("Cancel", { close() })
                        }
                    }
                }
            }
        }
    }
}
