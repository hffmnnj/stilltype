package com.stilltype.android.ui.capture

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.ExpandLess
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.clickable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.material3.LinearProgressIndicator
import com.stilltype.android.core.RecognitionMode
import com.stilltype.android.ui.components.BusyRow
import com.stilltype.android.ui.components.InlineError
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.Rule
import com.stilltype.android.ui.components.StatusChip
import com.stilltype.android.ui.screens.label
import com.stilltype.android.ui.theme.MonoFamily
import com.stilltype.android.ui.theme.StilltypeTheme

fun formatClock(ms: Long): String {
    val total = (ms / 1000).toInt()
    return "%02d:%02d".format(total / 60, total % 60)
}

/**
 * Dark camera stage. Shows the real preview when [cameraActive], otherwise the
 * caller's [idle] content. Guides only appear over an open camera.
 */
@Composable
fun CaptureStage(
    cameraActive: Boolean,
    mirrored: Boolean,
    sink: FrameSink,
    state: CaptureState,
    live: LiveHint,
    modifier: Modifier = Modifier,
    ratio: Float = 16f / 10f,
    idle: @Composable ColumnScope.() -> Unit,
) {
    val colors = StilltypeTheme.colors
    Box(
        modifier
            .fillMaxWidth()
            .aspectRatio(ratio)
            .clip(MaterialTheme.shapes.medium)
            .background(colors.stage),
    ) {
        if (cameraActive) {
            CameraStage(active = true, mirrored = mirrored, sink = sink)
            FaceGuide(colors.onStage.copy(alpha = 0.45f))
            StageStatus(state, live)
        } else {
            CornerMarkers(colors.onStage.copy(alpha = 0.3f))
            Column(
                Modifier.fillMaxSize().padding(16.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
                content = idle,
            )
        }
    }
}

@Composable
private fun CornerMarkers(color: Color) {
    Canvas(Modifier.fillMaxSize().padding(14.dp)) {
        val arm = 22.dp.toPx()
        val w = 1.5.dp.toPx()
        fun corner(x: Float, y: Float, dx: Float, dy: Float) {
            drawLine(color, Offset(x, y), Offset(x + dx * arm, y), w, StrokeCap.Round)
            drawLine(color, Offset(x, y), Offset(x, y + dy * arm), w, StrokeCap.Round)
        }
        corner(0f, 0f, 1f, 1f)
        corner(size.width, 0f, -1f, 1f)
        corner(0f, size.height, 1f, -1f)
        corner(size.width, size.height, -1f, -1f)
    }
}

@Composable
private fun FaceGuide(color: Color) {
    Canvas(Modifier.fillMaxSize()) {
        val h = size.height * 0.78f
        val w = h * 0.72f
        val topLeft = Offset((size.width - w) / 2f, (size.height - h) / 2f)
        drawOval(color, topLeft, Size(w, h), style = Stroke(1.5.dp.toPx()))
        val y = topLeft.y + h * 0.72f
        drawLine(color, Offset(size.width / 2f - w * 0.14f, y), Offset(size.width / 2f + w * 0.14f, y), 1.5.dp.toPx(), StrokeCap.Round)
    }
}

@Composable
private fun BoxScope.StageStatus(state: CaptureState, live: LiveHint) {
    val colors = StilltypeTheme.colors
    val recording = state.phase == CapturePhase.RECORDING
    val label = when (state.phase) {
        CapturePhase.STARTING -> "Opening camera"
        CapturePhase.RECORDING -> "${formatClock(state.elapsedMs)} / ${formatClock(state.limitMs)}"
        else -> "Camera on, not recording"
    }
    Row(
        Modifier
            .align(Alignment.TopStart)
            .padding(10.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(colors.stage.copy(alpha = 0.72f))
            .padding(horizontal = 8.dp, vertical = 4.dp)
            .semantics { liveRegion = LiveRegionMode.Polite; contentDescription = label },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        if (recording) Box(Modifier.size(8.dp).clip(CircleShape).background(Color(0xFFFF8C70)))
        Text(label, color = colors.onStage, style = MaterialTheme.typography.labelMedium.copy(fontFamily = MonoFamily))
    }
    Column(
        Modifier
            .align(Alignment.BottomStart)
            .fillMaxWidth()
            .padding(10.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(colors.stage.copy(alpha = 0.72f))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        val hint = when {
            live.text.isNotBlank() -> live.text
            state.phase == CapturePhase.STARTING -> "Looking for your face"
            else -> ""
        }
        if (hint.isNotEmpty()) {
            Text(hint, color = colors.onStage, style = MaterialTheme.typography.bodySmall, maxLines = 2)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            StageChip(if (live.faceDetected) "Face found" else "No face")
            StageChip("Light ${live.lightPercent}%")
            StageChip("Motion ${live.motionPercent}%")
        }
    }
}

@Composable
private fun StageChip(text: String) {
    val colors = StilltypeTheme.colors
    Text(
        text,
        color = colors.onStage,
        style = MaterialTheme.typography.labelMedium,
        maxLines = 1,
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .border(1.dp, colors.onStage.copy(alpha = 0.25f), RoundedCornerShape(6.dp))
            .padding(horizontal = 6.dp, vertical = 2.dp),
    )
}

/** Circular record control; becomes Stop while capturing. Minimum 64 dp. */
@Composable
fun RecordButton(
    phase: CapturePhase,
    onRecord: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    size: Dp = 64.dp,
) {
    val colors = StilltypeTheme.colors
    val busy = phase == CapturePhase.PROCESSING
    val capturing = phase == CapturePhase.RECORDING || phase == CapturePhase.STARTING
    val description = when {
        busy -> "Processing"
        capturing -> "Stop recording"
        else -> "Start recording"
    }
    val fill = if (enabled && !busy) colors.capture else MaterialTheme.colorScheme.surfaceContainerHighest
    Box(
        modifier
            .size(size)
            .clip(CircleShape)
            .background(fill)
            .border(2.dp, MaterialTheme.colorScheme.background, CircleShape)
            .border(1.dp, colors.border, CircleShape)
            .clickable(enabled = enabled && !busy, role = Role.Button, onClickLabel = description) {
                if (capturing) onStop() else onRecord()
            }
            .semantics { contentDescription = description },
        contentAlignment = Alignment.Center,
    ) {
        if (capturing) {
            Box(Modifier.size(size * 0.3f).clip(RoundedCornerShape(4.dp)).background(colors.onCapture))
        } else {
            Box(Modifier.size(size * 0.3f).clip(CircleShape).background(if (enabled && !busy) colors.onCapture else colors.muted))
        }
    }
}

/** Failure beside the action that caused it. */
@Composable
fun FailureNotice(failure: CaptureFailure, onDismiss: () -> Unit, modifier: Modifier = Modifier) {
    InlineError(failure.title, failure.detail, modifier, actionLabel = "Dismiss", onAction = onDismiss)
}

/** Editable latest transcript with copy/share and inspectable raw output. */
@Composable
fun TranscriptPanel(
    result: CaptureResult,
    onEdit: (String) -> Unit,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    modifier: Modifier = Modifier,
    minLines: Int = 3,
) {
    val colors = StilltypeTheme.colors
    var showDetails by rememberSaveable { mutableStateOf(false) }
    val haptic = LocalHapticFeedback.current
    LaunchedEffect(result.id) { haptic.performHapticFeedback(HapticFeedbackType.Confirm) }

    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = result.text,
            onValueChange = onEdit,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Transcript") },
            textStyle = MaterialTheme.typography.bodyLarge,
            minLines = minLines,
            maxLines = 8,
            shape = MaterialTheme.shapes.small,
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences),
            colors = fieldColors(),
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                when {
                    result.copied -> "Copied to clipboard"
                    result.saved -> "Saved to history"
                    else -> "Not saved"
                } + if (result.copied && result.saved) " · saved" else "",
                style = MaterialTheme.typography.bodySmall,
                color = colors.muted,
                modifier = Modifier.weight(1f).padding(end = 8.dp),
            )
            IconButton(onClick = onCopy) { Icon(Icons.Outlined.ContentCopy, contentDescription = "Copy transcript") }
            IconButton(onClick = onShare) { Icon(Icons.Outlined.Share, contentDescription = "Share transcript") }
            IconButton(onClick = { showDetails = !showDetails }) {
                Icon(
                    if (showDetails) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore,
                    contentDescription = if (showDetails) "Hide recognition details" else "Show recognition details",
                )
            }
        }
        if (showDetails) {
            Rule()
            RecognitionDetails(result)
        }
    }
}

@Composable
fun RecognitionDetails(result: CaptureResult, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        DetailLine("Raw output", result.raw.ifBlank { "(empty)" }, mono = false)
        val quality = result.mode == RecognitionMode.QUALITY
        DetailLine("Mode", result.mode.label(), mono = false)
        // Quality time covers gating, preprocessing and decoding, not just native calls.
        DetailLine(if (quality) "Processing time" else "Inference", "${formatDuration(result.inferenceMs)} on ${result.backend.ifBlank { "CPU" }}")
        DetailLine("Clip", "${formatClock(result.durationMs)} · ${result.frames} mouth frames")
        Text(
            if (quality) {
                "Experimental full beam50 decoding of the 2026 FLARE Base model, exported for this device and not newly trained. Weights are CC-BY-NC-4.0: non-commercial use only."
            } else {
                "Experimental optimized CTC/INT8 export of the 2026 USR2 weights. Not equal to desktop beam-search accuracy."
            },
            style = MaterialTheme.typography.bodySmall,
            color = StilltypeTheme.colors.muted,
        )
    }
}

private fun formatDuration(ms: Long): String = if (ms >= 10_000) "%.1f s".format(ms / 1000.0) else "$ms ms"

/**
 * Processing state with a real way out. Quality shows completed model windows
 * only once the engine reports them; Fast has no per-window signal.
 */
@Composable
fun ProcessingStatus(state: CaptureState, onCancel: () -> Unit, modifier: Modifier = Modifier) {
    val colors = StilltypeTheme.colors
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val progress = state.progress
        if (state.mode == RecognitionMode.QUALITY) {
            if (progress != null && progress.totalWindows > 0) {
                val fraction = (progress.completedWindows.toFloat() / progress.totalWindows).coerceIn(0f, 1f)
                val label = "Quality: ${progress.completedWindows} of ${progress.totalWindows} windows"
                LinearProgressIndicator(
                    progress = { fraction },
                    modifier = Modifier.fillMaxWidth().semantics { contentDescription = label },
                    color = MaterialTheme.colorScheme.onSurface,
                    trackColor = colors.hairline,
                )
                Text(label, style = MaterialTheme.typography.bodySmall, color = colors.muted)
            } else {
                BusyRow("Quality: starting first window")
            }
        } else {
            BusyRow("Processing on this device")
        }
        QuietButton("Cancel", onCancel)
    }
}

@Composable
fun DetailLine(label: String, value: String, mono: Boolean = true) {
    Column {
        Text(label, style = MaterialTheme.typography.labelMedium, color = StilltypeTheme.colors.muted)
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium.let { if (mono) it.copy(fontFamily = MonoFamily) else it },
        )
    }
}

@Composable
fun fieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = StilltypeTheme.colors.focus,
    unfocusedBorderColor = StilltypeTheme.colors.border,
    focusedLabelColor = StilltypeTheme.colors.focus,
    unfocusedLabelColor = StilltypeTheme.colors.muted,
    cursorColor = StilltypeTheme.colors.focus,
)
