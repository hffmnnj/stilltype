package com.stilltype.android.ui.screens

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.KeyboardArrowRight
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.stilltype.android.core.AppSettings
import com.stilltype.android.core.AppTheme
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.MAX_SESSION_SECONDS
import com.stilltype.android.core.ExportFormat
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.RecognitionMode
import com.stilltype.android.platform.QuickAccess
import com.stilltype.android.platform.TileAddResult
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.Rule
import com.stilltype.android.ui.components.SecondaryButton
import com.stilltype.android.ui.components.SectionLabel
import com.stilltype.android.ui.components.StatusChip
import com.stilltype.android.ui.theme.StilltypeTheme

internal tailrec fun Context.unwrapActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.unwrapActivity()
    else -> null
}

@Composable
fun SettingsScreen(
    padding: PaddingValues,
    settings: AppSettings,
    historyCount: Int,
    model: ModelStatus,
    quality: ModelStatus,
    onChange: ((AppSettings) -> AppSettings) -> Unit,
    onExport: (ExportFormat) -> Unit,
    onClearHistory: () -> Unit,
    onOpenModels: () -> Unit,
    notify: (String) -> Unit,
) {
    val context = LocalContext.current
    val colors = StilltypeTheme.colors
    var confirmClear by remember { mutableStateOf(false) }
    var tileStatus by remember { mutableStateOf<String?>(null) }
    var pinStatus by remember { mutableStateOf<String?>(null) }

    ScreenColumn(padding) {
        SectionLabel("Appearance", Modifier.padding(top = 8.dp))
        Choice(
            "Theme",
            AppTheme.entries,
            settings.theme,
            { when (it) { AppTheme.SYSTEM -> "System"; AppTheme.LIGHT -> "Light"; AppTheme.DARK -> "Dark" } },
        ) { value -> onChange { it.copy(theme = value) } }
        Rule()
        SwitchRow("Mirror preview", "Show the front camera like a mirror. Recognition always sees the unmirrored image.", settings.mirrored) { v ->
            onChange { it.copy(mirrored = v) }
        }

        SectionLabel("Recording")
        // Quality also needs Fast: Fast runs the unchanged quiet/activity veto.
        val qualityAvailable = model.installed && quality.installed
        Choice(
            "Recognition",
            RecognitionMode.entries,
            settings.recognitionMode,
            { it.label() },
            optionEnabled = { it == RecognitionMode.FAST || qualityAvailable },
        ) { value -> onChange { it.copy(recognitionMode = value) } }
        Text(
            if (settings.recognitionMode == RecognitionMode.QUALITY) {
                "Quality is experimental: full beam50 search with non-commercial weights. It takes longer and uses more memory than Fast."
            } else {
                "Fast uses the optimized CTC model. It is the quickest and lightest option."
            },
            style = MaterialTheme.typography.bodySmall,
            color = colors.muted,
            modifier = Modifier.padding(bottom = 8.dp),
        )
        if (quality.unavailableReason != null) {
            Text(
                quality.unavailableReason ?: "",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(bottom = 8.dp),
            )
            if (settings.recognitionMode == RecognitionMode.QUALITY) {
                SecondaryButton(
                    "Use Fast",
                    { onChange { it.copy(recognitionMode = RecognitionMode.FAST) } },
                    modifier = Modifier.padding(bottom = 8.dp),
                )
            }
        } else if (!qualityAvailable) {
            Row(
                Modifier.fillMaxWidth().padding(bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(
                    if (model.installed) qualityHeadline(quality).first else modelHeadline(model).first,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f),
                )
                SecondaryButton("Set up", onOpenModels)
            }
        }
        Rule()
        Choice(
            "Compute",
            ComputeMode.entries,
            settings.computeMode,
            { if (it == ComputeMode.AUTO) "Auto" else "CPU" },
        ) { value -> onChange { it.copy(computeMode = value) } }
        Text(
            if (settings.computeMode == ComputeMode.AUTO) "Auto asks the device's NNAPI to run the model and falls back to CPU. Each result names the backend it used."
            else "CPU only. Slower, and the most predictable.",
            style = MaterialTheme.typography.bodySmall,
            color = colors.muted,
            modifier = Modifier.padding(bottom = 8.dp),
        )
        Rule()
        // Commit on release so dragging does not rewrite the store on every frame.
        // Continuous whole seconds across 3..120 so the two-minute limit stays reachable.
        var clip by remember(settings.maxClipSeconds) { mutableFloatStateOf(settings.maxClipSeconds.toFloat()) }
        val clipLabel = if (clip.toInt() >= MAX_SESSION_SECONDS) "2:00" else "${clip.toInt()} s"
        Column(Modifier.padding(vertical = 8.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Recording limit", style = MaterialTheme.typography.bodyLarge)
                Text(clipLabel, style = MaterialTheme.typography.bodyLarge)
            }
            Slider(
                value = clip,
                onValueChange = { clip = it },
                onValueChangeFinished = { onChange { it.copy(maxClipSeconds = clip.toInt().coerceIn(3, MAX_SESSION_SECONDS)) } },
                valueRange = 3f..MAX_SESSION_SECONDS.toFloat(),
                steps = 0,
                colors = SliderDefaults.colors(inactiveTrackColor = colors.border),
                modifier = Modifier.semantics { contentDescription = "Recording limit, $clipLabel" },
            )
        }
        Rule()
        SwitchRow("Copy result automatically", "Copies each finished transcript. Nothing is pasted into other apps.", settings.autoCopy) { v ->
            onChange { it.copy(autoCopy = v) }
        }

        SectionLabel("History")
        SwitchRow("Keep history", "Stored on this device without extra encryption.", settings.keepHistory) { v ->
            onChange { it.copy(keepHistory = v) }
        }
        Rule()
        Choice("Keep for", listOf(7, 30, 90, 365), settings.retentionDays, { if (it == 365) "1 year" else "$it days" }, enabled = settings.keepHistory) { v ->
            onChange { it.copy(retentionDays = v) }
        }
        Rule()
        Text("Export", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 12.dp, bottom = 8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ExportFormat.entries.forEach { format ->
                SecondaryButton(format.name, { onExport(format) }, enabled = historyCount > 0)
            }
        }
        if (historyCount == 0) {
            Text("Nothing to export yet.", style = MaterialTheme.typography.bodySmall, color = colors.muted, modifier = Modifier.padding(top = 4.dp))
        }
        QuietButton("Clear history", { confirmClear = true }, enabled = historyCount > 0, destructive = true, modifier = Modifier.padding(top = 8.dp))

        SectionLabel("Quick access")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SecondaryButton("Add tile", {
                val activity = context.unwrapActivity()
                if (activity == null) {
                    tileStatus = "Could not reach the system prompt."
                } else {
                    QuickAccess.requestTile(activity) { result ->
                        tileStatus = when (result) {
                            TileAddResult.ADDED -> "Tile added to Quick Settings."
                            TileAddResult.ALREADY_ADDED -> "The tile is already in Quick Settings."
                            TileAddResult.DISMISSED -> "Not added."
                            TileAddResult.UNSUPPORTED -> "Your Android version cannot add it for you. Swipe down twice, tap the edit pencil and drag Stilltype into your tiles."
                            TileAddResult.ERROR -> "The system could not add the tile. Add it from the Quick Settings edit screen."
                        }
                    }
                }
            })
            SecondaryButton("Pin shortcut", {
                pinStatus = if (QuickAccess.requestPinnedShortcut(context)) {
                    "Confirm the shortcut in your launcher's prompt."
                } else {
                    "This launcher does not support pinned shortcuts."
                }
            })
        }
        (tileStatus ?: pinStatus)?.let {
            Text(
                listOfNotNull(tileStatus, pinStatus).joinToString(" "),
                style = MaterialTheme.typography.bodySmall,
                color = colors.muted,
                modifier = Modifier.padding(top = 8.dp).semantics { contentDescription = it },
            )
        }
        Text(
            "Both open a compact capture sheet over whatever you were doing.",
            style = MaterialTheme.typography.bodySmall,
            color = colors.muted,
            modifier = Modifier.padding(top = 4.dp),
        )

        SectionLabel("Models")
        ModelRow("Fast model", modelHeadline(model).first, model.installed, onOpenModels)
        Rule()
        ModelRow("Quality model", qualityHeadline(quality).first, quality.installed, onOpenModels)

        SectionLabel("Privacy")
        Text(
            "Stilltype uses the camera only while you record. It has no microphone access, no network access, no accessibility service and no draw-over-apps permission. Video is never saved.",
            style = MaterialTheme.typography.bodyMedium,
            color = colors.muted,
            modifier = Modifier.padding(bottom = 24.dp),
        )
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            shape = MaterialTheme.shapes.large,
            title = { Text("Clear all history?", style = MaterialTheme.typography.titleMedium) },
            text = { Text("This deletes $historyCount saved transcripts from this device. It cannot be undone.", style = MaterialTheme.typography.bodyMedium) },
            confirmButton = { QuietButton("Clear history", { confirmClear = false; onClearHistory() }, destructive = true) },
            dismissButton = { QuietButton("Cancel", { confirmClear = false }) },
        )
    }
}

@Composable
private fun ModelRow(title: String, status: String, ready: Boolean, onOpen: () -> Unit) {
    val colors = StilltypeTheme.colors
    Row(
        Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(onClickLabel = "Open models", onClick = onOpen),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(status, style = MaterialTheme.typography.bodySmall, color = colors.muted)
        }
        if (ready) StatusChip("Ready", tone = colors.success)
        Icon(Icons.AutoMirrored.Outlined.KeyboardArrowRight, contentDescription = null, tint = colors.muted)
    }
}

@Composable
private fun SwitchRow(title: String, detail: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().heightIn(min = 56.dp).toggleable(value = checked, role = Role.Switch, onValueChange = onChange).padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(detail, style = MaterialTheme.typography.bodySmall, color = StilltypeTheme.colors.muted)
        }
        Switch(checked = checked, onCheckedChange = null)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun <T> Choice(
    label: String,
    options: List<T>,
    selected: T,
    name: (T) -> String,
    enabled: Boolean = true,
    optionEnabled: (T) -> Boolean = { true },
    onSelect: (T) -> Unit,
) {
    Column(Modifier.padding(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            options.forEachIndexed { index, option ->
                SegmentedButton(
                    selected = option == selected,
                    onClick = { onSelect(option) },
                    enabled = enabled && optionEnabled(option),
                    shape = SegmentedButtonDefaults.itemShape(index, options.size),
                    modifier = Modifier.heightIn(min = 48.dp),
                    label = { Text(name(option), style = MaterialTheme.typography.labelLarge, maxLines = 1) },
                )
            }
        }
    }
}
