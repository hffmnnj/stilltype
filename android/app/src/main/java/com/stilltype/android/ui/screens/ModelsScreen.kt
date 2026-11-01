package com.stilltype.android.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material3.Checkbox
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.ui.components.BusyRow
import com.stilltype.android.ui.components.InlineError
import com.stilltype.android.ui.components.Panel
import com.stilltype.android.ui.components.PrimaryButton
import com.stilltype.android.ui.components.Rule
import com.stilltype.android.ui.components.SecondaryButton
import com.stilltype.android.ui.components.StatusChip
import com.stilltype.android.ui.theme.StilltypeTheme

@Composable
fun ModelsScreen(
    padding: PaddingValues,
    status: ModelStatus,
    compute: ComputeMode,
    onPrepare: () -> Unit,
    onOpenSettings: () -> Unit,
) {
    val colors = StilltypeTheme.colors
    var consent by rememberSaveable { mutableStateOf(false) }
    val (headline, detail) = modelHeadline(status)
    val canPrepare = status.bundled && !status.installed && !status.preparing

    ScreenColumn(padding) {
        Panel(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(headline, style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                    if (status.installed) StatusChip("Ready", tone = colors.success)
                }
                Text(detail, style = MaterialTheme.typography.bodyMedium, color = colors.muted)

                if (status.preparing) {
                    val progress = status.progress.coerceIn(0f, 1f)
                    if (status.totalBytes > 0) {
                        LinearProgressIndicator(
                            progress = { progress },
                            modifier = Modifier.fillMaxWidth().semantics { contentDescription = "Preparing model, ${(progress * 100).toInt()} percent" },
                            color = MaterialTheme.colorScheme.onSurface,
                            trackColor = colors.hairline,
                        )
                        Text("${(progress * 100).toInt()}% of ${formatBytes(status.totalBytes)}", style = MaterialTheme.typography.bodySmall, color = colors.muted)
                    } else {
                        BusyRow(status.message)
                    }
                }

                if (status.error != null && status.bundled) {
                    InlineError("Could not prepare the model", status.error)
                }

                if (canPrepare) {
                    Rule()
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .heightIn(min = 48.dp)
                            .toggleable(value = consent, role = Role.Checkbox, onValueChange = { consent = it }),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Checkbox(checked = consent, onCheckedChange = null)
                        Text(
                            "I accept the research and checkpoint terms that apply to these weights. They are separate from Stilltype's MIT license.",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                    PrimaryButton(
                        if (status.error != null) "Try again" else "Prepare model",
                        onPrepare,
                        enabled = consent,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }

        Column(Modifier.padding(top = 24.dp)) {
            Text("About this model", style = MaterialTheme.typography.titleSmall)
            Rule(Modifier.padding(top = 8.dp))
            KeyValueRow("Weights", "2026 USR2, the same checkpoint as desktop")
            Rule()
            KeyValueRow("Export", "Optimized CTC/INT8 for phones. Not newly trained.")
            Rule()
            KeyValueRow("Accuracy", "Experimental. Below desktop beam-search quality.")
            Rule()
            KeyValueRow("Size", if (status.totalBytes > 0) formatBytes(status.totalBytes) else "About 170 MB")
            Rule()
            KeyValueRow("Compute", if (compute == ComputeMode.AUTO) "Auto: device NNAPI, CPU fallback" else "CPU only")
            Rule()
            KeyValueRow("Network", "None. Files are copied from the app package.")
        }
        Text(
            "The backend that ran each result is shown in its details. NNAPI availability alone does not mean the GPU was used.",
            style = MaterialTheme.typography.bodySmall,
            color = colors.muted,
            modifier = Modifier.padding(top = 12.dp),
        )
        SecondaryButton("Compute settings", onOpenSettings, modifier = Modifier.padding(top = 16.dp))
    }
}

fun formatBytes(bytes: Long): String = when {
    bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1e9)
    bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1e6)
    else -> "%.0f KB".format(bytes / 1e3)
}
