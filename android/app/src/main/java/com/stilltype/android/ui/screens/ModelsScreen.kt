package com.stilltype.android.ui.screens

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.RecognitionMode
import com.stilltype.android.ui.components.BusyRow
import com.stilltype.android.ui.components.InlineError
import com.stilltype.android.ui.components.Panel
import com.stilltype.android.ui.components.PrimaryButton
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.Rule
import com.stilltype.android.ui.components.SecondaryButton
import com.stilltype.android.ui.components.StatusChip
import com.stilltype.android.ui.theme.StilltypeTheme

private const val QUALITY_TERMS_URL = "https://creativecommons.org/licenses/by-nc/4.0/"

@Composable
fun ModelsScreen(
    padding: PaddingValues,
    status: ModelStatus,
    quality: ModelStatus,
    mode: RecognitionMode,
    compute: ComputeMode,
    onPrepare: () -> Unit,
    onPrepareQuality: () -> Unit,
    onOpenSettings: () -> Unit,
    notify: (String) -> Unit,
) {
    val colors = StilltypeTheme.colors
    val context = LocalContext.current

    ScreenColumn(padding) {
        SetupPanel(
            name = "Fast model",
            selected = mode == RecognitionMode.FAST,
            status = status,
            detail = modelHeadline(status).second,
            consentKey = "fast",
            consentLabel = "I accept the research and checkpoint terms that apply to these weights. They are separate from Stilltype's MIT license.",
            onPrepare = onPrepare,
        )

        Column(Modifier.padding(top = 16.dp)) {
            SetupPanel(
                name = "Quality model",
                selected = mode == RecognitionMode.QUALITY,
                status = quality,
                detail = qualityHeadline(quality).second,
                consentKey = "quality",
                consentLabel = "I accept CC-BY-NC-4.0 for these weights: non-commercial use only. They are separate from Stilltype's MIT license.",
                onPrepare = onPrepareQuality,
                extra = {
                    QuietButton("View terms", {
                        try {
                            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(QUALITY_TERMS_URL)))
                        } catch (_: ActivityNotFoundException) {
                            notify("No browser can open the terms")
                        }
                    })
                },
            )
        }
        if (!status.installed && quality.bundled && quality.unavailableReason == null) {
            Text(
                "Quality also needs the Fast model, which runs the quiet check first.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.muted,
                modifier = Modifier.padding(top = 8.dp),
            )
        }

        Column(Modifier.padding(top = 24.dp)) {
            Text("About Fast", style = MaterialTheme.typography.titleSmall)
            Rule(Modifier.padding(top = 8.dp))
            KeyValueRow("Weights", "2026 USR2, the same checkpoint as desktop")
            Rule()
            KeyValueRow("Export", "Optimized CTC/INT8 for phones. Not newly trained.")
            Rule()
            KeyValueRow("Accuracy", "Experimental. Below desktop beam-search quality.")
            Rule()
            KeyValueRow("Size", if (status.totalBytes > 0) formatBytes(status.totalBytes) else "About 170 MB")
        }

        Column(Modifier.padding(top = 24.dp)) {
            Text("About Quality", style = MaterialTheme.typography.titleSmall)
            Rule(Modifier.padding(top = 8.dp))
            KeyValueRow("Weights", "2026 FLARE Base, full beam50 search")
            Rule()
            KeyValueRow("Export", "Exported with compact storage. Not newly trained.")
            Rule()
            KeyValueRow("License", "CC-BY-NC-4.0. Non-commercial use only.")
            Rule()
            KeyValueRow("Cost", "Experimental. Slower and uses more memory than Fast.")
            Rule()
            KeyValueRow("Memory", "Needs 8 GB of RAM. Fast works on any supported device.")
            Rule()
            KeyValueRow("Size", if (quality.totalBytes > 0) formatBytes(quality.totalBytes) else "Shown once checked")
        }

        Column(Modifier.padding(top = 24.dp)) {
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
        SecondaryButton("Recording settings", onOpenSettings, modifier = Modifier.padding(top = 16.dp, bottom = 16.dp))
    }
}

/** One model pack: real readiness or progress, its own terms checkbox, and an explicit prepare action. */
@Composable
private fun SetupPanel(
    name: String,
    selected: Boolean,
    status: ModelStatus,
    detail: String,
    consentKey: String,
    consentLabel: String,
    onPrepare: () -> Unit,
    extra: @Composable () -> Unit = {},
) {
    val colors = StilltypeTheme.colors
    var consent by rememberSaveable(consentKey) { mutableStateOf(false) }
    val unavailable = status.unavailableReason != null
    val canPrepare = status.bundled && !status.installed && !status.preparing && !unavailable

    Panel(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(name, style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                if (selected) StatusChip("Selected")
                if (unavailable) StatusChip("Unavailable", tone = colors.muted)
                if (status.installed) StatusChip("Ready", tone = colors.success)
            }
            Text(detail, style = MaterialTheme.typography.bodyMedium, color = colors.muted)

            if (status.preparing) {
                val progress = status.progress.coerceIn(0f, 1f)
                if (status.totalBytes > 0) {
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier.fillMaxWidth().semantics { contentDescription = "Preparing $name, ${(progress * 100).toInt()} percent" },
                        color = MaterialTheme.colorScheme.onSurface,
                        trackColor = colors.hairline,
                    )
                    Text("${(progress * 100).toInt()}% of ${formatBytes(status.totalBytes)}", style = MaterialTheme.typography.bodySmall, color = colors.muted)
                } else {
                    BusyRow(status.message)
                }
            }

            // The hardware explanation is already the detail line above; do not repeat it as a failure.
            if (status.error != null && status.bundled && !unavailable) {
                InlineError("Could not prepare $name", status.error)
            }
            if (unavailable) extra()

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
                    Text(consentLabel, style = MaterialTheme.typography.bodyMedium)
                }
                extra()
                PrimaryButton(
                    if (status.error != null) "Try again" else "Prepare $name",
                    onPrepare,
                    enabled = consent,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

fun formatBytes(bytes: Long): String = when {
    bytes >= 1_000_000_000 -> "%.1f GB".format(bytes / 1e9)
    bytes >= 1_000_000 -> "%.0f MB".format(bytes / 1e6)
    else -> "%.0f KB".format(bytes / 1e3)
}
