package com.stilltype.android.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.calculateEndPadding
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.dp
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.RecognitionMode
import com.stilltype.android.ui.theme.StilltypeTheme

/** Width above which Dictate lays out side by side and navigation becomes a rail. */
const val WIDE_DP = 840
const val RAIL_DP = 600

@Composable
fun isWide(): Boolean = LocalConfiguration.current.screenWidthDp >= WIDE_DP

/** Scrolling, width-capped content that respects the scaffold's insets. */
@Composable
fun ScreenColumn(
    padding: PaddingValues,
    modifier: Modifier = Modifier,
    maxWidth: Int = 680,
    content: @Composable () -> Unit,
) {
    val dir = LocalLayoutDirection.current
    Box(
        modifier
            .fillMaxSize()
            .padding(
                start = padding.calculateStartPadding(dir),
                end = padding.calculateEndPadding(dir),
                top = padding.calculateTopPadding(),
                bottom = padding.calculateBottomPadding(),
            ),
        contentAlignment = Alignment.TopCenter,
    ) {
        Column(
            Modifier
                .widthIn(max = maxWidth.dp)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) { content() }
    }
}

/** One label/value row separated by a quiet rule from its neighbours. */
@Composable
fun KeyValueRow(label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier.fillMaxWidth().padding(vertical = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = StilltypeTheme.colors.muted, modifier = Modifier.weight(0.4f))
        Text(value, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(0.6f))
    }
}

/** The repository's initial status, before it has resolved the bundled assets. */
fun isChecking(status: ModelStatus): Boolean =
    !status.bundled && !status.installed && !status.preparing && status.error == null && status.message.startsWith("Checking")

fun RecognitionMode.label(): String = if (this == RecognitionMode.QUALITY) "Quality" else "Fast"

/** Headline and detail for the optional Quality pack, which has its own status. */
fun qualityHeadline(status: ModelStatus): Pair<String, String> = when {
    // A fixed hardware limit outranks every setup or download state.
    status.unavailableReason != null -> "Quality unavailable" to (status.unavailableReason ?: status.message)
    status.installed -> "Quality model ready" to "Verified and prepared on this device."
    status.preparing -> "Preparing Quality model" to status.message
    status.error != null -> "Quality setup failed" to (status.error ?: status.message)
    isChecking(status) -> "Checking Quality model" to "Verifying the bundled files."
    !status.bundled -> "Quality model not bundled" to "This build has no Quality model."
    else -> "Quality model not set up" to "Prepare the optional pack once to use Quality."
}

/**
 * Whether a recording in [mode] can start. Fast is always required because it
 * runs the unchanged quiet/activity gate; Quality additionally needs its pack.
 */
data class CaptureReadiness(
    val ready: Boolean,
    val headline: String,
    val detail: String,
    /** True when the selected Quality mode is blocked by hardware, so an explicit "Use Fast" resolves it. */
    val offerFast: Boolean = false,
)

fun captureReadiness(mode: RecognitionMode, fast: ModelStatus, quality: ModelStatus, loading: Boolean): CaptureReadiness = when {
    loading -> CaptureReadiness(false, "Loading", "")
    mode == RecognitionMode.QUALITY && quality.unavailableReason != null ->
        qualityHeadline(quality).let { CaptureReadiness(false, it.first, it.second, offerFast = true) }
    !fast.installed -> modelHeadline(fast).let { CaptureReadiness(false, it.first, it.second) }
    mode == RecognitionMode.QUALITY && !quality.installed ->
        qualityHeadline(quality).let { CaptureReadiness(false, it.first, it.second) }
    else -> CaptureReadiness(true, "${mode.label()} ready", "")
}

/** Headline and detail for the model state shared by Dictate and Models. */
fun modelHeadline(status: ModelStatus): Pair<String, String> = when {
    status.installed -> "Model ready" to "Verified and prepared on this device."
    status.preparing -> "Preparing model" to status.message
    status.error != null -> "Model setup failed" to (status.error ?: status.message)
    isChecking(status) -> "Checking model" to "Verifying the bundled files."
    !status.bundled -> "Model not bundled" to "This build has no recognition model. Install the full build to record."
    else -> "Model not set up" to "Prepare the bundled model once to start recording."
}
