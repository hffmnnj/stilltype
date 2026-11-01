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
    !status.bundled && !status.installed && !status.preparing && status.error == null && status.message == "Checking model"

/** Headline and detail for the model state shared by Dictate and Models. */
fun modelHeadline(status: ModelStatus): Pair<String, String> = when {
    status.installed -> "Model ready" to "Verified and prepared on this device."
    status.preparing -> "Preparing model" to status.message
    status.error != null -> "Model setup failed" to (status.error ?: status.message)
    isChecking(status) -> "Checking model" to "Verifying the bundled files."
    !status.bundled -> "Model not bundled" to "This build has no recognition model. Install the full build to record."
    else -> "Model not set up" to "Prepare the bundled model once to start recording."
}
