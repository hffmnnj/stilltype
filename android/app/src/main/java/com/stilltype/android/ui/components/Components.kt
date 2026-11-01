package com.stilltype.android.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ErrorOutline
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.stilltype.android.R
import com.stilltype.android.ui.theme.StilltypeTheme

/** The generated brand mark (warm vermilion mouth on near-black). Decorative raster. */
@Composable
fun SeamMark(modifier: Modifier = Modifier, size: Dp = 28.dp) {
    Image(
        painter = painterResource(R.drawable.stilltype_brand),
        contentDescription = null,
        modifier = modifier.size(size).clip(RoundedCornerShape(6.dp)),
        contentScale = ContentScale.Crop,
    )
}

@Composable
fun hairline(): Color = StilltypeTheme.colors.hairline

/** Quiet horizontal rule. */
@Composable
fun Rule(modifier: Modifier = Modifier) {
    HorizontalDivider(modifier, thickness = 1.dp, color = hairline())
}

/** 12 dp panel on the surface tone with a hairline, never nested. */
@Composable
fun Panel(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Surface(
        modifier = modifier,
        shape = MaterialTheme.shapes.medium,
        color = StilltypeTheme.colors.panel,
        border = BorderStroke(1.dp, hairline()),
        content = content,
    )
}

@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null,
) {
    Button(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        enabled = enabled,
        shape = MaterialTheme.shapes.small,
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 10.dp),
    ) { ButtonContent(text, icon) }
}

@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null,
    destructive: Boolean = false,
) {
    val content = if (destructive) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        enabled = enabled,
        shape = MaterialTheme.shapes.small,
        border = BorderStroke(1.dp, if (destructive) content else StilltypeTheme.colors.border),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = content),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 10.dp),
    ) { ButtonContent(text, icon) }
}

@Composable
fun QuietButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    destructive: Boolean = false,
) {
    TextButton(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        enabled = enabled,
        shape = MaterialTheme.shapes.small,
        colors = ButtonDefaults.textButtonColors(
            contentColor = if (destructive) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
        ),
    ) { Text(text, style = MaterialTheme.typography.labelLarge) }
}

@Composable
private fun RowScope.ButtonContent(text: String, icon: ImageVector?) {
    if (icon != null) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
        androidx.compose.foundation.layout.Spacer(Modifier.size(8.dp))
    }
    Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 2, overflow = TextOverflow.Ellipsis)
}

/** Error that stays beside the failed action; optional retry. */
@Composable
fun InlineError(
    title: String,
    detail: String?,
    modifier: Modifier = Modifier,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.errorContainer,
        contentColor = MaterialTheme.colorScheme.onErrorContainer,
    ) {
        Row(Modifier.padding(12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Icon(Icons.Outlined.ErrorOutline, contentDescription = null, modifier = Modifier.padding(top = 2.dp).size(20.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(title, style = MaterialTheme.typography.titleSmall)
                if (detail != null) Text(detail, style = MaterialTheme.typography.bodyMedium)
                if (actionLabel != null && onAction != null) {
                    TextButton(
                        onClick = onAction,
                        modifier = Modifier.heightIn(min = 48.dp),
                        colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.onErrorContainer),
                    ) { Text(actionLabel, style = MaterialTheme.typography.labelLarge) }
                }
            }
        }
    }
}

@Composable
fun BusyRow(label: String, modifier: Modifier = Modifier) {
    Row(
        modifier.semantics(mergeDescendants = true) { contentDescription = label },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp, color = MaterialTheme.colorScheme.onSurface)
        Text(label, style = MaterialTheme.typography.bodyMedium, color = StilltypeTheme.colors.muted)
    }
}

/** Compact status pill; text carries the state, color is secondary. */
@Composable
fun StatusChip(text: String, modifier: Modifier = Modifier, tone: Color = StilltypeTheme.colors.muted) {
    Box(
        modifier
            .clip(RoundedCornerShape(6.dp))
            .border(1.dp, hairline(), RoundedCornerShape(6.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp),
    ) {
        Text(text, style = MaterialTheme.typography.labelMedium, color = tone, maxLines = 1)
    }
}

@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) {
    Text(
        text,
        modifier = modifier.padding(top = 24.dp, bottom = 4.dp),
        style = MaterialTheme.typography.labelLarge,
        color = StilltypeTheme.colors.muted,
    )
}

/** Centred empty state with one optional action. */
@Composable
fun EmptyState(
    title: String,
    detail: String,
    modifier: Modifier = Modifier,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
) {
    Column(
        modifier.fillMaxWidth().padding(vertical = 48.dp, horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        SeamMark(size = 36.dp)
        Text(title, style = MaterialTheme.typography.titleMedium)
        Text(
            detail,
            style = MaterialTheme.typography.bodyMedium,
            color = StilltypeTheme.colors.muted,
            modifier = Modifier.widthIn(max = 320.dp),
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
        if (actionLabel != null && onAction != null) {
            androidx.compose.foundation.layout.Spacer(Modifier.size(8.dp))
            SecondaryButton(actionLabel, onAction)
        }
    }
}
