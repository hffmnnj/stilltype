package com.stilltype.android.ui.theme

import android.content.Context
import android.content.ContextWrapper
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.DisposableEffect
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.remember
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.ExperimentalTextApi
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.shape.RoundedCornerShape
import com.stilltype.android.R
import com.stilltype.android.core.AppTheme

/** Roles Material3 has no slot for. Values follow DESIGN.md. */
@Immutable
data class StilltypeColors(
    val muted: Color,
    val hairline: Color,
    val border: Color,
    val focus: Color,
    val capture: Color,
    val onCapture: Color,
    val stage: Color,
    val onStage: Color,
    val success: Color,
    val warning: Color,
    val panel: Color,
)

private val LightExtras = StilltypeColors(
    muted = Color(0xFF666666),
    hairline = Color(0xFFEAEAEA),
    border = Color(0xFFD4D4D4),
    focus = Color(0xFF2563EB),
    capture = Color(0xFFB72D14),
    onCapture = Color.White,
    stage = Color(0xFF101010),
    onStage = Color(0xFFEDEDED),
    success = Color(0xFF2F6B3F),
    warning = Color(0xFF8A5A00),
    panel = Color(0xFFFAFAFA),
)

private val DarkExtras = StilltypeColors(
    muted = Color(0xFFA1A1A1),
    hairline = Color(0xFF262626),
    border = Color(0xFF404040),
    focus = Color(0xFF8AB4FF),
    capture = Color(0xFFFF8C70),
    onCapture = Color(0xFF0A0A0A),
    stage = Color(0xFF101010),
    onStage = Color(0xFFEDEDED),
    success = Color(0xFF7FC48F),
    warning = Color(0xFFE3B04B),
    panel = Color(0xFF141414),
)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF171717),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFF0F0F0),
    onPrimaryContainer = Color(0xFF171717),
    secondary = Color(0xFF171717),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFF0F0F0),
    onSecondaryContainer = Color(0xFF171717),
    background = Color.White,
    onBackground = Color(0xFF171717),
    surface = Color.White,
    onSurface = Color(0xFF171717),
    surfaceVariant = Color(0xFFFAFAFA),
    onSurfaceVariant = Color(0xFF666666),
    surfaceContainerLowest = Color.White,
    surfaceContainerLow = Color(0xFFFAFAFA),
    surfaceContainer = Color(0xFFFAFAFA),
    surfaceContainerHigh = Color(0xFFF5F5F5),
    surfaceContainerHighest = Color(0xFFF0F0F0),
    outline = Color(0xFF8F8F8F),
    outlineVariant = Color(0xFFEAEAEA),
    error = Color(0xFFB72D14),
    onError = Color.White,
    errorContainer = Color(0xFFFDEEEA),
    onErrorContainer = Color(0xFF8F2210),
    scrim = Color.Black,
)

private val DarkScheme = darkColorScheme(
    primary = Color(0xFFEDEDED),
    onPrimary = Color(0xFF0A0A0A),
    primaryContainer = Color(0xFF1F1F1F),
    onPrimaryContainer = Color(0xFFEDEDED),
    secondary = Color(0xFFEDEDED),
    onSecondary = Color(0xFF0A0A0A),
    secondaryContainer = Color(0xFF1F1F1F),
    onSecondaryContainer = Color(0xFFEDEDED),
    background = Color(0xFF0A0A0A),
    onBackground = Color(0xFFEDEDED),
    surface = Color(0xFF0A0A0A),
    onSurface = Color(0xFFEDEDED),
    surfaceVariant = Color(0xFF141414),
    onSurfaceVariant = Color(0xFFA1A1A1),
    surfaceContainerLowest = Color(0xFF0A0A0A),
    surfaceContainerLow = Color(0xFF141414),
    surfaceContainer = Color(0xFF141414),
    surfaceContainerHigh = Color(0xFF1A1A1A),
    surfaceContainerHighest = Color(0xFF1F1F1F),
    outline = Color(0xFF7A7A7A),
    outlineVariant = Color(0xFF262626),
    error = Color(0xFFFF8C70),
    onError = Color(0xFF0A0A0A),
    errorContainer = Color(0xFF3A1A12),
    onErrorContainer = Color(0xFFFFB4A3),
    scrim = Color.Black,
)

@OptIn(ExperimentalTextApi::class)
private val Sans = FontFamily(
    Font(R.font.geist_sans_variable, FontWeight.Normal, variationSettings = FontVariation.Settings(FontVariation.weight(400))),
    Font(R.font.geist_sans_variable, FontWeight.Medium, variationSettings = FontVariation.Settings(FontVariation.weight(500))),
    Font(R.font.geist_sans_variable, FontWeight.SemiBold, variationSettings = FontVariation.Settings(FontVariation.weight(600))),
)

/** Regular only: wordmark and restrained headings. */
private val Serif = FontFamily(Font(R.font.geist_serif_regular, FontWeight.Normal))

val MonoFamily: FontFamily = FontFamily.Monospace

private fun sans(size: Int, line: Int, weight: FontWeight = FontWeight.Normal, spacing: Double = 0.0) =
    TextStyle(fontFamily = Sans, fontSize = size.sp, lineHeight = line.sp, fontWeight = weight, letterSpacing = spacing.sp)

private val StilltypeTypography = Typography(
    displaySmall = TextStyle(fontFamily = Serif, fontSize = 26.sp, lineHeight = 32.sp, fontWeight = FontWeight.Normal),
    headlineMedium = TextStyle(fontFamily = Serif, fontSize = 26.sp, lineHeight = 32.sp, fontWeight = FontWeight.Normal),
    headlineSmall = sans(20, 26, FontWeight.SemiBold),
    titleLarge = sans(20, 26, FontWeight.SemiBold),
    titleMedium = sans(16, 22, FontWeight.SemiBold),
    titleSmall = sans(14, 20, FontWeight.SemiBold),
    bodyLarge = sans(16, 24),
    bodyMedium = sans(14, 21),
    bodySmall = sans(13, 18),
    labelLarge = sans(14, 20, FontWeight.Medium),
    labelMedium = sans(12, 16, FontWeight.Medium),
    labelSmall = sans(12, 16, FontWeight.Medium),
)

val WordmarkStyle: TextStyle = TextStyle(fontFamily = Serif, fontSize = 26.sp, lineHeight = 32.sp, fontWeight = FontWeight.Normal)

/** 8 dp controls, 12 dp panels, 20 dp quick-capture sheet. */
private val StilltypeShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(20.dp),
    extraLarge = RoundedCornerShape(20.dp),
)

private val LocalExtras = compositionLocalOf { LightExtras }

object StilltypeTheme {
    val colors: StilltypeColors
        @Composable @ReadOnlyComposable get() = LocalExtras.current
}

@Composable
fun StilltypeTheme(theme: AppTheme, content: @Composable () -> Unit) {
    val dark = when (theme) {
        AppTheme.SYSTEM -> isSystemInDarkTheme()
        AppTheme.LIGHT -> false
        AppTheme.DARK -> true
    }
    val scheme: ColorScheme = if (dark) DarkScheme else LightScheme
    val activity = LocalContext.current.findComponentActivity()
    DisposableEffect(activity, dark) {
        // Bar icon contrast must follow the in-app theme, not only the system one.
        val style = SystemBarStyle.auto(android.graphics.Color.TRANSPARENT, android.graphics.Color.TRANSPARENT) { dark }
        activity?.enableEdgeToEdge(statusBarStyle = style, navigationBarStyle = style)
        onDispose {}
    }
    CompositionLocalProvider(LocalExtras provides if (dark) DarkExtras else LightExtras) {
        MaterialTheme(colorScheme = scheme, typography = StilltypeTypography, shapes = StilltypeShapes, content = content)
    }
}

private tailrec fun Context.findComponentActivity(): ComponentActivity? = when (this) {
    is ComponentActivity -> this
    is ContextWrapper -> baseContext.findComponentActivity()
    else -> null
}

@Composable
fun isDarkTheme(): Boolean = MaterialTheme.colorScheme.background.red < 0.5f

/** True when the user turned system animations off. */
@Composable
fun rememberReducedMotion(): Boolean {
    val resolver = LocalContext.current.contentResolver
    return remember {
        Settings.Global.getFloat(resolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
}
