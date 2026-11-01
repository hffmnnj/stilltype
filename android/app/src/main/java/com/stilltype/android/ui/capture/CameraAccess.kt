package com.stilltype.android.ui.capture

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Stable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect

/** Runtime camera permission state. The system dialog is only ever raised by [request]. */
@Stable
class CameraAccess internal constructor(
    granted: Boolean,
    private val launch: () -> Unit,
    private val settings: () -> Unit,
) {
    var granted by mutableStateOf(granted)
        internal set

    /** The user declined, and Android will not show the dialog again. */
    var blocked by mutableStateOf(false)
        internal set

    var denied by mutableStateOf(false)
        internal set

    fun request() = if (blocked) settings() else launch()
    fun openSettings() = settings()
}

@Composable
fun rememberCameraAccess(onGranted: () -> Unit): CameraAccess {
    val context = LocalContext.current
    val granted = { ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED }
    val onGrantedNow = rememberUpdatedState(onGranted)
    val holder = remember { arrayOfNulls<CameraAccess>(1) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        val access = holder[0] ?: return@rememberLauncherForActivityResult
        access.granted = ok
        if (ok) {
            access.blocked = false
            access.denied = false
            onGrantedNow.value()
        } else {
            val activity = context.findActivity()
            access.denied = true
            access.blocked = activity != null &&
                !ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)
        }
    }
    val access = remember {
        CameraAccess(
            granted = granted(),
            launch = { launcher.launch(Manifest.permission.CAMERA) },
            settings = {
                context.startActivity(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                )
            },
        ).also { holder[0] = it }
    }
    // The user may change the permission in system settings while we are paused.
    LifecycleResumeEffect(Unit) {
        val now = granted()
        access.granted = now
        if (now) {
            access.blocked = false
            access.denied = false
        }
        onPauseOrDispose {}
    }
    return access
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}
