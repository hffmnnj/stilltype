package com.stilltype.android.ui.capture

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Camera preview plus analysis, bound only while [active] and this composable
 * is in the composition. Callers drop it on ON_PAUSE, so a translucent
 * quick-capture sheet never shares the camera with the screen behind it.
 *
 * Only this owner's use cases are unbound on disposal; the process-wide camera
 * provider and recognizer stay open for whoever binds next.
 */
@Composable
fun CameraStage(
    active: Boolean,
    mirrored: Boolean,
    sink: FrameSink,
    modifier: Modifier = Modifier,
) {
    if (!active) return
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentSink = rememberUpdatedState(sink)
    val previewView = remember {
        PreviewView(context).apply {
            // TextureView-backed so the display-only mirror transform applies.
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
            importantForAccessibility = android.view.View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
    }
    var frontFacing by remember { mutableStateOf(true) }

    DisposableEffect(lifecycleOwner) {
        val executor: ExecutorService = Executors.newSingleThreadExecutor()
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        val preview = Preview.Builder().build()
        var provider: ProcessCameraProvider? = null
        var released = false
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (released) return@addListener
            try {
                val cameraProvider = future.get()
                provider = cameraProvider
                val selector = when {
                    cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) -> CameraSelector.DEFAULT_FRONT_CAMERA
                    cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) -> {
                        frontFacing = false
                        CameraSelector.DEFAULT_BACK_CAMERA
                    }
                    else -> {
                        currentSink.value.onCameraError(currentSink.value.session, "This device has no usable camera.")
                        return@addListener
                    }
                }
                preview.surfaceProvider = previewView.surfaceProvider
                val frameGate = FrameGate()
                analysis.setAnalyzer(executor) { image ->
                    analyze(image, frameGate, currentSink.value)
                }
                cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
            } catch (e: Exception) {
                currentSink.value.onCameraError(currentSink.value.session, e.message ?: "The camera could not be opened.")
            }
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            released = true
            analysis.clearAnalyzer()
            provider?.unbind(preview, analysis)
            executor.shutdown()
        }
    }

    // PreviewView already mirrors front-camera previews; "mirrored = false" flips it back for display only.
    AndroidView(
        factory = { previewView },
        modifier = modifier.fillMaxSize().graphicsLayer { scaleX = if (frontFacing && !mirrored) -1f else 1f },
    )
}

private class FrameGate {
    var lastAcceptedMs = 0L
}

/** Samples at most 25 fps on a monotonic clock and always closes the proxy. */
private fun analyze(image: ImageProxy, gate: FrameGate, sink: FrameSink) {
    try {
        val session = sink.session
        val now = SystemClock.elapsedRealtime()
        if (session == 0L || now - gate.lastAcceptedMs < MIN_FRAME_INTERVAL_MS) return
        gate.lastAcceptedMs = now
        val upright = image.toUprightBitmap()
        try {
            sink.onFrame(session, upright, now)
        } finally {
            upright.recycle()
        }
    } catch (e: Exception) {
        sink.onCameraError(sink.session, e.message ?: "A camera frame could not be read.")
    } finally {
        image.close()
    }
}

/** Rotated to upright and never mirrored; mirroring is a display-only setting. */
private fun ImageProxy.toUprightBitmap(): Bitmap {
    val raw = toBitmap()
    val degrees = imageInfo.rotationDegrees
    if (degrees == 0) return raw
    val rotated = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)
    if (rotated !== raw) raw.recycle()
    return rotated
}

private const val MIN_FRAME_INTERVAL_MS = 40L
