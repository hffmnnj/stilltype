package com.stilltype.android.ui.capture

import android.app.Application
import android.content.ClipData
import android.content.ClipDescription
import android.content.ClipboardManager
import android.graphics.Bitmap
import android.os.Build
import android.os.PersistableBundle
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.stilltype.android.StilltypeApplication
import com.stilltype.android.core.HistoryEntry
import com.stilltype.android.core.MAX_SESSION_FRAMES
import com.stilltype.android.core.MAX_SESSION_SECONDS
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.data.TextRules
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

enum class CapturePhase { IDLE, STARTING, RECORDING, PROCESSING }

enum class FailureKind { NO_FACE, STILL, TOO_SHORT, INTERRUPTED, MODEL_MISSING, CAMERA, EMPTY, FAILED }

data class CaptureFailure(val kind: FailureKind, val title: String, val detail: String)

data class CaptureResult(
    val id: String,
    val text: String,
    val raw: String,
    val createdAt: Long,
    val durationMs: Long,
    val inferenceMs: Long,
    val frames: Int,
    val backend: String,
    val saved: Boolean,
    val copied: Boolean,
)

data class CaptureState(
    val phase: CapturePhase = CapturePhase.IDLE,
    /** True while a camera owner should be bound (preview or capture). */
    val cameraOn: Boolean = false,
    val elapsedMs: Long = 0,
    val limitMs: Long = 120_000,
    val result: CaptureResult? = null,
    val failure: CaptureFailure? = null,
)

/** Coarse, real detector output for the framing hint. Quantized so it updates a few times a second. */
data class LiveHint(
    val faceDetected: Boolean = false,
    val lightPercent: Int = 0,
    val motionPercent: Int = 0,
    val text: String = "",
)

/** What the camera analyzer needs from the capture owner. Called on the analyzer thread. */
interface FrameSink {
    /** Current session token; 0 means no session wants frames. */
    val session: Long
    fun onFrame(session: Long, bitmap: Bitmap, nowMs: Long)
    fun onCameraError(session: Long, message: String)
}

/**
 * Single capture owner shared by the main screen and the quick-capture sheet.
 * Every async step carries the session token that started it. Cancel, pause and
 * restart bump the token, so a late frame or recognition result is dropped
 * without saving, copying or displaying anything.
 */
class CaptureViewModel(application: Application) : AndroidViewModel(application), FrameSink {
    private val app = application as StilltypeApplication
    private val lock = Any()

    private val _state = MutableStateFlow(CaptureState())
    val state: StateFlow<CaptureState> = _state.asStateFlow()

    private val _live = MutableStateFlow(LiveHint())
    val live: StateFlow<LiveHint> = _live.asStateFlow()

    @Volatile private var token = 0L
    override val session: Long get() = token

    private val frames = ArrayList<MouthFrame>()
    private var observed = 0
    private var withFace = 0
    private var recordStartMs = 0L
    private var lastLiveMs = 0L
    private var ticker: Job? = null
    private var worker: Job? = null
    private var persistJob: Job? = null

    /** Starts capture. Returns false when one is already running. */
    fun start(): Boolean {
        val settings = app.repository.state.value.settings
        if (app.models.installedModels() == null) {
            _state.update {
                it.copy(
                    failure = CaptureFailure(FailureKind.MODEL_MISSING, "Model not ready", "Set up the recognition model before recording."),
                )
            }
            return false
        }
        val mine: Long
        synchronized(lock) {
            if (_state.value.phase != CapturePhase.IDLE) return false
            token += 1
            mine = token
            frames.clear()
            observed = 0
            withFace = 0
            _state.update {
                it.copy(
                    phase = CapturePhase.STARTING,
                    cameraOn = true,
                    elapsedMs = 0,
                    limitMs = settings.maxClipSeconds.coerceIn(3, MAX_SESSION_SECONDS) * 1000L,
                    failure = null,
                )
            }
        }
        viewModelScope.launch {
            delay(CAMERA_START_TIMEOUT_MS)
            synchronized(lock) {
                if (token == mine && _state.value.phase == CapturePhase.STARTING) {
                    failLocked(FailureKind.CAMERA, "Camera did not start", "No frames arrived. Close other camera apps and try again.")
                }
            }
        }
        return true
    }

    /** Camera-only framing check. Nothing is recorded. */
    fun setPreview(on: Boolean) {
        synchronized(lock) {
            if (_state.value.phase != CapturePhase.IDLE) return
            token += 1
            _state.update { it.copy(cameraOn = on, failure = null) }
            if (!on) _live.value = LiveHint()
        }
    }

    fun stop() {
        val clip: List<MouthFrame>
        val seen: Int
        val faces: Int
        val mine: Long
        val limit: Long
        synchronized(lock) {
            if (_state.value.phase != CapturePhase.RECORDING) return
            ticker?.cancel()
            clip = ArrayList(frames)
            seen = observed
            faces = withFace
            mine = token
            limit = _state.value.limitMs
            // Frames arriving from here on are ignored; the camera is released by the UI.
            _state.update { it.copy(phase = CapturePhase.PROCESSING, cameraOn = false, elapsedMs = minOf(it.elapsedMs, limit)) }
            _live.value = LiveHint()
        }
        worker = viewModelScope.launch { finish(mine, clip, seen, faces) }
    }

    /** Cancel, Back, or the app leaving the foreground. Produces no transcript. */
    fun cancel() {
        synchronized(lock) { cancelLocked() }
    }

    /** Lifecycle ON_PAUSE: release the camera and drop anything pending. */
    fun onPaused() {
        synchronized(lock) { cancelLocked() }
        flushEdit()
    }

    fun clearFailure() = _state.update { it.copy(failure = null) }

    override fun onFrame(session: Long, bitmap: Bitmap, nowMs: Long) {
        if (session != token) return
        // Detector runs outside the lock; only the bookkeeping is synchronized.
        val analysis = app.recognizer.analyze(bitmap, nowMs)
        synchronized(lock) {
            if (session != token) return
            publishLive(analysis.faceDetected, analysis.brightness, analysis.motion, analysis.hint, nowMs)
            when (_state.value.phase) {
                CapturePhase.STARTING -> {
                    recordStartMs = nowMs
                    _state.update { it.copy(phase = CapturePhase.RECORDING, elapsedMs = 0) }
                    countLocked(analysis.faceDetected, analysis.mouth)
                    ticker?.cancel()
                    ticker = viewModelScope.launch { tick(session) }
                }
                CapturePhase.RECORDING -> countLocked(analysis.faceDetected, analysis.mouth)
                else -> Unit
            }
        }
    }

    override fun onCameraError(session: Long, message: String) {
        synchronized(lock) {
            if (session != token) return
            failLocked(FailureKind.CAMERA, "Camera unavailable", message)
        }
    }

    fun onPermissionDenied() {
        _state.update {
            it.copy(failure = CaptureFailure(FailureKind.CAMERA, "Camera access is off", "Stilltype needs the camera to read lip movement. Nothing is recorded to disk."))
        }
    }

    /** Edits the latest transcript; persisted with its original ID, raw text and timing. */
    fun editResult(text: String) {
        val current = _state.value.result ?: return
        _state.update { it.copy(result = current.copy(text = text)) }
        persistJob?.cancel()
        persistJob = viewModelScope.launch {
            delay(EDIT_SAVE_DELAY_MS)
            persistEdit()
        }
    }

    fun flushEdit() {
        if (persistJob?.isActive == true) {
            persistJob?.cancel()
            viewModelScope.launch { persistEdit() }
        }
    }

    fun copyResult(): Boolean {
        val text = _state.value.result?.text?.takeIf { it.isNotBlank() } ?: return false
        return copyToClipboard(text).also { ok ->
            if (ok) _state.update { s -> s.copy(result = s.result?.copy(copied = true)) }
        }
    }

    fun dismissResult() = _state.update { it.copy(result = null) }

    // --- internals -------------------------------------------------------

    private fun countLocked(face: Boolean, mouth: MouthFrame?) {
        observed += 1
        if (face) withFace += 1
        if (mouth != null && frames.size < MAX_SESSION_FRAMES) frames += mouth
    }

    private fun publishLive(face: Boolean, light: Float, motion: Float, hint: String, nowMs: Long) {
        if (nowMs - lastLiveMs < LIVE_INTERVAL_MS) return
        lastLiveMs = nowMs
        val next = LiveHint(face, (light * 20).toInt() * 5, (motion * 20).toInt() * 5, hint)
        if (next != _live.value) _live.value = next
    }

    private suspend fun tick(mine: Long) {
        while (true) {
            delay(100)
            val stopNow = synchronized(lock) {
                if (token != mine || _state.value.phase != CapturePhase.RECORDING) return
                val elapsed = SystemClock.elapsedRealtime() - recordStartMs
                _state.update { it.copy(elapsedMs = minOf(elapsed, it.limitMs)) }
                elapsed >= _state.value.limitMs
            }
            if (stopNow) {
                stop()
                return
            }
        }
    }

    private fun failLocked(kind: FailureKind, title: String, detail: String) {
        cancelLocked()
        _state.update { it.copy(failure = CaptureFailure(kind, title, detail)) }
    }

    private fun cancelLocked() {
        val active = _state.value.phase != CapturePhase.IDLE || _state.value.cameraOn
        if (!active) return
        token += 1
        ticker?.cancel()
        worker?.cancel()
        frames.clear()
        observed = 0
        withFace = 0
        _live.value = LiveHint()
        _state.update { it.copy(phase = CapturePhase.IDLE, cameraOn = false, elapsedMs = 0) }
    }

    private suspend fun finish(mine: Long, clip: List<MouthFrame>, seen: Int, faces: Int) {
        // The analyzer can only report frames with a mouth, so the face ratio is tracked here.
        if (seen == 0 || faces * 2 < seen) {
            return fail(mine, FailureKind.NO_FACE, "No face found", "Keep your face in frame and well lit for the whole clip.")
        }
        val settings = app.repository.state.value.settings
        val outcome = try {
            app.recognizer.recognize(clip, settings.computeMode)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            return fail(mine, FailureKind.FAILED, "Recognition failed", e.message ?: e.javaClass.simpleName)
        }
        outcome.error?.let { code ->
            return when {
                code == "no-face" -> fail(mine, FailureKind.NO_FACE, "No face found", "Keep your face in frame and well lit for the whole clip.")
                code == "too-short" -> fail(mine, FailureKind.TOO_SHORT, "Clip too short", "Record at least one second of mouthing before you stop.")
                code == "no-speech" -> fail(mine, FailureKind.STILL, "No mouth movement", "Mouth the words clearly while you record. A still face gives no text.")
                code == "invalid-clip" -> fail(mine, FailureKind.INTERRUPTED, "Capture was interrupted", "Frames were not continuous. Try again.")
                code == "models-missing" -> fail(mine, FailureKind.MODEL_MISSING, "Model not ready", "Set up the recognition model before recording.")
                else -> fail(mine, FailureKind.FAILED, "Recognition failed", code.removePrefix("recognition-failed: "))
            }
        }
        val vocabulary = app.repository.state.value.vocabulary
        val text = TextRules.applyVocabulary(outcome.text, vocabulary).trim()
        if (text.isEmpty()) {
            return fail(mine, FailureKind.EMPTY, "No words recognised", "The model returned empty text. Try again, closer and slower.")
        }
        // Last gate before any side effect.
        if (!synchronized(lock) { token == mine && _state.value.phase == CapturePhase.PROCESSING }) return
        val entry = HistoryEntry(
            id = UUID.randomUUID().toString(),
            text = text,
            raw = outcome.raw,
            createdAt = System.currentTimeMillis(),
            durationMs = outcome.durationMs.coerceIn(0, 120_000),
            inferenceMs = outcome.inferenceMs.coerceAtLeast(0),
        )
        var saved = false
        if (settings.keepHistory) {
            saved = try {
                app.repository.saveEntry(entry)
                true
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                false
            }
        }
        val copied = settings.autoCopy && synchronized(lock) {
            token == mine && copyToClipboard(text)
        }
        synchronized(lock) {
            if (token != mine) return
            _state.update {
                it.copy(
                    phase = CapturePhase.IDLE,
                    result = CaptureResult(
                        entry.id, text, entry.raw, entry.createdAt, entry.durationMs, entry.inferenceMs,
                        outcome.frames, outcome.backend, saved, copied,
                    ),
                    failure = if (settings.keepHistory && !saved) {
                        CaptureFailure(FailureKind.FAILED, "Not saved to history", "The transcript is shown but could not be stored.")
                    } else null,
                )
            }
        }
    }

    private fun fail(mine: Long, kind: FailureKind, title: String, detail: String) {
        synchronized(lock) {
            if (token != mine) return
            _state.update { it.copy(phase = CapturePhase.IDLE, failure = CaptureFailure(kind, title, detail)) }
        }
    }

    private suspend fun persistEdit() {
        val result = _state.value.result ?: return
        if (!result.saved) return
        val stored = app.repository.state.value.history.firstOrNull { it.id == result.id } ?: return
        if (stored.text == result.text) return
        try {
            app.repository.saveEntry(stored.copy(text = result.text))
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            // The repository publishes its own error state.
        }
    }

    private fun copyToClipboard(text: String): Boolean = try {
        val manager = app.getSystemService(ClipboardManager::class.java)
        val clip = ClipData.newPlainText("Stilltype", text)
        if (Build.VERSION.SDK_INT >= 33) {
            clip.description.extras = PersistableBundle().apply { putBoolean(ClipDescription.EXTRA_IS_SENSITIVE, true) }
        }
        manager.setPrimaryClip(clip)
        true
    } catch (_: Exception) {
        false
    }

    override fun onCleared() {
        synchronized(lock) { token += 1 }
        super.onCleared()
    }

    private companion object {
        const val CAMERA_START_TIMEOUT_MS = 15_000L
        const val LIVE_INTERVAL_MS = 250L
        const val EDIT_SAVE_DELAY_MS = 700L
    }
}
