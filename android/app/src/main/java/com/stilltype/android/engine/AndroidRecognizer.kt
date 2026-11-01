package com.stilltype.android.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.FrameAnalysis
import com.stilltype.android.core.InstalledModels
import com.stilltype.android.core.MAX_SESSION_FRAMES
import com.stilltype.android.core.MAX_SESSION_SECONDS
import com.stilltype.android.core.MODEL_CHUNK_SECONDS
import com.stilltype.android.core.ModelRepository
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.core.RecognitionResult
import com.stilltype.android.core.VisualRecognizer
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.CoroutineContext
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray

/** Local-only visual recognition using installed, repository-verified model assets. */
class AndroidRecognizer(context: Context, private val models: ModelRepository) : VisualRecognizer {
    private val lock = Any()
    private val env = OrtEnvironment.getEnvironment()
    private var analyzer: FaceAnalyzer? = null
    private var recognition: OrtSession? = null
    private var sessionMode: ComputeMode? = null
    private var sessionDirectory: File? = null
    private var detectorDirectory: File? = null
    private var provider = "CPU"

    override fun analyze(bitmap: Bitmap, timestampMs: Long): FrameAnalysis = synchronized(lock) {
        try {
            val installed = models.installedModels() ?: return@synchronized FrameAnalysis(hint = "Install the model to begin")
            if (analyzer == null || detectorDirectory != installed.directory) {
                analyzer?.close()
                analyzer = null
                analyzer = FaceAnalyzer(installed.directory, env)
                detectorDirectory = installed.directory
            }
            analyzer!!.analyze(bitmap, timestampMs)
        } catch (error: Exception) {
            FrameAnalysis(hint = "Face detector unavailable: ${error.message ?: "unknown error"}")
        }
    }

    override suspend fun recognize(frames: List<MouthFrame>, compute: ComputeMode): RecognitionResult =
        withContext(Dispatchers.Default) {
            val context = coroutineContext
            synchronized(lock) { recognizeLocked(frames, compute, context) }
        }

    private fun recognizeLocked(
        frames: List<MouthFrame>,
        compute: ComputeMode,
        context: CoroutineContext,
    ): RecognitionResult {
        val duration = if (frames.size >= 2) frames.last().timestampMs - frames.first().timestampMs else 0L
        fun failure(code: String) = RecognitionResult("", "", max(0L, duration), 0, frames.size, "", code)
        if (frames.isEmpty()) return failure("no-face")
        if (duration <= 0 || frames.any { it.pixels.size != 96 * 96 } ||
            frames.zipWithNext().any { (a, b) -> b.timestampMs <= a.timestampMs }) return failure("invalid-clip")
        sessionLimitError(frames.size, duration)?.let { return failure(it) }
        if (duration < 500) return failure("too-short")
        val installed = models.installedModels() ?: return failure("models-missing")
        return try {
            context.ensureActive()
            val chunks = partitionWindows(frames, MODEL_CHUNK_SECONDS * 1000L)
            val session = getSession(installed, compute)
            val tokensJson = JSONArray(File(installed.directory, "tokens.json").readText())
            require(tokensJson.length() == 1049) { "Invalid token list" }
            val tokens = List(tokensJson.length()) { tokensJson.getString(it) }
            val parts = ArrayList<String>()
            var hadSpeech = false
            var totalMs = 0L
            for (chunk in chunks) {
                context.ensureActive()
                if (chunk.size < 2) continue
                val count = min(chunk.size, installed.manifest.maxFrames)
                val bounded = if (chunk.size <= count) chunk else List(count) { i ->
                    chunk[round(i.toDouble() * (chunk.size - 1) / (count - 1)).toInt()]
                }
                if (bounded.size < 2) continue
                if (bounded.last().timestampMs - bounded.first().timestampMs <= 0) continue
                val fps = (bounded.size - 1) * 1000.0 / (bounded.last().timestampMs - bounded.first().timestampMs)
                val span = speechSpan(bounded, fps) ?: continue
                hadSpeech = true
                val selected = bounded.subList(span.first, span.second)
                if (selected.size < 1) continue
                val outCount = if (abs(fps - 25.0) > 0.5) max(1, round(selected.size * 25.0 / fps).toInt()) else selected.size
                val n = min(installed.manifest.maxFrames, outCount)
                if (n < 1) continue
                val input = FloatArray(n * 88 * 88)
                for (i in 0 until n) {
                    val index = if (n == 1) 0 else round(i.toDouble() * (selected.size - 1) / (n - 1)).toInt()
                    val pixels = selected[index].pixels
                    for (y in 0 until 88) for (x in 0 until 88) {
                        input[i * 88 * 88 + y * 88 + x] = (((pixels[(y + 4) * 96 + x + 4].toInt() and 255) / 255f) - 0.421f) / 0.165f
                    }
                }
                val started = System.nanoTime()
                val text = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, n.toLong(), 88, 88)).use { tensor ->
                    session.run(mapOf(installed.manifest.inputName to tensor)).use { result ->
                        val logits = result.get(installed.manifest.outputName).orElseThrow { IllegalStateException("Missing logits") } as OnnxTensor
                        val shape = (logits.info as TensorInfo).shape
                        require(shape.contentEquals(longArrayOf(1, n.toLong(), tokens.size.toLong()))) { "Invalid CTC output shape" }
                        val values = logits.floatBuffer
                        var previous = -1
                        val words = StringBuilder()
                        repeat(n) {
                            var best = 0
                            var score = Float.NEGATIVE_INFINITY
                            for (id in tokens.indices) {
                                val value = values.get()
                                if (value > score) { score = value; best = id }
                            }
                            if (best != previous && best != installed.manifest.blankId && best != installed.manifest.eosId) words.append(tokens[best])
                            previous = best
                        }
                        words.toString().replace("▁", " ").trim()
                    }
                }
                totalMs += (System.nanoTime() - started) / 1_000_000
                if (text.isNotEmpty()) parts.add(text)
            }
            if (!hadSpeech) return failure("no-speech")
            val joined = parts.joinToString(" ").trim()
            RecognitionResult(joined, joined, duration, totalMs, frames.size, provider)
        } catch (e: CancellationException) {
            throw e
        } catch (error: Exception) {
            failure("recognition-failed: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun getSession(installed: InstalledModels, mode: ComputeMode): OrtSession {
        if (recognition != null && sessionMode == mode && sessionDirectory == installed.directory) return recognition!!
        recognition?.close()
        recognition = null
        val path = File(installed.directory, "usr2-ctc-int8.onnx").absolutePath
        fun create(nnapi: Boolean): OrtSession = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            if (nnapi) options.addNnapi()
            env.createSession(path, options)
        }
        recognition = if (mode == ComputeMode.AUTO) {
            try {
                create(true).also { provider = "NNAPI requested, CPU fallback" }
            } catch (_: Exception) {
                create(false).also { provider = "CPU (NNAPI unavailable)" }
            }
        } else create(false).also { provider = "CPU" }
        sessionMode = mode
        sessionDirectory = installed.directory
        return recognition!!
    }

    override fun close() {
        synchronized(lock) {
            try { analyzer?.close() } finally {
                analyzer = null
                detectorDirectory = null
                try { recognition?.close() } finally {
                    recognition = null
                    sessionDirectory = null
                    sessionMode = null
                }
            }
        }
    }
}

/** The desktop motion gate: mean-removed inner-mouth difference, lagged and smoothed. */
internal fun speechSpan(frames: List<MouthFrame>, fps: Double): Pair<Int, Int>? {
    if (frames.size < 2 || !fps.isFinite() || fps <= 0) return null
    val lag = max(1, round(fps * 0.1).toInt())
    if (frames.size <= lag) return null
    val means = DoubleArray(frames.size)
    for (i in frames.indices) {
        var sum = 0.0
        for (y in 24 until 72) for (x in 16 until 80) sum += (frames[i].pixels[y * 96 + x].toInt() and 255)
        means[i] = sum / (48 * 64)
    }
    val motion = DoubleArray(frames.size)
    for (i in lag until frames.size) {
        var difference = 0.0
        for (y in 24 until 72) for (x in 16 until 80) {
            val p = y * 96 + x
            difference += abs(((frames[i].pixels[p].toInt() and 255) - means[i]) -
                ((frames[i - lag].pixels[p].toInt() and 255) - means[i - lag]))
        }
        motion[i] = difference / (48 * 64)
    }
    for (i in 0 until lag) motion[i] = motion[lag]
    val width = max(1, (fps * 0.2).toInt())
    val moving = frames.indices.filter { i ->
        var total = 0.0
        for (j in 0 until width) {
            val index = i + (width - 1) / 2 - j
            if (index in motion.indices) total += motion[index]
        }
        total / width > 8.0
    }
    if (moving.size < 0.3 * fps) return null
    val padding = (0.4 * fps).toInt()
    return max(0, moving.first() - padding) to minOf(frames.size, moving.last() + 1 + padding)
}

/** True when a session exceeds the user recording limit (120 s / 3001 frames). */
internal fun sessionLimitError(frameCount: Int, durationMs: Long): String? =
    if (frameCount > MAX_SESSION_FRAMES || durationMs > MAX_SESSION_SECONDS * 1000L) "invalid-clip" else null

/**
 * Split a chronological session into balanced timestamp windows no longer than
 * [chunkMs]. The window count is ceil(covered span / chunkMs), where the span is
 * last-first plus one mean frame interval, so a low-frame-rate session still
 * splits by time rather than by frame count. Every input frame lands in exactly
 * one window, in order.
 */
internal fun partitionWindows(
    frames: List<MouthFrame>,
    chunkMs: Long = MODEL_CHUNK_SECONDS * 1000L,
): List<List<MouthFrame>> {
    if (frames.size < 2) return if (frames.isEmpty()) emptyList() else listOf(frames)
    val first = frames.first().timestampMs
    val last = frames.last().timestampMs
    val duration = (last - first).toDouble()
    if (duration <= 0) return listOf(frames)
    val interval = duration / (frames.size - 1)
    val span = duration + interval
    val count = ceil(span / chunkMs.toDouble()).toInt().coerceAtLeast(1)
    if (count <= 1) return listOf(frames)
    val width = span / count
    val buckets = List(count) { ArrayList<MouthFrame>() }
    for (frame in frames) {
        var index = ((frame.timestampMs - first).toDouble() / width).toInt()
        if (index < 0) index = 0
        if (index >= count) index = count - 1
        buckets[index].add(frame)
    }
    return buckets.filter { it.isNotEmpty() }
}
