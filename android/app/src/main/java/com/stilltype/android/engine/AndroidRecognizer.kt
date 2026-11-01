package com.stilltype.android.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.FrameAnalysis
import com.stilltype.android.core.InstalledModels
import com.stilltype.android.core.InstalledQualityModels
import com.stilltype.android.core.ImagePoint
import com.stilltype.android.core.MAX_SESSION_FRAMES
import com.stilltype.android.core.MAX_SESSION_SECONDS
import com.stilltype.android.core.MODEL_CHUNK_SECONDS
import com.stilltype.android.core.ModelRepository
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.core.QualityFrameAccumulator
import com.stilltype.android.core.QualityModelRepository
import com.stilltype.android.core.RecognitionProgress
import com.stilltype.android.core.RecognitionResult
import com.stilltype.android.core.VisualRecognizer
import com.stilltype.android.data.checkQualityMemory
import java.io.File
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.coroutines.CoroutineContext
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray

/** Local-only visual recognition using installed, repository-verified model assets. */
class AndroidRecognizer(
    context: Context,
    private val models: ModelRepository,
    private val qualityModels: QualityModelRepository? = null,
) : VisualRecognizer {
    private val lock = Any()
    private val env = OrtEnvironment.getEnvironment()
    private var analyzer: FaceAnalyzer? = null
    private var recognition: OrtSession? = null
    private var sessionMode: ComputeMode? = null
    private var sessionDirectory: File? = null
    private var detectorDirectory: File? = null
    private var provider = "CPU"
    private val profileDirectory = context.applicationContext.cacheDir
    private val appContext = context.applicationContext
    private var profilePrefix: File? = null
    private var qualityDecoder: FlareBeamDecoder? = null
    private var qualityDirectory: File? = null
    private var qualityMode: ComputeMode? = null

    override fun analyze(bitmap: Bitmap, timestampMs: Long): FrameAnalysis = analyzeFrame(bitmap, timestampMs, false)

    override fun analyzeForQuality(bitmap: Bitmap, timestampMs: Long): FrameAnalysis = analyzeFrame(bitmap, timestampMs, true)

    private fun analyzeFrame(bitmap: Bitmap, timestampMs: Long, quality: Boolean): FrameAnalysis = synchronized(lock) {
        try {
            val installed = models.installedModels() ?: return@synchronized FrameAnalysis(hint = "Install the model to begin")
            if (analyzer == null || detectorDirectory != installed.directory) {
                analyzer?.close()
                analyzer = null
                analyzer = FaceAnalyzer(installed.directory, env)
                detectorDirectory = installed.directory
            }
            analyzer!!.analyze(bitmap, timestampMs, quality)
        } catch (error: Exception) {
            FrameAnalysis(hint = "Face detector unavailable: ${error.message ?: "unknown error"}")
        }
    }

    override fun createQualityAccumulator(): QualityFrameAccumulator? {
        val installed = models.installedModels() ?: return null
        val points = JSONArray(File(installed.directory, "mean-face.json").readText())
        require(points.length() == 4) { "Invalid alignment reference" }
        return CenteredFaceAligner(List(4) { index ->
            val point = points.getJSONArray(index)
            ImagePoint(point.getDouble(0).toFloat(), point.getDouble(1).toFloat())
        })
    }

    /** Includes Fast gating, model loading, preprocessing, and decoding in inferenceMs. */
    override suspend fun recognizeQuality(
        frames: List<MouthFrame>,
        compute: ComputeMode,
        onProgress: (RecognitionProgress) -> Unit,
    ): RecognitionResult = withContext(Dispatchers.Default) {
        val context = coroutineContext
        val started = System.nanoTime()
        val duration = if (frames.size >= 2) frames.last().timestampMs - frames.first().timestampMs else 0L
        fun result(text: String = "", backend: String = "", error: String? = null) = RecognitionResult(
            text, text, max(0L, duration), (System.nanoTime() - started) / 1_000_000, frames.size, backend, error,
        )
        try {
            if (frames.isEmpty()) return@withContext result(error = "no-face")
            if (duration <= 0 || frames.any { it.pixels.size != 96 * 96 } ||
                frames.zipWithNext().any { (a, b) -> b.timestampMs <= a.timestampMs } ||
                sessionLimitError(frames.size, duration) != null) return@withContext result(error = "invalid-clip")
            if (duration < 500) return@withContext result(error = "too-short")
            if (frames.any { it.quality?.let { crops -> crops.face.size == 96 * 96 && crops.lip.size == 96 * 96 } != true }) {
                return@withContext result(error = "quality-crops-missing")
            }
            context.ensureActive()
            val memory = checkQualityMemory(appContext)
            val memoryReason = memory.unavailableReason ?: if (memory.lowMemory) {
                "The device is currently low on memory. Try again later or use Fast."
            } else null
            if (memoryReason != null) {
                return@withContext result(backend = "Quality decoder not run: $memoryReason", error = "quality-memory-low")
            }
            val installed = synchronized(lock) { qualityModels?.installedModels() }
                ?: return@withContext result(error = "quality-models-missing")
            val chunks = partitionWindows(frames, MODEL_CHUNK_SECONDS * 1000L)
            val parts = mutableListOf<String>()
            var backend = "Quality decoder not run (Fast activity veto)"
            for ((index, chunk) in chunks.withIndex()) {
                context.ensureActive()
                // Callback is outside this lock: capture cancellation must not invert lock order.
                val window = synchronized(lock) {
                    context.ensureActive()
                    val gate = recognizeLocked(chunk, compute, context)
                    when {
                        gate.error == "no-speech" -> null
                        gate.error != null -> throw QualityGateException(gate.error)
                        gate.raw.isBlank() -> null
                        else -> {
                            val input = qualityInput(chunk)
                            val output = getQualityDecoder(installed, compute).decode(input.first, input.second, input.third) { context.ensureActive() }
                            val manifest = installed.manifest
                            val text = output.tokenIds.mapNotNull { id ->
                                require(id >= 0 && id < manifest.symbols.size) { "Invalid Quality token ID" }
                                when (id.toInt()) {
                                    manifest.bosId, manifest.padId, manifest.eosId -> null
                                    manifest.unkId -> "<unk>"
                                    else -> manifest.symbols[id.toInt()]
                                }
                            }.joinToString("").replace("▁", " ").trim()
                            text to output.backend
                        }
                    }
                }
                context.ensureActive()
                if (window != null) {
                    if (window.first.isNotBlank()) parts.add(window.first)
                    backend = window.second
                }
                onProgress(RecognitionProgress(index + 1, chunks.size))
            }
            context.ensureActive()
            val joined = parts.joinToString(" ").trim()
            if (joined.isEmpty()) result(backend = backend, error = "no-speech") else result(joined, backend)
        } catch (error: OutOfMemoryError) {
            // Release owned native sessions before reporting the failure; close resets them for a later retry.
            try { close() } catch (_: Exception) { } catch (_: OutOfMemoryError) { }
            result(error = "quality-memory-low")
        } catch (error: CancellationException) {
            throw error
        } catch (error: QualityGateException) {
            result(error = "quality-gate-failed: ${error.code}")
        } catch (error: Exception) {
            result(error = "quality-recognition-failed: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    private class QualityGateException(val code: String) : Exception(code)

    private fun qualityInput(chunk: List<MouthFrame>): Triple<FloatArray, FloatArray, Int> {
        require(chunk.size >= 2)
        val fps = (chunk.size - 1) * 1000.0 / (chunk.last().timestampMs - chunk.first().timestampMs)
        val count = min(200, max(1, if (abs(fps - 25.0) > 0.5) round(chunk.size * 25.0 / fps).toInt() else chunk.size))
        val face = FloatArray(count * 88 * 88)
        val lip = FloatArray(face.size)
        for (i in 0 until count) {
            val index = if (count == 1) 0 else round(i.toDouble() * (chunk.size - 1) / (count - 1)).toInt()
            val crops = requireNotNull(chunk[index].quality)
            for (y in 0 until 88) for (x in 0 until 88) {
                val src = (y + 4) * 96 + x + 4
                val dst = i * 88 * 88 + y * 88 + x
                face[dst] = (((crops.face[src].toInt() and 255) / 255f) - 0.421f) / 0.165f
                lip[dst] = (((crops.lip[src].toInt() and 255) / 255f) - 0.421f) / 0.165f
            }
        }
        return Triple(face, lip, count)
    }

    private fun getQualityDecoder(installed: InstalledQualityModels, mode: ComputeMode): FlareBeamDecoder {
        if (qualityDecoder != null && qualityDirectory == installed.directory && qualityMode == mode) return qualityDecoder!!
        qualityDecoder?.close()
        qualityDecoder = null
        qualityDirectory = null
        qualityMode = null
        val directory = installed.directory
        val files = FlareGraphFiles(File(directory, "encoder.onnx"), File(directory, "cross.onnx"),
            File(directory, "decoder-beam.onnx"), File(directory, "cache-gather.onnx"), File(directory, "cache-merge.onnx"))
        return FlareBeamDecoder(env, files, 50, mode, profileDirectory).also {
            qualityDecoder = it
            qualityDirectory = directory
            qualityMode = mode
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
                val spans = activitySpans(bounded, fps)
                val span = spans.candidate ?: continue
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
                var completed = false
                val decoded = try {
                    val decoded = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, n.toLong(), 88, 88)).use { tensor ->
                        session.run(mapOf(installed.manifest.inputName to tensor)).use { result ->
                            val logits = result.get(installed.manifest.outputName).orElseThrow { IllegalStateException("Missing logits") } as OnnxTensor
                            val shape = (logits.info as TensorInfo).shape
                            require(shape.contentEquals(longArrayOf(1, n.toLong(), tokens.size.toLong()))) { "Invalid CTC output shape" }
                            decodeCtc(logits.floatBuffer, n, tokens, installed.manifest.blankId, installed.manifest.eosId)
                        }
                    }
                    totalMs += (System.nanoTime() - started) / 1_000_000
                    completed = true
                    decoded
                } finally {
                    finishProfiling(session, completed)
                }
                if (spans.strong == null && !decoded.hasActivityEvidence()) continue
                hadSpeech = true
                if (decoded.text.isNotEmpty()) parts.add(decoded.text)
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
        recognition?.let { session ->
            try { finishProfiling(session, false) } finally { session.close() }
        }
        recognition = null
        val path = File(installed.directory, "usr2-ctc-int8.onnx").absolutePath
        fun create(nnapi: Boolean): OrtSession = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            if (nnapi) options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
            if (mode == ComputeMode.AUTO) {
                val prefix = try { File.createTempFile("ort-profile-", ".tmp", profileDirectory) } catch (_: Exception) { null }
                if (prefix != null) {
                    try {
                        options.enableProfiling(prefix.absolutePath)
                        profilePrefix = prefix
                    } catch (_: Exception) {
                        cleanProfile(prefix)
                    }
                }
            }
            try {
                env.createSession(path, options)
            } catch (error: Exception) {
                profilePrefix?.let { cleanProfile(it) }
                profilePrefix = null
                throw error
            }
        }
        provider = if (mode == ComputeMode.AUTO) "NNAPI requested; execution unverified" else "CPU"
        recognition = if (mode == ComputeMode.AUTO) {
            try {
                create(true)
            } catch (_: Exception) {
                create(false)
            }
        } else create(false)
        sessionMode = mode
        sessionDirectory = installed.directory
        return recognition!!
    }

    private fun finishProfiling(session: OrtSession, completed: Boolean) {
        val prefix = profilePrefix ?: return
        profilePrefix = null
        try {
            val result = File(session.endProfiling())
            if (completed && result.parentFile == prefix.parentFile && result.name.startsWith(prefix.name + "_") && result.name.endsWith(".json")) {
                val diagnostics = parseComputeProfile(result.readText())
                provider = diagnostics.label
                Log.i("StilltypeCompute", "ORT node events: NNAPI=${diagnostics.nnapiNodes}, CPU=${diagnostics.cpuNodes}, complete=${diagnostics.complete}")
            }
        } catch (_: Exception) {
            // Diagnostics must not replace or suppress recognition errors.
        } finally {
            cleanProfile(prefix)
        }
    }

    private fun cleanProfile(prefix: File) {
        try {
            val files = prefix.parentFile?.listFiles()
            if (files == null) {
                Log.w("StilltypeCompute", "Could not inspect temporary ORT profile directory")
                return
            }
            for (file in files) {
                if (file == prefix || (file.name.startsWith(prefix.name + "_") && file.name.endsWith(".json"))) {
                    if (!file.delete()) Log.w("StilltypeCompute", "Could not delete temporary ORT profile")
                }
            }
        } catch (_: Exception) {
            Log.w("StilltypeCompute", "Could not delete temporary ORT profile")
        }
    }

    override fun close() {
        synchronized(lock) {
            try { analyzer?.close() } finally {
                analyzer = null
                detectorDirectory = null
                try {
                    recognition?.let { session ->
                        try { finishProfiling(session, false) } finally { session.close() }
                    }
                } finally {
                    recognition = null
                    sessionDirectory = null
                    sessionMode = null
                    try { qualityDecoder?.close() } finally {
                        qualityDecoder = null
                        qualityDirectory = null
                        qualityMode = null
                    }
                }
            }
        }
    }
}

/** The desktop motion gate: mean-removed inner-mouth difference, lagged and smoothed. */
internal data class ActivitySpans(val strong: Pair<Int, Int>?, val candidate: Pair<Int, Int>?)

internal fun speechSpan(frames: List<MouthFrame>, fps: Double): Pair<Int, Int>? = activitySpans(frames, fps).strong

internal fun activitySpans(frames: List<MouthFrame>, fps: Double): ActivitySpans {
    if (frames.size < 2 || !fps.isFinite() || fps <= 0) return ActivitySpans(null, null)
    val lag = max(1, round(fps * 0.1).toInt())
    if (frames.size <= lag) return ActivitySpans(null, null)
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
    val smoothed = DoubleArray(frames.size) { i ->
        var total = 0.0
        for (j in 0 until width) {
            val index = i + (width - 1) / 2 - j
            if (index in motion.indices) total += motion[index]
        }
        total / width
    }
    fun span(threshold: Double): Pair<Int, Int>? {
        val moving = frames.indices.filter { smoothed[it] > threshold }
        if (moving.size < 0.3 * fps) return null
        val padding = (0.4 * fps).toInt()
        return max(0, moving.first() - padding) to minOf(frames.size, moving.last() + 1 + padding)
    }
    return ActivitySpans(span(8.0), span(4.0))
}

internal data class CtcDecode(val text: String, val emissions: Int, val meanProbability: Double) {
    fun hasActivityEvidence(): Boolean = text.isNotBlank() && emissions >= 2 && meanProbability >= 0.30
}

internal fun decodeCtc(values: FloatBuffer, frames: Int, tokens: List<String>, blank: Int, eos: Int): CtcDecode {
    var previous = -1
    var count = 0
    var probabilitySum = 0.0
    val words = StringBuilder()
    repeat(frames) {
        var best = 0
        var score = Float.NEGATIVE_INFINITY
        for (id in tokens.indices) {
            val value = values.get()
            require(value.isFinite()) { "Non-finite CTC output" }
            if (value > score) { score = value; best = id }
        }
        if (best != previous && best != blank && best != eos) {
            words.append(tokens[best])
            count++
            probabilitySum += exp(score.toDouble())
        }
        previous = best
    }
    return CtcDecode(words.toString().replace("▁", " ").trim(), count, if (count == 0) 0.0 else probabilitySum / count)
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
