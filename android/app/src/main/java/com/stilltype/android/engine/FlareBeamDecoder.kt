package com.stilltype.android.engine

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import com.stilltype.android.core.ComputeMode
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import java.util.PriorityQueue
import kotlin.math.exp
import kotlin.math.ln

/** Result owners outlive all tensors borrowed from them. Only the encoder may use NNAPI. */
internal class FlareBeamDecoder(
    private val environment: OrtEnvironment,
    files: FlareGraphFiles,
    private val beamSize: Int,
    computeMode: ComputeMode = ComputeMode.CPU,
    profileDirectory: File? = null,
) : AutoCloseable {
    private val sessions = Resources()
    private val encoder: OrtSession
    private val cross: OrtSession
    private val decoder: OrtSession
    private val gather: OrtSession
    private val merge: OrtSession
    private var profilePrefix: File? = null
    private var encoderBackend = if (computeMode == ComputeMode.AUTO) "NNAPI requested; execution unverified" else "CPU"

    init {
        require(beamSize == 5 || beamSize == 50)
        try {
            fun createEncoder(nnapi: Boolean): OrtSession = OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(4)
                options.setInterOpNumThreads(1)
                if (nnapi) options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
                if (computeMode == ComputeMode.AUTO && profileDirectory != null) {
                    val prefix = try { File.createTempFile("ort-profile-", ".tmp", profileDirectory) } catch (_: Exception) { null }
                    if (prefix != null) try {
                        options.enableProfiling(prefix.absolutePath)
                        profilePrefix = prefix
                    } catch (_: Exception) { cleanProfile(prefix) }
                }
                try { environment.createSession(files.encoder.absolutePath, options) } catch (error: Throwable) {
                    profilePrefix?.let { cleanProfile(it) }
                    profilePrefix = null
                    throw error
                }
            }
            encoder = sessions.add(if (computeMode == ComputeMode.AUTO) {
                try { createEncoder(true) } catch (_: Exception) { createEncoder(false) }
            } else createEncoder(false))
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(4)
                options.setInterOpNumThreads(1)
                cross = sessions.add(environment.createSession(files.cross.absolutePath, options))
                decoder = sessions.add(environment.createSession(files.decoder.absolutePath, options))
                gather = sessions.add(environment.createSession(files.cacheGather.absolutePath, options))
                merge = sessions.add(environment.createSession(files.cacheMerge.absolutePath, options))
            }
        } catch (error: Throwable) {
            try { sessions.close() } catch (closeError: Throwable) { error.addSuppressed(closeError) }
            profilePrefix?.let { cleanProfile(it) }
            profilePrefix = null
            throw error
        }
    }

    override fun close() {
        try { finishProfiling(false) } finally { sessions.close() }
    }

    private fun finishProfiling(completed: Boolean) {
        val prefix = profilePrefix ?: return
        profilePrefix = null
        try {
            val result = File(encoder.endProfiling())
            if (completed && result.parentFile == prefix.parentFile && result.name.startsWith(prefix.name + "_") && result.name.endsWith(".json")) {
                val diagnostics = parseComputeProfile(result.readText())
                encoderBackend = diagnostics.label
                Log.i("StilltypeCompute", "Quality encoder ORT node events: NNAPI=${diagnostics.nnapiNodes}, CPU=${diagnostics.cpuNodes}, complete=${diagnostics.complete}")
            }
        } catch (_: Exception) {
            // Profiling must not mask recognition or close failures.
        } finally { cleanProfile(prefix) }
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
        } catch (_: Exception) { Log.w("StilltypeCompute", "Could not delete temporary ORT profile") }
    }

    fun decode(face: FloatArray, lip: FloatArray, frames: Int, checkCancelled: () -> Unit = {}): FlareBeamOutput {
        require(frames in 1..200 && face.size == frames * 88 * 88 && lip.size == face.size)
        require(face.all { it.isFinite() } && lip.all { it.isFinite() })
        checkCancelled()
        val suffixes = FlareCacheLayout.suffixes
        var encoderMs = 0.0
        var crossMs = 0.0
        var decoderMs = 0.0
        var gatherMs = 0.0
        var mergeMs = 0.0
        var copyMs = 0.0
        var selectionMs = 0.0
        var calls = 0
        var maximumRows = 0
        Resources().use { inputs ->
            val shape = longArrayOf(1, 1, frames.toLong(), 88, 88)
            val feeds = mapOf(
                "face" to inputs.add(OnnxTensor.createTensor(environment, FloatBuffer.wrap(face), shape)),
                "lip" to inputs.add(OnnxTensor.createTensor(environment, FloatBuffer.wrap(lip), shape)),
                "face_mask" to inputs.add(OnnxTensor.createTensor(environment, booleanArrayOf(true))),
                "lip_mask" to inputs.add(OnnxTensor.createTensor(environment, booleanArrayOf(true))),
                "padding_mask" to inputs.add(OnnxTensor.createTensor(environment, arrayOf(BooleanArray(frames)))),
            )
            checkCancelled()
            var start = System.nanoTime()
            encoder.run(feeds).use { encodedResult ->
                encoderMs = elapsed(start)
                finishProfiling(true)
                val encoded = output(encodedResult, "encoder_out", OnnxJavaType.FLOAT, longArrayOf(frames.toLong(), 1, 512))
                finite(encoded.floatBuffer, "encoder_out")
                val mask = output(encodedResult, "encoder_padding_mask", OnnxJavaType.BOOL, longArrayOf(1, frames.toLong()))
                @Suppress("UNCHECKED_CAST")
                val maskRow = (mask.value as Array<BooleanArray>)[0].copyOf()
                checkCancelled()
                start = System.nanoTime()
                cross.run(mapOf("encoder_out" to encoded)).use { crossResult ->
                    crossMs = elapsed(start)
                    Resources().use { fixed ->
                        val one = mutableMapOf<String, OnnxTensor>()
                        val five = mutableMapOf<String, OnnxTensor>()
                        for (suffix in suffixes) {
                            val name = "cross_$suffix"
                            val tensor = output(crossResult, name, OnnxJavaType.FLOAT, longArrayOf(1, 4, frames.toLong(), 192))
                            val buffer = tensor.floatBuffer
                            val row = FloatArray(buffer.remaining()).also { buffer.get(it) }
                            require(row.all { it.isFinite() }) { "Non-finite $name" }
                            one[suffix] = fixed.add(cacheTensor(row, 1, frames))
                            val repeated = FloatArray(row.size * 5)
                            repeat(5) { row.copyInto(repeated, it * row.size) }
                            five[suffix] = fixed.add(cacheTensor(repeated, 5, frames))
                        }
                        val maskOne = fixed.add(OnnxTensor.createTensor(environment, arrayOf(maskRow)))
                        val maskFive = fixed.add(OnnxTensor.createTensor(environment, Array(5) { maskRow.copyOf() }))
                        val emptyPast = fixed.add(cacheTensor(FloatArray(0), 1, 0))
                        var cache: OrtSession.Result? = null
                        var prefixes = listOf(longArrayOf(FlareCacheLayout.EOS.toLong()))
                        var cumulative = floatArrayOf(0f)
                        var ignored = BooleanArray(beamSize)
                        val finalized = mutableListOf<Finalized>()
                        val loopStart = System.nanoTime()
                        try {
                            for (step in 0 until FlareCacheLayout.MAX_PREFIX) {
                                checkCancelled()
                                val batch = prefixes.size
                                val length = prefixes[0].size
                                check(batch == if (step == 0) 1 else beamSize)
                                val logits = FloatArray(batch * FlareCacheLayout.VOCABULARY)
                                Resources().use { chunkResults ->
                                    val results = mutableListOf<OrtSession.Result>()
                                    for (rowStart in 0 until batch step 5) {
                                        checkCancelled()
                                        val rows = minOf(5, batch - rowStart)
                                        check(rows == 1 || rows == 5)
                                        Resources().use { temporary ->
                                            val stepFeeds = mutableMapOf<String, OnnxTensor>()
                                            val values = LongArray(rows * length) { prefixes[rowStart + it / length][it % length] }
                                            stepFeeds["tokens"] = temporary.add(longTensor(values, longArrayOf(rows.toLong(), length.toLong())))
                                            stepFeeds["encoder_padding_mask"] = if (rows == 1) maskOne else maskFive
                                            val selected = if (cache != null) {
                                                val gatherFeeds = mutableMapOf<String, OnnxTensor>()
                                                gatherFeeds["indices"] = temporary.add(longTensor(LongArray(rows) { (rowStart + it).toLong() }, longArrayOf(rows.toLong())))
                                                for (suffix in suffixes) gatherFeeds["past_$suffix"] = output(cache!!, "past_$suffix", OnnxJavaType.FLOAT, cacheShape(batch, length - 1))
                                                start = System.nanoTime()
                                                temporary.add(gather.run(gatherFeeds)).also { gatherMs += elapsed(start) }
                                            } else null
                                            for (suffix in suffixes) {
                                                stepFeeds["cross_$suffix"] = (if (rows == 1) one else five).getValue(suffix)
                                                stepFeeds["past_$suffix"] = if (selected == null) emptyPast else output(selected, "selected_$suffix", OnnxJavaType.FLOAT, cacheShape(rows, length - 1))
                                            }
                                            checkCancelled()
                                            start = System.nanoTime()
                                            val result = chunkResults.add(decoder.run(stepFeeds))
                                            decoderMs += elapsed(start)
                                            calls++
                                            maximumRows = maxOf(maximumRows, rows)
                                            results.add(result)
                                            start = System.nanoTime()
                                            val logitBuffer = output(result, "next_logits", OnnxJavaType.FLOAT, longArrayOf(rows.toLong(), FlareCacheLayout.VOCABULARY.toLong())).floatBuffer
                                            val count = logitBuffer.remaining()
                                            logitBuffer.get(logits, rowStart * FlareCacheLayout.VOCABULARY, count)
                                            for (i in rowStart * FlareCacheLayout.VOCABULARY until rowStart * FlareCacheLayout.VOCABULARY + count) check(logits[i].isFinite()) { "Non-finite next_logits" }
                                            copyMs += elapsed(start)
                                        }
                                    }
                                    start = System.nanoTime()
                                    val candidates = bestCandidates(logits, cumulative, step)
                                    val eosMask = BooleanArray(2 * beamSize) { j ->
                                        candidates[j].word == FlareCacheLayout.EOS && candidates[j].score.isFinite() && (step == 0 || j >= beamSize || !ignored[j])
                                    }
                                    for (j in 0 until beamSize) if (eosMask[j] && finalized.size < beamSize) {
                                        val candidate = candidates[j]
                                        finalized.add(Finalized(candidate.score / (step + 1).toFloat(), prefixes[candidate.parent].toList() + FlareCacheLayout.EOS.toLong()))
                                    }
                                    val done = finalized.size == beamSize
                                    val selected = if (done) emptyList() else {
                                        val ranks = (0 until 2 * beamSize).map { j -> j + if (eosMask[j] || (step > 0 && j < beamSize && ignored[j])) 2 * beamSize else 0 }
                                        val indices = (0 until 2 * beamSize).sortedBy { ranks[it] }.take(beamSize)
                                        ignored = BooleanArray(beamSize) { ranks[indices[it]] >= 2 * beamSize }
                                        check(ignored.any { !it }) { "All active beam candidates ignored" }
                                        indices.map { candidates[it] }
                                    }
                                    selectionMs += elapsed(start)
                                    Resources().use { mergeInputs ->
                                        val mergeFeeds = mutableMapOf<String, OnnxTensor>()
                                        mergeFeeds["indices"] = mergeInputs.add(longTensor(selected.map { it.parent.toLong() }.toLongArray(), longArrayOf(selected.size.toLong())))
                                        val empty = if (results.size < beamSize / 5) mergeInputs.add(cacheTensor(FloatArray(0), 0, length)) else null
                                        for (chunk in 0 until beamSize / 5) for (suffix in suffixes) {
                                            mergeFeeds["chunk_${chunk}_$suffix"] = if (chunk < results.size) output(results[chunk], "present_$suffix", OnnxJavaType.FLOAT, cacheShape(if (step == 0) 1 else 5, length)) else empty!!
                                        }
                                        checkCancelled()
                                        start = System.nanoTime()
                                        val next = merge.run(mergeFeeds)
                                        mergeMs += elapsed(start)
                                        try {
                                            val flag = output(next, "cache_finite", OnnxJavaType.BOOL, longArrayOf()).value as Boolean
                                            check(flag) { "Non-finite decoder cache" }
                                            for (suffix in suffixes) output(next, "past_$suffix", OnnxJavaType.FLOAT, cacheShape(selected.size, length))
                                            checkCancelled()
                                        } catch (error: Throwable) {
                                            try { next.close() } catch (closeError: Throwable) { error.addSuppressed(closeError) }
                                            throw error
                                        }
                                        val old = cache
                                        cache = next
                                        old?.close()
                                    }
                                    if (done) {
                                        checkCancelled()
                                        return FlareBeamOutput(finalized.maxByOrNull { it.score }!!.tokens, FlareBeamTimings(encoderMs, crossMs, decoderMs, gatherMs, mergeMs, copyMs, selectionMs, elapsed(loopStart), calls, maximumRows), "$encoderBackend encoder; CPU decoder")
                                    }
                                    prefixes = selected.map { prefixes[it.parent] + it.word.toLong() }
                                    cumulative = FloatArray(beamSize) { selected[it].score }
                                }
                            }
                            error("Beam reached 128 prefixes without natural finalization")
                        } finally {
                            cache?.close()
                        }
                    }
                }
            }
        }
    }

    private data class Candidate(val parent: Int, val word: Int, val score: Float, val flatIndex: Int)
    private data class Finalized(val score: Float, val tokens: List<Long>)

    private fun bestCandidates(logits: FloatArray, cumulative: FloatArray, step: Int): List<Candidate> {
        val limit = 2 * beamSize
        val worstFirst = Comparator<Candidate> { a, b -> when {
            a.score < b.score -> -1
            a.score > b.score -> 1
            else -> b.flatIndex.compareTo(a.flatIndex)
        } }
        val best = PriorityQueue(limit, worstFirst)
        for (row in cumulative.indices) {
            val offset = row * FlareCacheLayout.VOCABULARY
            var maximum = Float.NEGATIVE_INFINITY
            for (word in 0 until FlareCacheLayout.VOCABULARY) maximum = maxOf(maximum, logits[offset + word])
            var sum = 0f
            for (word in 0 until FlareCacheLayout.VOCABULARY) sum += exp((logits[offset + word] - maximum).toDouble()).toFloat()
            val logSum = ln(sum.toDouble()).toFloat()
            for (word in 0 until FlareCacheLayout.VOCABULARY) {
                val score = if (word == FlareCacheLayout.PAD || (step == 0 && word == FlareCacheLayout.EOS)) Float.NEGATIVE_INFINITY else (logits[offset + word] - maximum - logSum) + cumulative[row]
                val flatIndex = offset + word
                if (best.size < limit || score > best.element().score || (score == best.element().score && flatIndex < best.element().flatIndex)) {
                    if (best.size == limit) best.remove()
                    best.add(Candidate(row, word, score, flatIndex))
                }
            }
        }
        check(best.size == limit)
        return best.sortedWith(worstFirst.reversed())
    }

    private fun cacheTensor(values: FloatArray, rows: Int, length: Int) = OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), cacheShape(rows, length))
    private fun longTensor(values: LongArray, shape: LongArray) = OnnxTensor.createTensor(environment, LongBuffer.wrap(values), shape)
    private fun cacheShape(rows: Int, length: Int) = longArrayOf(rows.toLong(), 4, length.toLong(), 192)
    private fun elapsed(start: Long) = (System.nanoTime() - start) / 1_000_000.0
    private fun finite(buffer: FloatBuffer, name: String) { while (buffer.hasRemaining()) check(buffer.get().isFinite()) { "Non-finite $name" } }
    private fun output(result: OrtSession.Result, name: String, type: OnnxJavaType, shape: LongArray): OnnxTensor {
        val tensor = result.get(name).orElseThrow { IllegalStateException("Missing output $name") } as? OnnxTensor ?: error("Output $name is not a tensor")
        val info = tensor.info as TensorInfo
        check(info.type == type && info.shape.contentEquals(shape)) { "Unexpected $name type or shape ${info.shape.contentToString()}" }
        return tensor
    }

    private class Resources : AutoCloseable {
        private val values = mutableListOf<AutoCloseable>()
        fun <T : AutoCloseable> add(value: T): T = value.also { values.add(it) }
        override fun close() {
            var failure: Throwable? = null
            for (value in values.asReversed()) try { value.close() } catch (error: Throwable) {
                if (failure == null) failure = error else failure.addSuppressed(error)
            }
            values.clear()
            failure?.let { throw it }
        }
    }
}
