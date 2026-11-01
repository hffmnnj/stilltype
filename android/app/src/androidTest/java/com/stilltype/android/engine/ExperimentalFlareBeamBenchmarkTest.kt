package com.stilltype.android.engine

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.security.MessageDigest
import kotlin.math.exp
import kotlin.math.ln
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Inference-only benchmark of the experimental batched cached decoder. */
@RunWith(AndroidJUnit4::class)
class ExperimentalFlareBeamBenchmarkTest {
    @Test
    fun batchedGraphs_decodeRealFixtureOnCpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.filesDir, "experimental-flare-beam")
        fun fixture(name: String): File = File(directory, name).also {
            assertTrue("Missing fixture ${it.absolutePath}", it.isFile)
        }

        val manifest = JSONObject(fixture("manifest.json").readText())
        assertEquals(1, manifest.getInt("schemaVersion"))
        assertEquals("flare-base-transformer-2026-research", manifest.getString("modelId"))
        assertEquals("beam", manifest.getString("decoderMode"))
        assertEquals(BEAM, manifest.getInt("beamSize"))
        val frames = manifest.getInt("frames")
        assertTrue("frames must be in 1..200", frames in 1..200)
        assertEquals(512, manifest.getInt("featureSize"))
        assertEquals(VOCABULARY, manifest.getInt("vocabularySize"))
        assertEquals(EOS, manifest.getInt("eosId"))
        assertEquals(PAD, manifest.getInt("padId"))
        assertEquals(MAX_PREFIX, manifest.getInt("maxPrefix"))
        assertEquals(LAYERS, manifest.getInt("cacheLayers"))
        assertEquals(HEADS, manifest.getInt("cacheHeads"))
        assertEquals(HEAD_DIM, manifest.getInt("cacheHeadDim"))
        val expectedJson = manifest.getJSONArray("expectedTokenIds")
        val expected = List(expectedJson.length()) { expectedJson.getLong(it) }
        assertTrue("Expected tokens must include initial and final EOS", expected.size >= 2 && expected.first() == EOS.toLong() && expected.last() == EOS.toLong())
        assertTrue("Expected token IDs must be in vocabulary", expected.all { it in 0L until VOCABULARY.toLong() })

        fun checkedGraph(name: String, field: String): File {
            val file = fixture(name)
            val size = manifest.getLong("${field}Bytes")
            assertTrue("$field graph size must be positive", size > 0)
            assertEquals("$field graph file size mismatch", size, file.length())
            val expectedHash = manifest.getString("${field}Sha256")
            assertTrue("$field SHA-256 must be 64 hex characters", expectedHash.matches(Regex("[0-9a-fA-F]{64}")))
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { stream ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val count = stream.read(buffer)
                    if (count == -1) break
                    digest.update(buffer, 0, count)
                }
            }
            val actualHash = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            assertTrue("$field graph SHA-256 mismatch", expectedHash.equals(actualHash, ignoreCase = true))
            return file
        }
        val encoderFile = checkedGraph("encoder.onnx", "encoder")
        val crossFile = checkedGraph("cross.onnx", "cross")
        val decoderFile = checkedGraph("decoder-beam.onnx", "beamDecoder")
        fun readVisual(name: String): FloatBuffer {
            val file = fixture(name)
            assertEquals("$name must contain frames * 88 * 88 float32 values", frames.toLong() * 88 * 88 * 4, file.length())
            val values = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            assertFinite(values.duplicate(), name)
            return values
        }
        val face = readVisual("face.f32")
        val lip = readVisual("lip.f32")
        val environment = OrtEnvironment.getEnvironment()
        val pssBefore = Debug.getPss()
        var encoderLoadMs = 0.0
        var crossLoadMs = 0.0
        var decoderLoadMs = 0.0
        var encoderMs = 0.0
        var crossMs = 0.0
        var decoderRunMs = 0.0
        var decoderLoopMs = 0.0
        var pssAfter = 0L
        var generated = emptyList<Long>()
        val suffixes = (0 until LAYERS).flatMap { listOf("${it}_key", "${it}_value") }

        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            var start = System.nanoTime()
            environment.createSession(encoderFile.absolutePath, options).use { encoder ->
                encoderLoadMs = elapsedMs(start)
                start = System.nanoTime()
                environment.createSession(crossFile.absolutePath, options).use { cross ->
                    crossLoadMs = elapsedMs(start)
                    start = System.nanoTime()
                    environment.createSession(decoderFile.absolutePath, options).use { decoder ->
                        decoderLoadMs = elapsedMs(start)
                        assertEquals(setOf("face", "lip", "face_mask", "lip_mask", "padding_mask"), encoder.inputNames)
                        assertEquals(setOf("encoder_out", "encoder_padding_mask"), encoder.outputNames)
                        assertEquals(setOf("encoder_out"), cross.inputNames)
                        assertEquals(suffixes.map { "cross_$it" }.toSet(), cross.outputNames)
                        assertEquals(setOf("tokens", "encoder_padding_mask") + suffixes.flatMap { listOf("past_$it", "cross_$it") }, decoder.inputNames)
                        assertEquals(setOf("next_logits") + suffixes.map { "present_$it" }, decoder.outputNames)
                        Tensors().use { input ->
                            val visualShape = longArrayOf(1, 1, frames.toLong(), 88, 88)
                            val feeds = mapOf(
                                "face" to input.add(OnnxTensor.createTensor(environment, face, visualShape)),
                                "lip" to input.add(OnnxTensor.createTensor(environment, lip, visualShape)),
                                "face_mask" to input.add(OnnxTensor.createTensor(environment, booleanArrayOf(true))),
                                "lip_mask" to input.add(OnnxTensor.createTensor(environment, booleanArrayOf(true))),
                                "padding_mask" to input.add(OnnxTensor.createTensor(environment, arrayOf(BooleanArray(frames)))),
                            )
                            start = System.nanoTime()
                            val encoderResult = encoder.run(feeds)
                            encoderMs = elapsedMs(start)
                            encoderResult.use {
                                val encoded = output(encoderResult, "encoder_out", OnnxJavaType.FLOAT, longArrayOf(frames.toLong(), 1, 512))
                                val encodedMask = output(encoderResult, "encoder_padding_mask", OnnxJavaType.BOOL, longArrayOf(1, frames.toLong()))
                                assertFinite(encoded.floatBuffer, "encoder output")
                                @Suppress("UNCHECKED_CAST")
                                val maskRow = (encodedMask.value as Array<BooleanArray>)[0].copyOf()
                                start = System.nanoTime()
                                val crossResult = cross.run(mapOf("encoder_out" to encoded))
                                crossMs = elapsedMs(start)
                                crossResult.use {
                                    val projected = suffixes.associateWith { suffix ->
                                        val name = "cross_$suffix"
                                        val tensor = output(crossResult, name, OnnxJavaType.FLOAT, longArrayOf(1, HEADS.toLong(), frames.toLong(), HEAD_DIM.toLong()))
                                        readFinite(tensor, name)
                                    }
                                    // Cross projections and the padding mask do not change across steps.
                                    Tensors().use { fixed ->
                                        val crossOne = projected.mapValues { (_, row) ->
                                            fixed.add(floatTensor(environment, row, 1, frames))
                                        }
                                        val crossFive = projected.mapValues { (_, row) ->
                                            val repeated = FloatArray(row.size * BEAM)
                                            repeat(BEAM) { row.copyInto(repeated, it * row.size) }
                                            fixed.add(floatTensor(environment, repeated, BEAM, frames))
                                        }
                                        val maskOne = fixed.add(OnnxTensor.createTensor(environment, arrayOf(maskRow)))
                                        val maskFive = fixed.add(OnnxTensor.createTensor(environment, Array(BEAM) { maskRow.copyOf() }))
                                        var prefixes = listOf(longArrayOf(EOS.toLong()))
                                        var cumulative = floatArrayOf(0f)
                                        var ignored = BooleanArray(BEAM)
                                        var cache = suffixes.associateWith { FloatArray(0) }
                                        val finalized = mutableListOf<Finalized>()
                                        val loopStart = System.nanoTime()
                                        for (step in 0 until MAX_PREFIX) {
                                            val batch = prefixes.size
                                            val length = prefixes[0].size
                                            assertTrue("Unexpected beam batch", batch == if (step == 0) 1 else BEAM)
                                            Tensors().use { stepInputs ->
                                                val tokenValues = LongArray(batch * length) { prefixes[it / length][it % length] }
                                                val decoderFeeds = mutableMapOf(
                                                    "tokens" to stepInputs.add(OnnxTensor.createTensor(environment, LongBuffer.wrap(tokenValues), longArrayOf(batch.toLong(), length.toLong()))),
                                                    "encoder_padding_mask" to if (batch == 1) maskOne else maskFive,
                                                )
                                                val crossInputs = if (batch == 1) crossOne else crossFive
                                                for (suffix in suffixes) {
                                                    decoderFeeds["cross_$suffix"] = crossInputs.getValue(suffix)
                                                    decoderFeeds["past_$suffix"] = stepInputs.add(floatTensor(environment, cache.getValue(suffix), batch, length - 1))
                                                }
                                                start = System.nanoTime()
                                                val result = decoder.run(decoderFeeds)
                                                decoderRunMs += elapsedMs(start)
                                                result.use {
                                                    val logits = readFinite(output(result, "next_logits", OnnxJavaType.FLOAT, longArrayOf(batch.toLong(), VOCABULARY.toLong())), "next_logits")
                                                    val present = suffixes.associateWith { suffix ->
                                                        val name = "present_$suffix"
                                                        readFinite(output(result, name, OnnxJavaType.FLOAT, longArrayOf(batch.toLong(), HEADS.toLong(), length.toLong(), HEAD_DIM.toLong())), name)
                                                    }
                                                    val candidates = bestCandidates(logits, cumulative, step)
                                                    val eosMask = BooleanArray(2 * BEAM) { j ->
                                                        candidates[j].word == EOS && candidates[j].score.isFinite() && (step == 0 || j >= BEAM || !ignored[j])
                                                    }
                                                    for (j in 0 until BEAM) {
                                                        if (eosMask[j] && finalized.size < BEAM) {
                                                            val candidate = candidates[j]
                                                            finalized.add(Finalized(candidate.score / (step + 1).toFloat(), prefixes[candidate.parent].toList() + EOS.toLong()))
                                                        }
                                                    }
                                                    if (finalized.size == BEAM) {
                                                        generated = finalized.maxByOrNull { it.score }!!.tokens
                                                    } else {
                                                        val ranks = (0 until 2 * BEAM).map { j ->
                                                            val inactive = eosMask[j] || (step > 0 && j < BEAM && ignored[j])
                                                            j + if (inactive) 2 * BEAM else 0
                                                        }
                                                        val selected = (0 until 2 * BEAM).sortedBy { ranks[it] }.take(BEAM)
                                                        ignored = BooleanArray(BEAM) { ranks[selected[it]] >= 2 * BEAM }
                                                        assertTrue("All active beam candidates ignored", ignored.any { !it })
                                                        val selectedCandidates = selected.map { candidates[it] }
                                                        prefixes = selectedCandidates.map { prefixes[it.parent] + it.word.toLong() }
                                                        cumulative = FloatArray(BEAM) { selectedCandidates[it].score }
                                                        val rowSize = HEADS * length * HEAD_DIM
                                                        cache = present.mapValues { (_, values) ->
                                                            FloatArray(BEAM * rowSize).also { reordered ->
                                                                selectedCandidates.forEachIndexed { row, candidate ->
                                                                    values.copyInto(reordered, row * rowSize, candidate.parent * rowSize, (candidate.parent + 1) * rowSize)
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            if (generated.isNotEmpty()) break
                                        }
                                        decoderLoopMs = elapsedMs(loopStart)
                                        pssAfter = Debug.getPss()
                                        assertTrue("Beam reached 128 prefixes without natural finalization", generated.isNotEmpty())
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        val summary = JSONObject()
            .put("modelId", manifest.getString("modelId"))
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("sdk", Build.VERSION.SDK_INT)
            .put("abi", Build.SUPPORTED_ABIS.joinToString())
            .put("provider", "CPU")
            .put("frames", frames)
            .put("beamSize", BEAM)
            .put("generatedTokenCount", generated.size - 1)
            .put("encoderLoadMs", encoderLoadMs)
            .put("crossLoadMs", crossLoadMs)
            .put("beamDecoderLoadMs", decoderLoadMs)
            .put("encoderMs", encoderMs)
            .put("crossMs", crossMs)
            .put("nativeDecoderRunMs", decoderRunMs)
            .put("completeDecoderLoopMs", decoderLoopMs)
            .put("sampledPssBeforeKiB", pssBefore)
            .put("sampledPssAfterKiB", pssAfter)
            .put("excludes", "camera, preprocessing, model preparation, graph hashing; not a physical-phone claim")
        File(directory, "results.json").writeText(JSONObject(summary.toString()).put("generatedTokenIds", JSONArray(generated)).toString())
        val summaryJson = summary.toString()
        Log.i("StilltypeBenchmark", summaryJson)
        instrumentation.sendStatus(0, Bundle().apply { putString("stilltypeExperimentalFlareBeam", summaryJson) })
        assertEquals("Generated tokens differ from independent reference; compare app-private results.json", expected, generated)
    }

    private data class Candidate(val parent: Int, val word: Int, val score: Float, val flatIndex: Int)
    private data class Finalized(val score: Float, val tokens: List<Long>)

    private fun bestCandidates(logits: FloatArray, cumulative: FloatArray, step: Int): List<Candidate> {
        val best = mutableListOf<Candidate>()
        for (row in cumulative.indices) {
            val offset = row * VOCABULARY
            var maximum = Float.NEGATIVE_INFINITY
            for (word in 0 until VOCABULARY) maximum = maxOf(maximum, logits[offset + word])
            var sum = 0f
            for (word in 0 until VOCABULARY) sum += exp((logits[offset + word] - maximum).toDouble()).toFloat()
            val logSum = ln(sum.toDouble()).toFloat()
            for (word in 0 until VOCABULARY) {
                val score = if (word == PAD || (step == 0 && word == EOS)) Float.NEGATIVE_INFINITY
                else (logits[offset + word] - maximum - logSum) + cumulative[row]
                val flatIndex = offset + word
                val candidate = Candidate(row, word, score, flatIndex)
                val position = best.indexOfFirst { score > it.score || (score == it.score && flatIndex < it.flatIndex) }
                if (position >= 0) best.add(position, candidate) else if (best.size < 2 * BEAM) best.add(candidate)
                if (best.size > 2 * BEAM) best.removeAt(best.lastIndex)
            }
        }
        assertEquals(2 * BEAM, best.size)
        return best
    }

    private fun floatTensor(environment: OrtEnvironment, values: FloatArray, batch: Int, length: Int): OnnxTensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), longArrayOf(batch.toLong(), HEADS.toLong(), length.toLong(), HEAD_DIM.toLong()))

    private fun readFinite(tensor: OnnxTensor, name: String): FloatArray {
        val buffer = tensor.floatBuffer
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        for (value in values) if (!value.isFinite()) throw AssertionError("Non-finite $name")
        return values
    }

    private fun assertFinite(values: FloatBuffer, name: String) {
        while (values.hasRemaining()) if (!values.get().isFinite()) throw AssertionError("Non-finite $name")
    }

    private fun output(result: OrtSession.Result, name: String, type: OnnxJavaType, shape: LongArray): OnnxTensor {
        val tensor = result.get(name).orElseThrow { AssertionError("Missing output $name") } as? OnnxTensor
            ?: throw AssertionError("Output $name is not a tensor")
        val info = tensor.info as TensorInfo
        assertEquals("Unexpected $name type", type, info.type)
        assertTrue("Unexpected $name shape ${info.shape.contentToString()}", info.shape.contentEquals(shape))
        return tensor
    }

    private fun elapsedMs(start: Long): Double = (System.nanoTime() - start) / 1_000_000.0

    private class Tensors : AutoCloseable {
        private val values = mutableListOf<OnnxTensor>()
        fun add(tensor: OnnxTensor): OnnxTensor = tensor.also { values.add(it) }
        override fun close() {
            var failure: Throwable? = null
            for (tensor in values.asReversed()) {
                try {
                    tensor.close()
                } catch (error: Throwable) {
                    if (failure == null) failure = error else failure.addSuppressed(error)
                }
            }
            failure?.let { throw it }
        }
    }

    private companion object {
        const val BEAM = 5
        const val VOCABULARY = 1000
        const val EOS = 2
        const val PAD = 1
        const val MAX_PREFIX = 128
        const val LAYERS = 6
        const val HEADS = 4
        const val HEAD_DIM = 192
    }
}
