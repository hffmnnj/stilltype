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
import java.nio.LongBuffer
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Inference-only research benchmark using app-private, externally supplied graphs and inputs. */
@RunWith(AndroidJUnit4::class)
class ExperimentalFlareBenchmarkTest {
    @Test
    fun exportedGraphs_decodeRealFixtureOnCpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.filesDir, "experimental-flare")
        fun fixture(name: String): File = File(directory, name).also {
            assertTrue("Missing fixture ${it.absolutePath}; supply the experimental FLARE fixture in app-private storage", it.isFile)
        }

        val manifest = JSONObject(fixture("manifest.json").readText())
        assertEquals(1, manifest.getInt("schemaVersion"))
        assertEquals("flare-base-transformer-2026-research", manifest.getString("modelId"))
        val frames = manifest.getInt("frames")
        assertTrue("frames must be in 1..200", frames in 1..200)
        assertEquals(512, manifest.getInt("featureSize"))
        assertEquals(1000, manifest.getInt("vocabularySize"))
        assertEquals(2, manifest.getInt("eosId"))
        assertEquals(1, manifest.getInt("padId"))
        assertEquals(128, manifest.getInt("maxPrefix"))

        fun checkedGraph(name: String, field: String): File {
            val graph = fixture(name)
            val expectedBytes = manifest.getLong("${field}Bytes")
            assertTrue("$field graph size must be positive", expectedBytes > 0)
            assertEquals("$field graph file size mismatch", expectedBytes, graph.length())
            val expectedHash = manifest.getString("${field}Sha256")
            assertTrue("$field SHA-256 must be 64 hex characters", expectedHash.matches(Regex("[0-9a-fA-F]{64}")))
            val digest = MessageDigest.getInstance("SHA-256")
            graph.inputStream().buffered().use { stream ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val count = stream.read(buffer)
                    if (count == -1) break
                    digest.update(buffer, 0, count)
                }
            }
            val actualHash = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            assertTrue("$field graph SHA-256 mismatch", expectedHash.equals(actualHash, ignoreCase = true))
            return graph
        }

        val encoderFile = checkedGraph("encoder.onnx", "encoder")
        val decoderFile = checkedGraph("decoder.onnx", "decoder")
        fun readVisual(name: String): ByteBuffer {
            val file = fixture(name)
            assertEquals("$name must contain frames * 88 * 88 little-endian float32 values", frames.toLong() * 88 * 88 * 4, file.length())
            return ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        }
        val face = readVisual("face.f32")
        val lip = readVisual("lip.f32")
        val environment = OrtEnvironment.getEnvironment()
        val pssBefore = Debug.getPss()
        var encoderLoadMs = 0.0
        var decoderLoadMs = 0.0
        var encoderMs = 0.0
        var decoderMs = 0.0
        var pssAfter = 0L
        val tokens = mutableListOf(2L)

        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            var start = System.nanoTime()
            val encoder = environment.createSession(encoderFile.absolutePath, options)
            encoderLoadMs = elapsedMs(start)
            encoder.use {
                start = System.nanoTime()
                val decoder = environment.createSession(decoderFile.absolutePath, options)
                decoderLoadMs = elapsedMs(start)
                decoder.use {
                    val visualShape = longArrayOf(1, 1, frames.toLong(), 88, 88)
                    OnnxTensor.createTensor(environment, face.asFloatBuffer(), visualShape).use { faceTensor ->
                        OnnxTensor.createTensor(environment, lip.asFloatBuffer(), visualShape).use { lipTensor ->
                            OnnxTensor.createTensor(environment, booleanArrayOf(true)).use { faceMask ->
                                OnnxTensor.createTensor(environment, booleanArrayOf(true)).use { lipMask ->
                                    OnnxTensor.createTensor(environment, arrayOf(BooleanArray(frames))).use { paddingMask ->
                                        start = System.nanoTime()
                                        val result = encoder.run(mapOf(
                                            "face" to faceTensor, "lip" to lipTensor,
                                            "face_mask" to faceMask, "lip_mask" to lipMask, "padding_mask" to paddingMask,
                                        ))
                                        encoderMs = elapsedMs(start)
                                        result.use {
                                            val encoded = output(result, "encoder_out", OnnxJavaType.FLOAT, longArrayOf(frames.toLong(), 1, 512))
                                            val encodedMask = output(result, "encoder_padding_mask", OnnxJavaType.BOOL, longArrayOf(1, frames.toLong()))
                                            val encoderValues = encoded.floatBuffer
                                            while (encoderValues.hasRemaining()) assertTrue("Non-finite encoder output", encoderValues.get().isFinite())

                                            var terminated = false
                                            start = System.nanoTime()
                                            for (step in 0 until 128) {
                                                OnnxTensor.createTensor(environment, LongBuffer.wrap(tokens.toLongArray()), longArrayOf(1, tokens.size.toLong())).use { prefix ->
                                                    decoder.run(mapOf("tokens" to prefix, "encoder_out" to encoded, "encoder_padding_mask" to encodedMask)).use { decoderResult ->
                                                        val logits = output(decoderResult, "next_logits", OnnxJavaType.FLOAT, longArrayOf(1, 1000)).floatBuffer
                                                        var bestId = -1
                                                        var bestValue = Float.NEGATIVE_INFINITY
                                                        for (id in 0 until 1000) {
                                                            val value = logits.get()
                                                            assertTrue("Non-finite decoder output", value.isFinite())
                                                            if (id != 1 && (step != 0 || id != 2) && value > bestValue) {
                                                                bestId = id
                                                                bestValue = value
                                                            }
                                                        }
                                                        assertTrue("No eligible decoder token", bestId >= 0)
                                                        tokens.add(bestId.toLong())
                                                        terminated = bestId == 2
                                                    }
                                                }
                                                if (terminated) break
                                            }
                                            decoderMs = elapsedMs(start)
                                            pssAfter = Debug.getPss()
                                            assertTrue("Decoder did not generate EOS within 128 iterations", terminated)
                                        }
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
            .put("generatedTokenCount", tokens.size - 1)
            .put("encoderLoadMs", encoderLoadMs)
            .put("decoderLoadMs", decoderLoadMs)
            .put("encoderMs", encoderMs)
            .put("fullPrefixDecoderMs", decoderMs)
            .put("sampledPssBeforeKiB", pssBefore)
            .put("sampledPssAfterKiB", pssAfter)
            .put("excludes", "camera, face analysis, model preparation, graph hashing, original app recognition")
        File(directory, "results.json").writeText(JSONObject(summary.toString()).put("generatedTokenIds", JSONArray(tokens)).toString())
        val summaryJson = summary.toString()
        Log.i("StilltypeBenchmark", summaryJson)
        instrumentation.sendStatus(0, Bundle().apply { putString("stilltypeExperimentalFlare", summaryJson) })
        val expectedJson = manifest.getJSONArray("expectedTokenIds")
        val expected = List(expectedJson.length()) { expectedJson.getLong(it) }
        assertTrue("Expected tokens must include initial and final EOS", expected.size >= 2 && expected.first() == 2L && expected.last() == 2L)
        assertTrue("Expected token IDs must be in vocabulary", expected.all { it in 0L until 1000L })
        assertTrue("Generated tokens differ from the independent reference; compare app-private results.json", tokens == expected)
    }

    private fun elapsedMs(start: Long): Double = (System.nanoTime() - start) / 1_000_000.0

    private fun output(result: OrtSession.Result, name: String, type: OnnxJavaType, shape: LongArray): OnnxTensor {
        val tensor = result.get(name).orElseThrow { AssertionError("Missing output $name") } as? OnnxTensor
            ?: throw AssertionError("Output $name is not a tensor")
        val info = tensor.info as TensorInfo
        assertEquals("Unexpected $name type", type, info.type)
        assertTrue("Unexpected $name shape ${info.shape.contentToString()}", info.shape.contentEquals(shape))
        return tensor
    }
}
