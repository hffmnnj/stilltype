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
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeFlareCacheBenchmarkTest {
    private val directory get() = File(InstrumentationRegistry.getInstrumentation().targetContext.filesDir, "experimental-flare-native-cache")

    @Test
    fun nativeCacheDecoder_realFixture_matchesCompleteReferenceTokens() {
        val fixture = fixture()
        val frames = fixture.manifest.getInt("frames")
        fun visual(name: String): FloatArray {
            val file = required(name)
            assertEquals(frames.toLong() * 88 * 88 * 4, file.length())
            val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also { values ->
                buffer.get(values)
                assertTrue("Non-finite visual input", values.all { it.isFinite() })
            }
        }
        val face = visual("face.f32")
        val lip = visual("lip.f32")
        val expectedJson = fixture.manifest.getJSONArray("expectedTokenIds")
        val expected = List(expectedJson.length()) { expectedJson.getLong(it) }
        assertTrue(expected.size >= 2 && expected.first() == 2L && expected.last() == 2L)
        assertTrue(expected.all { it in 0L until 1000L })
        val pssBefore = Debug.getPss()
        val heapBefore = usedHeap()
        val result = FlareBeamDecoder(OrtEnvironment.getEnvironment(), fixture.files, fixture.beamSize).use {
            it.decode(face, lip, frames)
        }
        val t = result.timings
        val summary = JSONObject()
            .put("modelId", fixture.manifest.getString("modelId"))
            .put("beamSize", fixture.beamSize).put("frames", frames)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("sdk", Build.VERSION.SDK_INT).put("abi", Build.SUPPORTED_ABIS.joinToString())
            .put("provider", "CPU")
            .put("encoderMs", t.encoderMs).put("crossMs", t.crossMs)
            .put("decoderRunMs", t.decoderRunMs).put("cacheGatherMs", t.cacheGatherMs)
            .put("cacheMergeMs", t.cacheMergeMs).put("outputCopyMs", t.outputCopyMs)
            .put("candidateSelectionMs", t.candidateSelectionMs).put("completeDecoderLoopMs", t.completeDecoderLoopMs)
            .put("decoderCalls", t.decoderCalls).put("maximumNativeRows", t.maximumNativeRows)
            .put("sampledJavaHeapBeforeBytes", heapBefore).put("sampledJavaHeapAfterBytes", usedHeap())
            .put("sampledPssBeforeKiB", pssBefore).put("sampledPssAfterSessionsCloseKiB", Debug.getPss())
            .put("excludes", "camera, preprocessing, graph hashing; not a physical-phone claim")
        File(directory, "results.json").writeText(JSONObject(summary.toString()).put("generatedTokenIds", JSONArray(result.tokenIds)).toString())
        report("stilltypeNativeFlareCache", summary)
        assertEquals("Generated tokens differ from the complete reference", expected, result.tokenIds)
    }

    @Test
    fun nativeCacheHelpers_prefix128_expandGatherAndVetoUnselectedNan() {
        val fixture = fixture()
        val env = OrtEnvironment.getEnvironment()
        val suffixes = FlareCacheLayout.suffixes
        val rowSize = 4 * 128 * 192
        val seed = FloatArray(rowSize) { (it % 97).toFloat() }
        val heapBefore = usedHeap()
        val pssBefore = Debug.getPss()
        val start = System.nanoTime()
        var maximumSampledHeap = 0L
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            env.createSession(fixture.files.cacheMerge.absolutePath, options).use { merge ->
                env.createSession(fixture.files.cacheGather.absolutePath, options).use { gather ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(seed), shape(1)).use { row ->
                        OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(0)), shape(0)).use { empty ->
                            longTensor(env, LongArray(50), longArrayOf(50)).use { indices ->
                                val feeds = mutableMapOf("indices" to indices)
                                for (chunk in 0 until fixture.beamSize / 5) for (suffix in suffixes) feeds["chunk_${chunk}_$suffix"] = if (chunk == 0) row else empty
                                merge.run(feeds).use { expanded ->
                                    assertTrue(scalar(expanded, "cache_finite"))
                                    val gatherFeeds = mutableMapOf<String, OnnxTensor>()
                                    longTensor(env, longArrayOf(49, 0, 17, 49, 2), longArrayOf(5)).use { selection ->
                                        gatherFeeds["indices"] = selection
                                        for (suffix in suffixes) gatherFeeds["past_$suffix"] = tensor(expanded, "past_$suffix", shape(50))
                                        maximumSampledHeap = maxOf(maximumSampledHeap, usedHeap())
                                        // Pin only a small selected output; full caches remain native-owned.
                                        val selectedBuffer = ByteBuffer.allocateDirect(5 * rowSize * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                                        OnnxTensor.createTensor(env, selectedBuffer, shape(5)).use { pinned ->
                                            val remainingOutputs = suffixes.drop(1).map { "selected_$it" }.toSet()
                                            gather.run(gatherFeeds, remainingOutputs, mapOf("selected_0_key" to pinned)).use { selectedResult ->
                                                for (suffix in suffixes) tensor(selectedResult, "selected_$suffix", shape(5))
                                                maximumSampledHeap = maxOf(maximumSampledHeap, usedHeap())
                                                for (rowIndex in 0 until 5) for (element in 0 until rowSize) {
                                                    assertEquals(seed[element], selectedBuffer.get(rowIndex * rowSize + element), 0f)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            val distinctRows = FloatArray(rowSize * 2) { if (it < rowSize) 3f else 7f }
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(distinctRows), shape(2)).use { twoRows ->
                                longTensor(env, longArrayOf(1, 0, 1), longArrayOf(3)).use { indices ->
                                    val feeds = mutableMapOf("indices" to indices)
                                    for (suffix in suffixes) feeds["past_$suffix"] = twoRows
                                    val buffer = ByteBuffer.allocateDirect(3 * rowSize * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                                    OnnxTensor.createTensor(env, buffer, shape(3)).use { pinned ->
                                        gather.run(feeds, emptySet(), mapOf("selected_0_key" to pinned)).use {
                                            for (i in 0 until rowSize) {
                                                assertEquals(7f, buffer.get(i), 0f)
                                                assertEquals(3f, buffer.get(rowSize + i), 0f)
                                                assertEquals(7f, buffer.get(2 * rowSize + i), 0f)
                                            }
                                        }
                                    }
                                }
                            }
                            val nanRows = FloatArray(rowSize * 2) { if (it < rowSize) 1f else Float.NaN }
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(nanRows), shape(2)).use { twoRows ->
                                longTensor(env, longArrayOf(0), longArrayOf(1)).use { selected ->
                                    val feeds = mutableMapOf("indices" to selected)
                                    for (chunk in 0 until fixture.beamSize / 5) for (suffix in suffixes) feeds["chunk_${chunk}_$suffix"] = if (chunk == 0) twoRows else empty
                                    merge.run(feeds).use { result ->
                                        tensor(result, "past_0_key", shape(1))
                                        assertTrue("Unselected non-finite cache row must veto merge", !scalar(result, "cache_finite"))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        report("stilltypeNativeFlareCacheStress", JSONObject()
            .put("elapsedMs", (System.nanoTime() - start) / 1_000_000.0)
            .put("sampledJavaHeapBeforeBytes", heapBefore).put("maximumSampledJavaHeapWhileFullCacheLiveBytes", maximumSampledHeap)
            .put("sampledPssBeforeKiB", pssBefore).put("sampledPssAfterKiB", Debug.getPss())
            .put("excludes", "camera, preprocessing, graph hashing; not a physical-phone claim"))
    }

    private data class Fixture(val manifest: JSONObject, val beamSize: Int, val files: FlareGraphFiles)

    private fun fixture(): Fixture {
        val manifest = JSONObject(required("manifest.json").readText())
        val metadata = JSONObject(required("metadata.json").readText())
        val cache = JSONObject(required("cache.json").readText())
        val beam = manifest.getInt("beamSize")
        assertTrue(beam == 5 || beam == 50)
        for (json in listOf(manifest, metadata, cache)) {
            assertEquals(1, json.getInt("schemaVersion"))
            assertEquals("flare-base-transformer-2026-research", json.getString("modelId"))
            assertEquals(beam, json.getInt("beamSize"))
            assertEquals(6, json.getInt("cacheLayers"))
            assertEquals(4, json.getInt("cacheHeads"))
            assertEquals(192, json.getInt("cacheHeadDim"))
        }
        for (json in listOf(manifest, metadata)) {
            assertEquals("beam", json.getString("decoderMode"))
            assertEquals(512, json.getInt("featureSize"))
            assertEquals(1000, json.getInt("vocabularySize"))
            assertEquals(2, json.getInt("eosId"))
            assertEquals(1, json.getInt("padId"))
            assertEquals(128, json.getInt("maxPrefix"))
        }
        assertTrue(manifest.getInt("frames") in 1..200)
        assertEquals(5, cache.getInt("maximumNativeRows"))
        assertEquals(beam / 5, cache.getInt("cacheChunks"))
        fun graph(name: String, key: String, sources: List<JSONObject>): File {
            val file = required(name)
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { stream ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val count = stream.read(buffer)
                    if (count == -1) break
                    digest.update(buffer, 0, count)
                }
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            for (source in sources) {
                val size = source.getLong("${key}Bytes")
                val hash = source.getString("${key}Sha256")
                assertTrue(size > 0 && hash.matches(Regex("[0-9a-fA-F]{64}")))
                assertEquals("$name size", size, file.length())
                assertTrue("$name SHA-256", hash.equals(actual, ignoreCase = true))
            }
            return file
        }
        return Fixture(manifest, beam, FlareGraphFiles(
            graph("encoder.onnx", "encoder", listOf(manifest, metadata)),
            graph("cross.onnx", "cross", listOf(manifest, metadata)),
            graph("decoder-beam.onnx", "beamDecoder", listOf(manifest, metadata)),
            graph("cache-gather.onnx", "cacheGather", listOf(cache)),
            graph("cache-merge.onnx", "cacheMerge", listOf(cache)),
        ))
    }

    private fun required(name: String): File = File(directory, name).also { assertTrue("Missing fixture ${it.absolutePath}", it.isFile) }
    private fun shape(rows: Int) = longArrayOf(rows.toLong(), 4, 128, 192)
    private fun longTensor(env: OrtEnvironment, values: LongArray, shape: LongArray) = OnnxTensor.createTensor(env, LongBuffer.wrap(values), shape)
    private fun tensor(result: OrtSession.Result, name: String, shape: LongArray): OnnxTensor {
        val value = result.get(name).orElseThrow { AssertionError("Missing $name") } as OnnxTensor
        val info = value.info as TensorInfo
        assertEquals(OnnxJavaType.FLOAT, info.type)
        assertTrue("Unexpected $name shape", info.shape.contentEquals(shape))
        return value
    }
    private fun scalar(result: OrtSession.Result, name: String): Boolean {
        val value = result.get(name).orElseThrow { AssertionError("Missing $name") } as OnnxTensor
        val info = value.info as TensorInfo
        assertEquals(OnnxJavaType.BOOL, info.type)
        assertTrue(info.shape.isEmpty())
        return value.value as Boolean
    }
    private fun usedHeap() = Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() }
    private fun report(key: String, summary: JSONObject) {
        val text = summary.toString()
        Log.i("StilltypeBenchmark", text)
        InstrumentationRegistry.getInstrumentation().sendStatus(0, Bundle().apply { putString(key, text) })
    }
}
