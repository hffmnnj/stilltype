package com.stilltype.android.engine

import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.stilltype.android.StilltypeApplication
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.core.QualityCrops
import com.stilltype.android.core.RecognitionProgress
import com.stilltype.android.core.RecognitionResult
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** App-private prepared pixel fixtures and real installed local models; not a camera accuracy test. */
@RunWith(AndroidJUnit4::class)
class QualityRecognitionBenchmarkTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private suspend fun recognizer(): AndroidRecognizer {
        val app = context.applicationContext as StilltypeApplication
        app.models.prepareBundled()
        app.qualityModels.prepareBundled()
        assertNotNull("Fast model preparation failed: ${app.models.status.value.error}", app.models.installedModels())
        assertNotNull("Quality model preparation failed: ${app.qualityModels.status.value.error}", app.qualityModels.installedModels())
        return AndroidRecognizer(context, app.models, app.qualityModels)
    }

    private data class SpeechFixture(val frames: List<MouthFrame>, val metadata: JSONObject)

    private fun speech(): SpeechFixture {
        val directory = File(context.filesDir, "quality-recognition")
        val metadata = JSONObject(File(directory, "manifest.json").readText())
        val count = metadata.getInt("frames")
        val timestamps = metadata.getJSONArray("timestampsMs")
        assertTrue("Speech needs 14..200 frames", count in 14..200)
        assertEquals(count, timestamps.length())
        fun bytes(name: String): ByteArray = File(directory, name).readBytes().also {
            assertEquals("$name byte count", count * 96 * 96, it.size)
        }
        val mouth = bytes("mouths.u8")
        val face = bytes("faces.u8")
        val lip = bytes("lips.u8")
        fun frame(data: ByteArray, index: Int) = data.copyOfRange(index * 96 * 96, (index + 1) * 96 * 96)
        val frames = List(count) { index ->
            MouthFrame(frame(mouth, index), timestamps.getLong(index), QualityCrops(frame(face, index), frame(lip, index)))
        }
        assertTrue("Speech must be sampled at 25 fps", frames.zipWithNext().all { (a, b) -> b.timestampMs - a.timestampMs == 40L })
        return SpeechFixture(frames, metadata)
    }

    private fun quiet(): List<MouthFrame> {
        val metadata = JSONObject(File(context.filesDir, "activity-gate-fixtures.json").readText()).getJSONObject("quiet")
        val timestamps = metadata.getJSONArray("timestampsMs")
        val pixels = File(context.filesDir, "activity-gate-quiet.raw").readBytes()
        assertEquals(timestamps.length() * 96 * 96, pixels.size)
        val arbitrary = QualityCrops(ByteArray(96 * 96), ByteArray(96 * 96))
        return List(timestamps.length()) { i ->
            MouthFrame(pixels.copyOfRange(i * 96 * 96, (i + 1) * 96 * 96), timestamps.getLong(i), arbitrary)
        }
    }

    @Test
    fun qualitySpeechAndQuiet_cpuRespectsReferenceAndFastVeto() = runBlocking {
        val fixture = speech()
        val quiet = quiet()
        recognizer().use { recognizer ->
            val progress = mutableListOf<RecognitionProgress>()
            val start = System.nanoTime()
            val spoken = recognizer.recognizeQuality(fixture.frames, ComputeMode.CPU) { progress.add(it) }
            report("speech-cpu", spoken, start)
            assertNull("Quality speech failed: ${spoken.error}", spoken.error)
            val expectedRaw = fixture.metadata.getString("expectedRaw")
            assertTrue("Reference text must not be empty", expectedRaw.isNotBlank())
            assertEquals(expectedRaw, spoken.raw)
            assertEquals(spoken.raw, spoken.text)
            assertEquals("CPU encoder; CPU decoder", spoken.backend)
            assertEquals(listOf(RecognitionProgress(1, 1)), progress)
            assertEquals("no-speech", recognizer.recognize(quiet, ComputeMode.CPU).error)
            val quietProgress = mutableListOf<RecognitionProgress>()
            val quietStart = System.nanoTime()
            val silent = recognizer.recognizeQuality(quiet, ComputeMode.CPU) { quietProgress.add(it) }
            report("quiet-cpu", silent, quietStart)
            assertEquals("no-speech", silent.error)
            assertTrue(silent.raw.isEmpty())
            assertEquals("Quality decoder not run (Fast activity veto)", silent.backend)
            assertEquals(listOf(RecognitionProgress(1, 1)), quietProgress)
        }
    }

    @Test
    fun qualityMemoryFailure_afterFirstWindowReturnsNoPartialTextAndReloads() = runBlocking {
        val fixture = speech()
        val expectedRaw = fixture.metadata.getString("expectedRaw")
        assertTrue("Reference text must not be empty", expectedRaw.isNotBlank())
        recognizer().use { recognizer ->
            val progress = mutableListOf<RecognitionProgress>()
            val failed = recognizer.recognizeQuality(fixture.frames, ComputeMode.CPU) {
                progress.add(it)
                throw OutOfMemoryError("Simulated allocation failure")
            }
            assertEquals(listOf(RecognitionProgress(1, 1)), progress)
            assertEquals("quality-memory-low", failed.error)
            assertEquals("", failed.raw)
            assertEquals("", failed.text)
            assertEquals(fixture.frames.size, failed.frames)
            assertEquals(fixture.frames.last().timestampMs - fixture.frames.first().timestampMs, failed.durationMs)

            val recovered = recognizer.recognizeQuality(fixture.frames, ComputeMode.CPU)
            assertNull("Quality retry failed: ${recovered.error}", recovered.error)
            assertEquals(expectedRaw, recovered.raw)
            assertEquals(expectedRaw, recovered.text)
        }
    }

    @Test
    fun qualitySpeech_autoReportsObservedOrUnverifiedEncoder() = runBlocking {
        val fixture = speech()
        recognizer().use { recognizer ->
            val start = System.nanoTime()
            val result = recognizer.recognizeQuality(fixture.frames, ComputeMode.AUTO)
            report("speech-auto", result, start)
            assertNull("Auto Quality failed: ${result.error}", result.error)
            val expectedRaw = fixture.metadata.getString("expectedRaw")
            assertTrue("Reference text must not be empty", expectedRaw.isNotBlank())
            assertEquals(expectedRaw, result.raw)
            assertTrue("Unexpected backend ${result.backend}", result.backend in setOf(
                "CPU encoder; CPU decoder", "NNAPI execution encoder; CPU decoder",
                "NNAPI execution + CPU encoder; CPU decoder", "NNAPI requested; execution unverified encoder; CPU decoder",
            ))
        }
    }

    @Test
    fun qualityCancellation_afterFirstWindowDoesNotReturnLateResult() = runBlocking {
        val fixture = speech()
        val long = List(3001) { i -> fixture.frames[i % fixture.frames.size].copy(timestampMs = i * 40L) }
        recognizer().use { recognizer ->
            val firstWindow = CompletableDeferred<Unit>()
            val returned = AtomicBoolean(false)
            val job = launch {
                recognizer.recognizeQuality(long, ComputeMode.CPU) { firstWindow.complete(Unit) }
                returned.set(true)
            }
            withTimeout(900_000L) { firstWindow.await() }
            job.cancelAndJoin()
            assertTrue(job.isCancelled)
            assertFalse("Cancelled Quality call returned a result", returned.get())
        }
    }

    @Test
    fun qualityFullSession_processesEveryWindowAndReports120Seconds() = runBlocking {
        val fixture = speech()
        val long = List(3001) { i ->
            val source = fixture.frames[i % fixture.frames.size]
            val crops = requireNotNull(source.quality)
            MouthFrame(source.pixels.copyOf(), i * 40L, QualityCrops(crops.face.copyOf(), crops.lip.copyOf()))
        }
        recognizer().use { recognizer ->
            val progress = mutableListOf<RecognitionProgress>()
            val start = System.nanoTime()
            val result = recognizer.recognizeQuality(long, ComputeMode.CPU) { progress.add(it) }
            report("full-session-cpu", result, start)
            assertNull("Full Quality session failed: ${result.error}", result.error)
            assertEquals(3001, result.frames)
            assertEquals(120_000L, result.durationMs)
            assertEquals((1..16).map { RecognitionProgress(it, 16) }, progress)
            assertTrue(result.raw.isNotBlank())
        }
    }

    private fun report(case: String, result: RecognitionResult, start: Long) {
        val summary = JSONObject()
            .put("case", case).put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("sdk", Build.VERSION.SDK_INT).put("backend", result.backend)
            .put("frames", result.frames).put("durationMs", result.durationMs)
            .put("processingMs", result.inferenceMs)
            .put("processingWallMs", (System.nanoTime() - start) / 1_000_000.0)
            .put("sampledJavaHeapBytes", Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() })
            .put("sampledPssWhileModelsOpenKiB", Debug.getPss())
            .put("success", result.error == null)
            .put("scope", "Prepared crops through actual recognition; excludes camera, capture and model preparation. Memory is a sample, not peak; not a physical-phone claim.")
        File(context.filesDir, "quality-recognition/results-$case.json").writeText(
            JSONObject(summary.toString()).put("raw", result.raw).put("error", result.error).toString(),
        )
        Log.i("StilltypeBenchmark", summary.toString())
        InstrumentationRegistry.getInstrumentation().sendStatus(0, Bundle().apply {
            putString("stilltypeQualityRecognition", summary.toString())
        })
    }
}
