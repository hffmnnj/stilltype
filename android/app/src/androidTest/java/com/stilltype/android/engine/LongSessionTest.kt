package com.stilltype.android.engine

import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.stilltype.android.StilltypeApplication
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.MAX_SESSION_FRAMES
import com.stilltype.android.core.MouthFrame
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Repeated public-video mouth crops exercise the full session, not accuracy. */
@RunWith(AndroidJUnit4::class)
class LongSessionTest {
    @Test(timeout = 180_000)
    fun realRecognizerProcessesTheEntireTwoMinuteTimeline(): Unit = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as StilltypeApplication
        app.models.prepareBundled()
        val shape = JSONObject(File(context.filesDir, "benchmark-input.json").readText()).getJSONArray("shape")
        val count = shape.getInt(1)
        assertTrue(count in 25..200)
        val values = ByteBuffer.wrap(File(context.filesDir, "benchmark-input.f32").readBytes())
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        assertEquals(count * 88 * 88, values.remaining())
        val source = List(count) {
            ByteArray(96 * 96) { 124.toByte() }.also { pixels ->
                for (y in 0 until 88) for (x in 0 until 88) {
                    pixels[(y + 4) * 96 + x + 4] =
                        ((values.get() * 0.165f + 0.421f) * 255f).roundToInt().coerceIn(0, 255).toByte()
                }
            }
        }
        val frames = List(MAX_SESSION_FRAMES) { index ->
            MouthFrame(source[index % source.size], index * 40L)
        }
        val started = System.nanoTime()
        AndroidRecognizer(context, app.models).use { recognizer ->
            val result = recognizer.recognize(frames, ComputeMode.CPU)
            assertNull(result.error)
            assertEquals(120_000L, result.durationMs)
            assertEquals(3_001, result.frames)
            assertTrue("Real CTC decoding returned no text", result.raw.isNotBlank())
            assertEquals(result.raw, result.text)
            Log.i("StilltypeBenchmark", "longSessionMs=${result.durationMs}; frames=${result.frames}; " +
                "inferenceMs=${result.inferenceMs}; wallMs=${(System.nanoTime() - started) / 1_000_000}; " +
                "provider=${result.backend}; pssKiB=${Debug.getPss()}; fixture=repeated-public-video")
        }
    }
}
