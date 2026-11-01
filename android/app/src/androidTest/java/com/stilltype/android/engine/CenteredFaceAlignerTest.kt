package com.stilltype.android.engine

import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.stilltype.android.StilltypeApplication
import com.stilltype.android.core.AlignmentFrame
import com.stilltype.android.core.FrameAnalysis
import com.stilltype.android.core.ImagePoint
import com.stilltype.android.core.MouthFrame
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CenteredFaceAlignerTest {
    private val fixture: File
        get() = File(InstrumentationRegistry.getInstrumentation().targetContext.filesDir, "quality-alignment")

    private fun points(bytes: ByteArray): List<ImagePoint> {
        val floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return List(4) { ImagePoint(floats.get(), floats.get()) }
    }

    private fun reference(): List<ImagePoint> = points(File(fixture, "reference.f32").readBytes())

    @Test
    fun optInDetectorPayloadPreservesFastPixelsAndOwnsItsGrayscale() = runBlocking {
        val manifest = JSONObject(File(fixture, "manifest.json").readText())
        val width = manifest.getInt("width")
        val height = manifest.getInt("height")
        val gray = ByteArray(width * height)
        DataInputStream(File(fixture, "frames.gray").inputStream()).use { it.readFully(gray) }
        val bitmap = Bitmap.createBitmap(IntArray(gray.size) {
            val value = gray[it].toInt() and 255
            Color.rgb(value, value, value)
        }, width, height, Bitmap.Config.ARGB_8888)
        try {
            val app = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext as StilltypeApplication
            app.models.prepareBundled()
            val installed = requireNotNull(app.models.installedModels()) { "Prepare the bundled Fast model before this detector test" }
            FaceAnalyzer(installed.directory, OrtEnvironment.getEnvironment()).use { analyzer ->
                val fast = analyzer.analyze(bitmap, 1_000L)
                val quality = analyzer.analyze(bitmap, 1_040L, quality = true)
                assertTrue("The real fixture must contain a detected face", fast.faceDetected && quality.faceDetected)
                assertEquals(null, fast.alignment)
                assertArrayEquals(requireNotNull(fast.mouth).pixels, requireNotNull(quality.mouth).pixels)
                val alignment = requireNotNull(quality.alignment)
                assertEquals(640, alignment.width)
                assertEquals(480, alignment.height)
                assertEquals(4, alignment.landmarks.size)
                assertArrayEquals(gray, alignment.grayscale)
                val black = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                try {
                    analyzer.analyze(black, 1_080L, quality = true)
                } finally {
                    black.recycle()
                }
                assertArrayEquals("A later detector call must not mutate an accepted payload", gray, alignment.grayscale)
            }
        } finally {
            bitmap.recycle()
        }
    }

    @Test
    fun fullClipAndGapSegmentsMatchIndependentDesktopCrops() {
        val manifest = JSONObject(File(fixture, "manifest.json").readText())
        val count = manifest.getInt("frames")
        val width = manifest.getInt("width")
        val height = manifest.getInt("height")
        val split = manifest.getInt("split")
        assertTrue(count >= 30 && split > 6 && count - split > 6)
        for (gap in listOf(false, true)) {
            val pending = mutableMapOf<Long, ByteArray>()
            var emitted = 0
            DataInputStream(File(fixture, "frames.gray").inputStream().buffered()).use { images ->
                DataInputStream(File(fixture, "landmarks.f32").inputStream().buffered()).use { landmarks ->
                    DataInputStream(File(fixture, if (gap) "gap-faces.gray" else "faces.gray").inputStream().buffered()).use { faces ->
                        DataInputStream(File(fixture, if (gap) "gap-lips.gray" else "lips.gray").inputStream().buffered()).use { lips ->
                            CenteredFaceAligner(reference()).use { aligner ->
                                fun verify(outputs: List<MouthFrame>) {
                                    for (output in outputs) {
                                        assertEquals(emitted * 40L, output.timestampMs)
                                        assertSame(pending.remove(output.timestampMs), output.pixels)
                                        val quality = requireNotNull(output.quality)
                                        val expectedFace = ByteArray(96 * 96).also { faces.readFully(it) }
                                        val expectedLip = ByteArray(96 * 96).also { lips.readFully(it) }
                                        assertArrayEquals("face at $emitted (gap=$gap)", expectedFace, quality.face)
                                        assertArrayEquals("lip at $emitted (gap=$gap)", expectedLip, quality.lip)
                                        emitted++
                                    }
                                }
                                for (i in 0 until count) {
                                    if (gap && i == split) verify(aligner.offer(FrameAnalysis()))
                                    val gray = ByteArray(width * height).also { images.readFully(it) }
                                    val pointBytes = ByteArray(32).also { landmarks.readFully(it) }
                                    val pixels = ByteArray(96 * 96) { (i % 251).toByte() }
                                    val timestamp = i * 40L
                                    pending[timestamp] = pixels
                                    verify(aligner.offer(FrameAnalysis(
                                        mouth = MouthFrame(pixels, timestamp),
                                        alignment = AlignmentFrame(gray, width, height, points(pointBytes)),
                                    )))
                                    assertTrue(aligner.retainedFrameCount <= 13)
                                }
                                verify(aligner.finish())
                                assertEquals(0, aligner.retainedFrameCount)
                                assertTrue(pending.isEmpty())
                                assertEquals(count, emitted)
                                assertEquals(-1, faces.read())
                                assertEquals(-1, lips.read())
                                assertEquals(-1, images.read())
                                assertEquals(-1, landmarks.read())
                            }
                        }
                    }
                }
            }
        }
    }

    @Test(timeout = 300_000)
    fun resetDropsPendingFramesAndTwoMinuteSessionKeepsBoundedTail() {
        val ref = reference()
        val gray = ByteArray(256 * 256) { (it % 251).toByte() }
        val pixels = ByteArray(96 * 96) { 42 }
        fun observation(timestamp: Long) = FrameAnalysis(
            mouth = MouthFrame(pixels, timestamp),
            alignment = AlignmentFrame(gray, 256, 256, ref),
        )
        CenteredFaceAligner(ref).use { aligner ->
            assertEquals(1, aligner.offer(observation(-80)).size)
            assertTrue(aligner.offer(observation(-40)).isEmpty())
            aligner.reset()
            assertEquals(0, aligner.retainedFrameCount)
            var emitted = 0
            fun verify(outputs: List<MouthFrame>) {
                for (output in outputs) {
                    assertEquals(emitted * 40L, output.timestampMs)
                    assertSame(pixels, output.pixels)
                    assertNotNull(output.quality)
                    emitted++
                }
            }
            for (index in 0..3000) {
                verify(aligner.offer(observation(index * 40L)))
                assertTrue("frame $index", aligner.retainedFrameCount <= 13)
            }
            assertTrue("finish must emit the delayed tail", emitted < 3001)
            verify(aligner.finish())
            assertEquals(3001, emitted)
            assertEquals(0, aligner.retainedFrameCount)
            assertTrue(aligner.finish().isEmpty())
        }
    }
}
