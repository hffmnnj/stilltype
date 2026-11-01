package com.stilltype.android.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.stilltype.android.StilltypeApplication
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Real-model CPU smoke benchmark; the supplied public fixture is not an accuracy test. */
@RunWith(AndroidJUnit4::class)
class MobileBenchmarkTest {
    @Test
    fun analyze_uniformGrayFrame_reportsBrightnessWithoutDetectorError() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val models = (context.applicationContext as StilltypeApplication).models
        models.prepareBundled()
        assertNotNull("Bundled model preparation failed: ${models.status.value.error}", models.installedModels())
        val bitmap = Bitmap.createBitmap(320, 480, Bitmap.Config.ARGB_8888)
        try {
            bitmap.eraseColor(Color.GRAY)
            AndroidRecognizer(context.applicationContext, models).use { recognizer ->
                val analysis = recognizer.analyze(bitmap, 1_000L)
                assertFalse("Uniform gray frame must not detect a face", analysis.faceDetected)
                assertTrue(
                    "Uniform gray frame must report real brightness, was ${analysis.brightness}",
                    analysis.brightness.isFinite() && analysis.brightness > 0.2f,
                )
                assertFalse(
                    "Face detector must initialize, was hint=${analysis.hint}",
                    analysis.hint.contains("Face detector unavailable"),
                )
            }
        } finally {
            bitmap.recycle()
        }
    }

    @Test
    fun bundledModelRunsRealFixtureWithFiniteCtcLogits() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val models = (context.applicationContext as StilltypeApplication).models
        models.prepareBundled()
        val installed = models.installedModels()
        assertNotNull("Bundled model preparation failed: ${models.status.value.error}", installed)
        val model = requireNotNull(installed)
        val inputFile = File(context.filesDir, "benchmark-input.f32")
        val metadata = JSONObject(File(context.filesDir, "benchmark-input.json").readText())
        val jsonShape = metadata.getJSONArray("shape")
        val shape = LongArray(jsonShape.length()) { jsonShape.getLong(it) }
        assertEquals(4, shape.size)
        assertEquals(1L, shape[0])
        assertTrue(shape[1] in 1..model.manifest.maxFrames.toLong())
        assertEquals(88L, shape[2])
        assertEquals(88L, shape[3])
        assertEquals("float32-le", metadata.getString("dtype"))
        val bytes = inputFile.readBytes()
        assertEquals(shape.reduce(Long::times) * 4, bytes.size.toLong())
        val input = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val environment = OrtEnvironment.getEnvironment()
        val pssBefore = Debug.getPss()
        val loadStart = System.nanoTime()
        val session = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            environment.createSession(File(model.directory, "usr2-ctc-int8.onnx").absolutePath, options)
        }
        val loadMs = (System.nanoTime() - loadStart) / 1_000_000.0
        session.use {
            OnnxTensor.createTensor(environment, input, shape).use { tensor ->
                session.run(mapOf(model.manifest.inputName to tensor)).use { }
                val start = System.nanoTime()
                session.run(mapOf(model.manifest.inputName to tensor)).use { result ->
                    val warmMs = (System.nanoTime() - start) / 1_000_000.0
                    val logits = result.get(model.manifest.outputName).orElseThrow { AssertionError("Missing CTC logits") } as OnnxTensor
                    val actualShape = (logits.info as TensorInfo).shape
                    assertTrue("Unexpected CTC output shape ${actualShape.contentToString()}",
                        actualShape.contentEquals(longArrayOf(1, shape[1], 1049)))
                    val values = logits.floatBuffer
                    assertEquals(shape[1] * 1049, values.remaining().toLong())
                    while (values.hasRemaining()) assertTrue("Non-finite CTC logit", values.get().isFinite())
                    val summary = "device=${Build.MANUFACTURER} ${Build.MODEL}; sdk=${Build.VERSION.SDK_INT}; " +
                        "abi=${Build.SUPPORTED_ABIS.joinToString()}; provider=CPU; frames=${shape[1]}; " +
                        "loadMs=$loadMs; warmMs=$warmMs; pssBeforeKiB=$pssBefore; pssAfterKiB=${Debug.getPss()}"
                    Log.i("StilltypeBenchmark", summary)
                    instrumentation.sendStatus(0, Bundle().apply { putString("stilltypeBenchmark", summary) })
                }
            }
        }
    }
}
