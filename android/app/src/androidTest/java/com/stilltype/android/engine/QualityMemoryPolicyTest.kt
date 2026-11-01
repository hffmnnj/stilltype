package com.stilltype.android.engine

import android.app.ActivityManager
import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.InstalledModels
import com.stilltype.android.core.InstalledQualityModels
import com.stilltype.android.core.MIN_QUALITY_MEMORY_BYTES
import com.stilltype.android.core.ModelRepository
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.core.QUALITY_MEMORY_REQUIREMENT
import com.stilltype.android.core.QualityCrops
import com.stilltype.android.core.QualityModelRepository
import com.stilltype.android.data.BundledQualityModelRepository
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class QualityMemoryPolicyTest {
    @Test
    fun lowRamDevice_refusesPreparationAndRecognitionWithoutLoadingModels() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val info = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
        assumeTrue("Requires a device reporting positive usable RAM below the Quality threshold",
            info.totalMem > 0 && info.totalMem < MIN_QUALITY_MEMORY_BYTES)

        val requiredAssets = setOf("metadata.json", "cache.json", "encoder.onnx", "cross.onnx",
            "decoder-beam.onnx", "cache-gather.onnx", "cache-merge.onnx")
        val bundled = context.assets.list("quality")?.toSet()?.containsAll(requiredAssets) == true
        val repository = BundledQualityModelRepository(context)
        val initial = repository.status.value
        assertEquals(bundled, initial.bundled)
        assertEquals(QUALITY_MEMORY_REQUIREMENT, initial.unavailableReason)
        assertFalse(initial.installed)
        assertFalse(initial.preparing)

        val directory = File(context.filesDir, "models/quality")
        fun files(): Map<String, Pair<Long, Long>> = if (!directory.exists()) emptyMap() else
            directory.walkTopDown().associate { it.relativeTo(directory).path to (it.length() to it.lastModified()) }
        val before = files()
        repository.prepareBundled()
        val after = repository.status.value
        assertEquals(bundled, after.bundled)
        assertEquals(QUALITY_MEMORY_REQUIREMENT, after.unavailableReason)
        assertEquals(QUALITY_MEMORY_REQUIREMENT, after.error)
        assertFalse(after.installed)
        assertFalse(after.preparing)
        assertNull(repository.installedModels())
        assertEquals("Preparation must not change existing Quality files", before, files())

        val fast = object : ModelRepository {
            override val status = MutableStateFlow(ModelStatus())
            override suspend fun prepareBundled() = error("Fast preparation must not run")
            override fun installedModels(): InstalledModels? = error("Fast model must not be loaded")
        }
        val quality = object : QualityModelRepository {
            override val status = repository.status
            override suspend fun prepareBundled() = error("Quality preparation must not run")
            override fun installedModels(): InstalledQualityModels? = error("Quality model must not be loaded")
        }
        val crops = QualityCrops(ByteArray(96 * 96), ByteArray(96 * 96))
        val frames = List(14) { index -> MouthFrame(ByteArray(96 * 96), index * 40L, crops) }
        AndroidRecognizer(context, fast, quality).use { recognizer ->
            val result = recognizer.recognizeQuality(frames, ComputeMode.AUTO) { error("No progress expected") }
            assertEquals("quality-memory-low", result.error)
            assertEquals("", result.text)
            assertEquals("", result.raw)
            assertEquals(520L, result.durationMs)
            assertEquals(14, result.frames)
            assertTrue(result.backend.contains("not run"))
            assertTrue(result.backend.contains(QUALITY_MEMORY_REQUIREMENT))
        }
    }
}
