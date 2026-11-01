package com.stilltype.android.ui.capture

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.stilltype.android.core.AppData
import com.stilltype.android.core.AppRepository
import com.stilltype.android.core.AppSettings
import com.stilltype.android.core.ComputeMode
import com.stilltype.android.core.ExportFormat
import com.stilltype.android.core.FrameAnalysis
import com.stilltype.android.core.HistoryEntry
import com.stilltype.android.core.InstalledModels
import com.stilltype.android.core.InstalledQualityModels
import com.stilltype.android.core.MobileModelManifest
import com.stilltype.android.core.ModelRepository
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.core.QualityModelRepository
import com.stilltype.android.core.RecognitionMode
import com.stilltype.android.core.RecognitionResult
import com.stilltype.android.core.VisualRecognizer
import com.stilltype.android.core.VocabularyWord
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Capture ownership with in-memory fakes: no camera, models or real history are touched.
 * Frames are injected through [FrameSink.onFrame] and recognition is a gated fake.
 */
@RunWith(AndroidJUnit4::class)
class CaptureViewModelOwnershipTest {
    private val application = ApplicationProvider.getApplicationContext<Application>()

    private class FakeRecognizer : VisualRecognizer {
        val recognizeCalls = AtomicInteger()
        val entered = CountDownLatch(1)
        val exited = CountDownLatch(1)
        val gate = CompletableDeferred<Unit>()

        // When true the gate wait ignores cancellation, modelling a native call that returns late.
        var ignoreCancellation = false

        override fun analyze(bitmap: Bitmap, timestampMs: Long) =
            FrameAnalysis(MouthFrame(ByteArray(1), timestampMs), faceDetected = true, brightness = 0.5f, hint = "")

        override suspend fun recognize(frames: List<MouthFrame>, compute: ComputeMode): RecognitionResult {
            recognizeCalls.incrementAndGet()
            entered.countDown()
            try {
                if (ignoreCancellation) withContext(NonCancellable) { gate.await() } else gate.await()
            } finally {
                exited.countDown()
            }
            return RecognitionResult("hello world", "hello world", 1_000, 50, frames.size, "fake")
        }

        override fun close() = Unit
    }

    private class FakeRepository(keepHistory: Boolean = true) : AppRepository {
        val data = MutableStateFlow(AppData(settings = AppSettings(autoCopy = false, keepHistory = keepHistory)))
        override val state: StateFlow<AppData> = data
        override val loading: StateFlow<Boolean> = MutableStateFlow(false)
        override val error: StateFlow<String?> = MutableStateFlow(null)

        val saveStarted = CountDownLatch(1)
        val saveGate = CompletableDeferred<Unit>()
        var failSave = false
        val saved = ArrayList<HistoryEntry>()

        /** Set by the test so the fake can record what the screen showed when the write began. */
        @Volatile var observe: () -> CaptureState? = { null }
        @Volatile var stateAtSaveStart: CaptureState? = null

        val saveDone = CountDownLatch(1)

        override suspend fun saveEntry(entry: HistoryEntry) {
            try {
                stateAtSaveStart = observe()
                saveStarted.countDown()
                saveGate.await()
                if (failSave) error("disk full")
                synchronized(saved) { saved += entry }
                data.update { it.copy(history = it.history + entry) }
            } finally {
                saveDone.countDown()
            }
        }

        override suspend fun updateSettings(settings: AppSettings) = data.update { it.copy(settings = settings) }
        override suspend fun deleteEntry(id: String) = Unit
        override suspend fun clearHistory() = Unit
        override suspend fun saveWord(word: VocabularyWord) = Unit
        override suspend fun deleteWord(id: String) = Unit
        override suspend fun export(format: ExportFormat): Uri = Uri.EMPTY
    }

    private object FakeModels : ModelRepository {
        override val status = MutableStateFlow(ModelStatus(bundled = true, installed = true))
        override suspend fun prepareBundled() = Unit
        override fun installedModels() = InstalledModels(File("."), MobileModelManifest(sourceCheckpointSha256 = "test", files = emptyList()))
    }

    private object FakeQualityModels : QualityModelRepository {
        override val status = MutableStateFlow(ModelStatus())
        override suspend fun prepareBundled() = Unit
        override fun installedModels(): InstalledQualityModels? = null
    }

    private class Rig(application: Application, val recognizer: FakeRecognizer, val repository: FakeRepository) {
        val vm = CaptureViewModel(
            application,
            object : CaptureDependencies {
                override val repository = this@Rig.repository
                override val models = FakeModels
                override val qualityModels = FakeQualityModels
                override val recognizer = this@Rig.recognizer
            },
        )

        init {
            repository.observe = { vm.state.value }
        }

        private val bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)

        /** Starts a Fast session and delivers one frame so the session is RECORDING. */
        fun record() {
            assertTrue(vm.start())
            vm.onFrame(vm.session, bitmap, 1_000)
            await { vm.state.value.phase == CapturePhase.RECORDING }
        }

        /** Frees the bitmap and cancels any session left open. */
        fun close() {
            vm.cancel()
            bitmap.recycle()
        }

        fun deliverFrame() = vm.onFrame(vm.session, bitmap, 1_000)
    }

    /** Runs everything already queued on the main looper, including resumed continuations. */
    private fun drainMain() {
        InstrumentationRegistry.getInstrumentation().runOnMainSync { }
        InstrumentationRegistry.getInstrumentation().waitForIdleSync()
    }

    private fun rig(recognizer: FakeRecognizer = FakeRecognizer(), repository: FakeRepository = FakeRepository()) =
        Rig(application, recognizer, repository)

    @Test
    fun cancelBeforeLateRecognitionReturnsProducesNoResultCopyOrSave() {
        val recognizer = FakeRecognizer().apply { ignoreCancellation = true }
        val rig = rig(recognizer)
        rig.record()
        rig.vm.stop()
        assertTrue(recognizer.entered.await(5, TimeUnit.SECONDS))
        assertEquals(CapturePhase.PROCESSING, rig.vm.state.value.phase)

        rig.vm.cancel()
        // Complete on the main thread so the resumed continuation is queued there, then drain it.
        InstrumentationRegistry.getInstrumentation().runOnMainSync { recognizer.gate.complete(Unit) }
        assertTrue(recognizer.exited.await(5, TimeUnit.SECONDS))
        drainMain()

        val state = rig.vm.state.value
        assertEquals(CapturePhase.IDLE, state.phase)
        assertNull(state.result)
        assertNull(state.failure)
        assertEquals(1L, rig.repository.saveStarted.count)
        assertTrue(synchronized(rig.repository.saved) { rig.repository.saved.isEmpty() })
        rig.close()
    }

    @Test
    fun cancelRightAfterStopNeverStartsRecognition() {
        val recognizer = FakeRecognizer()
        val rig = rig(recognizer)
        // Keep the main looper busy so the worker queued by stop() cannot run before cancel().
        val release = CountDownLatch(1)
        val blocked = CountDownLatch(1)
        Handler(Looper.getMainLooper()).post {
            blocked.countDown()
            release.await(5, TimeUnit.SECONDS)
        }
        assertTrue(blocked.await(5, TimeUnit.SECONDS))

        rig.record0()
        rig.vm.stop()
        rig.vm.cancel()
        release.countDown()
        drainMain()

        assertEquals(0, recognizer.recognizeCalls.get())
        assertEquals(CapturePhase.IDLE, rig.vm.state.value.phase)
        assertNull(rig.vm.state.value.result)
        rig.close()
    }

    @Test
    fun completionIsPublishedBeforeTheSuspendedHistoryWriteBegins() {
        val recognizer = FakeRecognizer().apply { gate.complete(Unit) }
        val repository = FakeRepository()
        val rig = rig(recognizer, repository)
        rig.record()
        rig.vm.stop()

        assertTrue(repository.saveStarted.await(5, TimeUnit.SECONDS))
        val atSave = repository.stateAtSaveStart
        assertNotNull(atSave)
        assertEquals(CapturePhase.IDLE, atSave!!.phase)
        assertEquals("hello world", atSave.result?.text)
        assertFalse(atSave.result!!.saved)

        // Cancel after completion cannot revoke the finished dictation.
        rig.vm.cancel()
        assertEquals("hello world", rig.vm.state.value.result?.text)

        // A framing preview changes the session token while the save is still suspended.
        rig.vm.setPreview(true)
        rig.vm.setPreview(false)
        assertEquals("hello world", rig.vm.state.value.result?.text)

        repository.saveGate.complete(Unit)
        await { rig.vm.state.value.result?.saved == true }
        val persisted = synchronized(repository.saved) { repository.saved.toList() }
        assertEquals(1, persisted.size)
        assertEquals(rig.vm.state.value.result?.id, persisted.single().id)
        assertEquals("hello world", persisted.single().text)
        rig.close()
    }

    @Test
    fun lateFailedSaveDoesNotRepaintANewerSession() {
        val recognizer = FakeRecognizer().apply { gate.complete(Unit) }
        val repository = FakeRepository().apply { failSave = true }
        val rig = rig(recognizer, repository)
        rig.record()
        rig.vm.stop()
        assertTrue(repository.saveStarted.await(5, TimeUnit.SECONDS))

        // A newer session starts (the old result is retained) and is cancelled before the old save fails.
        assertTrue(rig.vm.start())
        assertEquals(CapturePhase.STARTING, rig.vm.state.value.phase)
        rig.vm.cancel()
        val before = rig.vm.state.value
        assertEquals(CapturePhase.IDLE, before.phase)
        assertNull(before.failure)
        assertNotNull(before.result)

        InstrumentationRegistry.getInstrumentation().runOnMainSync { repository.saveGate.complete(Unit) }
        assertTrue(repository.saveDone.await(5, TimeUnit.SECONDS))
        drainMain()

        assertEquals(before, rig.vm.state.value)
        assertFalse(rig.vm.state.value.result!!.saved)
        assertTrue(synchronized(repository.saved) { repository.saved.isEmpty() })
        rig.close()
    }

    private fun Rig.record0() {
        // Same as record() but without waiting on the (blocked) main looper.
        assertTrue(vm.start())
        deliverFrame()
        assertEquals(CapturePhase.RECORDING, vm.state.value.phase)
    }
}

private fun await(timeoutMs: Long = 5_000, condition: () -> Boolean) {
    val deadline = System.nanoTime() + timeoutMs * 1_000_000
    while (!condition()) {
        if (System.nanoTime() > deadline) throw AssertionError("Condition not met within ${timeoutMs}ms")
        Thread.sleep(10)
    }
}
