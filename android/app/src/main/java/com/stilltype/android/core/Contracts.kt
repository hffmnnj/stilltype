package com.stilltype.android.core

import android.graphics.Bitmap
import android.net.Uri
import java.io.File
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable

const val MAX_SESSION_SECONDS = 120
const val MAX_SESSION_FRAMES = MAX_SESSION_SECONDS * 25 + 1
const val MODEL_CHUNK_SECONDS = 8

@Serializable
enum class AppTheme { SYSTEM, LIGHT, DARK }

@Serializable
enum class ComputeMode { AUTO, CPU }

@Serializable
data class AppSettings(
    val theme: AppTheme = AppTheme.SYSTEM,
    val mirrored: Boolean = true,
    val autoCopy: Boolean = true,
    val keepHistory: Boolean = true,
    val retentionDays: Int = 30,
    val maxClipSeconds: Int = MAX_SESSION_SECONDS,
    val computeMode: ComputeMode = ComputeMode.AUTO,
)

@Serializable
data class HistoryEntry(
    val id: String,
    val text: String,
    val raw: String,
    val createdAt: Long,
    val durationMs: Long,
    val inferenceMs: Long,
    val favorite: Boolean = false,
)

@Serializable
data class VocabularyWord(
    val id: String,
    val word: String,
    val replacement: String = "",
)

@Serializable
data class AppData(
    val version: Int = 1,
    val settings: AppSettings = AppSettings(),
    val history: List<HistoryEntry> = emptyList(),
    val vocabulary: List<VocabularyWord> = emptyList(),
)

enum class ExportFormat { TXT, JSON, CSV }

interface AppRepository {
    val state: StateFlow<AppData>
    val loading: StateFlow<Boolean>
    val error: StateFlow<String?>
    suspend fun updateSettings(settings: AppSettings)
    suspend fun saveEntry(entry: HistoryEntry)
    suspend fun deleteEntry(id: String)
    suspend fun clearHistory()
    suspend fun saveWord(word: VocabularyWord)
    suspend fun deleteWord(id: String)
    suspend fun export(format: ExportFormat): Uri
}

@Serializable
data class ModelAsset(val name: String, val sizeBytes: Long, val sha256: String)

@Serializable
data class MobileModelManifest(
    val schemaVersion: Int = 1,
    val modelId: String = "usr2-baseplus-2026-ctc-int8",
    val releaseYear: Int = 2026,
    val exportKind: String = "ctc-int8",
    val sourceCheckpointSha256: String,
    val inputName: String = "video",
    val outputName: String = "logits",
    val fps: Int = 25,
    val cropSize: Int = 88,
    val maxFrames: Int = 200,
    val blankId: Int = 0,
    val eosId: Int = 1048,
    val files: List<ModelAsset>,
)

data class ModelStatus(
    val bundled: Boolean = false,
    val installed: Boolean = false,
    val preparing: Boolean = false,
    val progress: Float = 0f,
    val totalBytes: Long = 0,
    val modelId: String = "usr2-baseplus-2026-ctc-int8",
    val message: String = "Checking model",
    val error: String? = null,
)

data class InstalledModels(val directory: File, val manifest: MobileModelManifest)

interface ModelRepository {
    val status: StateFlow<ModelStatus>
    suspend fun prepareBundled()
    fun installedModels(): InstalledModels?
}

data class MouthFrame(val pixels: ByteArray, val timestampMs: Long, val quality: QualityCrops? = null)

data class FrameAnalysis(
    val mouth: MouthFrame? = null,
    val faceDetected: Boolean = false,
    val brightness: Float = 0f,
    val motion: Float = 0f,
    val hint: String = "Face the camera",
    val alignment: AlignmentFrame? = null,
)

data class RecognitionResult(
    val text: String,
    val raw: String,
    val durationMs: Long,
    val inferenceMs: Long,
    val frames: Int,
    val backend: String,
    val error: String? = null,
)

interface VisualRecognizer : AutoCloseable {
    /** Called serially off the UI thread. The caller owns and recycles the bitmap. */
    fun analyze(bitmap: Bitmap, timestampMs: Long): FrameAnalysis
    suspend fun recognize(frames: List<MouthFrame>, compute: ComputeMode): RecognitionResult
}

const val ACTION_CAPTURE = "com.stilltype.android.CAPTURE"
