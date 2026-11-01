package com.stilltype.android.core

import java.io.File
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.Serializable

const val QUALITY_MODEL_ID = "flare-base-transformer-2026-research"
// Android reports usable RAM after reservations; this selects 8 GB-class devices.
const val MIN_QUALITY_MEMORY_BYTES = 6L * 1024 * 1024 * 1024
const val QUALITY_MEMORY_REQUIREMENT = "Quality needs a device with at least 8 GB of RAM. Use Fast on this device."

@Serializable
data class QualityModelManifest(
    val schemaVersion: Int,
    val modelId: String,
    val releaseYear: Int,
    val license: String,
    val sourceCheckpointRepository: String,
    val sourceCheckpointRevision: String,
    val sourceCheckpointArtifact: String,
    val sourceCheckpointSha256: String,
    val decoderMode: String,
    val beamSize: Int,
    val fps: Int,
    val inputSize: Int,
    val cropSize: Int,
    val mean: Double,
    val std: Double,
    val maxFrames: Int,
    val maxPrefix: Int,
    val featureSize: Int,
    val vocabularySize: Int,
    val bosId: Int,
    val padId: Int,
    val eosId: Int,
    val unkId: Int,
    val cacheLayers: Int,
    val cacheHeads: Int,
    val cacheHeadDim: Int,
    val symbols: List<String>,
    val files: List<ModelAsset>,
    val encoderBytes: Long,
    val encoderSha256: String,
    val crossBytes: Long,
    val crossSha256: String,
    val beamDecoderBytes: Long,
    val beamDecoderSha256: String,
)

@Serializable
data class QualityCacheManifest(
    val schemaVersion: Int,
    val modelId: String,
    val beamSize: Int,
    val maximumNativeRows: Int,
    val cacheLayers: Int,
    val cacheHeads: Int,
    val cacheHeadDim: Int,
    val cacheChunks: Int,
    val cacheGatherBytes: Long,
    val cacheGatherSha256: String,
    val cacheMergeBytes: Long,
    val cacheMergeSha256: String,
    val files: List<ModelAsset>,
)

data class InstalledQualityModels(
    val directory: File,
    val manifest: QualityModelManifest,
    val cache: QualityCacheManifest,
)

interface QualityModelRepository {
    val status: StateFlow<ModelStatus>
    suspend fun prepareBundled()
    fun installedModels(): InstalledQualityModels?
}

data class RecognitionProgress(val completedWindows: Int, val totalWindows: Int)
