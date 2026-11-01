package com.stilltype.android.data

import com.stilltype.android.core.ModelAsset
import com.stilltype.android.core.QUALITY_MODEL_ID
import com.stilltype.android.core.QualityCacheManifest
import com.stilltype.android.core.QualityModelManifest

internal const val QUALITY_SOURCE_SHA =
  "fe8c8ecab633bd4e3e490924e2bc917420b4d18d7309a5cd612902ccc842853b"

private val qualityBounds = mapOf(
  "encoder.onnx" to 1_000_000L..1_000_000_000L,
  "cross.onnx" to 100_000L..500_000_000L,
  "decoder-beam.onnx" to 1_000_000L..1_000_000_000L,
  "cache-gather.onnx" to 100L..1_000_000L,
  "cache-merge.onnx" to 100L..1_000_000L,
)

private fun validateQualityFiles(files: List<ModelAsset>, expected: List<ModelAsset>): List<ModelAsset> {
  require(files.size == expected.size && files.map { it.name }.toSet() == expected.map { it.name }.toSet()) {
    "Unexpected or duplicate Quality model files"
  }
  val byName = files.associateBy { it.name }
  for (flat in expected) {
    val asset = byName.getValue(flat.name)
    require(asset == flat) { "Conflicting descriptor for ${asset.name}" }
    require(asset.sizeBytes in qualityBounds.getValue(asset.name)) { "Invalid size for ${asset.name}" }
    require(isLowerHexSha256(asset.sha256)) { "Invalid SHA-256 for ${asset.name}" }
  }
  return expected.map { byName.getValue(it.name) }
}

// Graph descriptors are trusted from the signed APK; matching them verifies copies,
// not the provenance of the exported graphs independently of the APK builder.
internal fun validateQualityManifests(model: QualityModelManifest, cache: QualityCacheManifest): List<ModelAsset> {
  require(model.schemaVersion == 1 && model.modelId == QUALITY_MODEL_ID && model.releaseYear == 2026 &&
    model.license == "CC-BY-NC-4.0" && model.sourceCheckpointRepository == "Anony94/FLARE-checkpoints" &&
    model.sourceCheckpointRevision == "6187ad650dc1786a8efc8b6bfbee37bbf62c4b2e" &&
    model.sourceCheckpointArtifact == "exp_22_base/finetuned/checkpoints/checkpoint_last.pt" &&
    model.sourceCheckpointSha256 == QUALITY_SOURCE_SHA) { "Unexpected Quality model provenance" }
  require(model.decoderMode == "beam" && model.beamSize == 50 && model.fps == 25 &&
    model.inputSize == 96 && model.cropSize == 88 && model.mean == 0.421 && model.std == 0.165 &&
    model.maxFrames == 200 && model.maxPrefix == 128 && model.featureSize == 512 &&
    model.vocabularySize == 1000 && model.symbols.size == 1000 &&
    model.bosId == 0 && model.padId == 1 && model.eosId == 2 && model.unkId == 3 &&
    model.cacheLayers == 6 && model.cacheHeads == 4 && model.cacheHeadDim == 192) {
    "Unexpected Quality model geometry or decoder configuration"
  }
  require(cache.schemaVersion == 1 && cache.modelId == model.modelId && cache.beamSize == model.beamSize &&
    cache.maximumNativeRows == 5 && cache.cacheLayers == model.cacheLayers &&
    cache.cacheHeads == model.cacheHeads && cache.cacheHeadDim == model.cacheHeadDim && cache.cacheChunks == 10) {
    "Unexpected Quality cache configuration"
  }
  return validateQualityFiles(model.files, listOf(
    ModelAsset("encoder.onnx", model.encoderBytes, model.encoderSha256),
    ModelAsset("cross.onnx", model.crossBytes, model.crossSha256),
    ModelAsset("decoder-beam.onnx", model.beamDecoderBytes, model.beamDecoderSha256),
  )) + validateQualityFiles(cache.files, listOf(
    ModelAsset("cache-gather.onnx", cache.cacheGatherBytes, cache.cacheGatherSha256),
    ModelAsset("cache-merge.onnx", cache.cacheMergeBytes, cache.cacheMergeSha256),
  ))
}
