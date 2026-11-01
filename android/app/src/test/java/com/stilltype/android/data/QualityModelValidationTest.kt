package com.stilltype.android.data

import com.stilltype.android.core.ModelAsset
import com.stilltype.android.core.QUALITY_MODEL_ID
import com.stilltype.android.core.QualityCacheManifest
import com.stilltype.android.core.QualityModelManifest
import java.io.IOException
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class QualityModelValidationTest {
  @get:Rule val folder = TemporaryFolder()
  private val sha = "0123456789abcdef".repeat(4)
  private val encoder = ModelAsset("encoder.onnx", 219_664_435, sha)
  private val cross = ModelAsset("cross.onnx", 9_500_000, sha)
  private val decoder = ModelAsset("decoder-beam.onnx", 108_000_000, sha)
  private val gather = ModelAsset("cache-gather.onnx", 1849, sha)
  private val merge = ModelAsset("cache-merge.onnx", 12839, sha)

  private fun model() = QualityModelManifest(
    schemaVersion = 1, modelId = QUALITY_MODEL_ID, releaseYear = 2026, license = "CC-BY-NC-4.0",
    sourceCheckpointRepository = "Anony94/FLARE-checkpoints",
    sourceCheckpointRevision = "6187ad650dc1786a8efc8b6bfbee37bbf62c4b2e",
    sourceCheckpointArtifact = "exp_22_base/finetuned/checkpoints/checkpoint_last.pt",
    sourceCheckpointSha256 = QUALITY_SOURCE_SHA, decoderMode = "beam", beamSize = 50, fps = 25,
    inputSize = 96, cropSize = 88, mean = 0.421, std = 0.165, maxFrames = 200, maxPrefix = 128,
    featureSize = 512, vocabularySize = 1000, bosId = 0, padId = 1, eosId = 2, unkId = 3,
    cacheLayers = 6, cacheHeads = 4, cacheHeadDim = 192, symbols = List(1000) { "token-$it" },
    files = listOf(encoder, cross, decoder), encoderBytes = encoder.sizeBytes, encoderSha256 = encoder.sha256,
    crossBytes = cross.sizeBytes, crossSha256 = cross.sha256,
    beamDecoderBytes = decoder.sizeBytes, beamDecoderSha256 = decoder.sha256,
  )

  private fun cache() = QualityCacheManifest(
    schemaVersion = 1, modelId = QUALITY_MODEL_ID, beamSize = 50, maximumNativeRows = 5,
    cacheLayers = 6, cacheHeads = 4, cacheHeadDim = 192, cacheChunks = 10,
    cacheGatherBytes = gather.sizeBytes, cacheGatherSha256 = gather.sha256,
    cacheMergeBytes = merge.sizeBytes, cacheMergeSha256 = merge.sha256, files = listOf(gather, merge),
  )

  @Test fun supportedMetadata_acceptsUnknownFieldsAndReturnsOnlyFixedFiles() {
    val json = Json { ignoreUnknownKeys = true }
    val encoded = Json.encodeToString(QualityModelManifest.serializer(), model())
    val parsed = json.decodeFromString(QualityModelManifest.serializer(), encoded.dropLast(1) + ",\"futureField\":42}")
    assertEquals(listOf(encoder, cross, decoder, gather, merge), validateQualityManifests(parsed, cache()))
  }

  @Test fun wrongModelAndGeometry_areRejected() {
    val invalid = listOf(
      model().copy(modelId = "different"), model().copy(sourceCheckpointSha256 = "0".repeat(64)),
      model().copy(sourceCheckpointRevision = "different"), model().copy(license = "MIT"),
      model().copy(cropSize = 96), model().copy(mean = 0.42), model().copy(beamSize = 5),
      model().copy(symbols = List(999) { "x" }), model().copy(decoderMode = "greedy"),
    )
    invalid.forEach { candidate -> assertThrows(IllegalArgumentException::class.java) { validateQualityManifests(candidate, cache()) } }
    assertThrows(IllegalArgumentException::class.java) { validateQualityManifests(model(), cache().copy(cacheChunks = 9)) }
    assertThrows(IllegalArgumentException::class.java) { validateQualityManifests(model(), cache().copy(maximumNativeRows = 50)) }
  }

  @Test fun fileDescriptors_rejectPathsDuplicatesMismatchedOrInvalidHashesAndSizes() {
    val invalid = listOf(
      model().copy(files = listOf(encoder.copy(name = "../encoder.onnx"), cross, decoder)),
      model().copy(files = listOf(encoder, encoder, decoder)),
      model().copy(encoderSha256 = "0".repeat(64)),
      model().copy(files = listOf(encoder.copy(sha256 = sha.uppercase()), cross, decoder), encoderSha256 = sha.uppercase()),
      model().copy(files = listOf(encoder.copy(sizeBytes = 0), cross, decoder), encoderBytes = 0),
    )
    invalid.forEach { candidate -> assertThrows(IllegalArgumentException::class.java) { validateQualityManifests(candidate, cache()) } }
    assertThrows(IllegalArgumentException::class.java) {
      validateQualityManifests(model(), cache().copy(cacheMergeSha256 = "0".repeat(64)))
    }
  }

  @Test fun copy_corruptBytesDoNotReplaceExistingFile() {
    val good = "correct bytes".toByteArray()
    val hash = MessageDigest.getInstance("SHA-256").digest(good).joinToString("") { "%02x".format(it.toInt() and 0xff) }
    val asset = ModelAsset("encoder.onnx", good.size.toLong(), hash)
    val dest = folder.newFile("encoder.onnx")
    dest.writeText("existing")
    assertThrows(IOException::class.java) { copyQualityVerified("corrupt bytes".byteInputStream(), dest, asset) }
    assertEquals("existing", dest.readText())
    assertFalse(folder.root.resolve("encoder.onnx.part").exists())
    copyQualityVerified(good.inputStream(), dest, asset)
    assertEquals("correct bytes", dest.readText())
  }
}
