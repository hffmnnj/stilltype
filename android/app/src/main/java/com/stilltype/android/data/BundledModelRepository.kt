package com.stilltype.android.data

import android.content.Context
import com.stilltype.android.core.InstalledModels
import com.stilltype.android.core.MobileModelManifest
import com.stilltype.android.core.ModelAsset
import com.stilltype.android.core.ModelRepository
import com.stilltype.android.core.ModelStatus
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json

internal fun isLowerHexSha256(value: String): Boolean =
  value.length == 64 && value.all { it in '0'..'9' || it in 'a'..'f' }

internal fun InputStream.readBounded(maxBytes: Long): ByteArray {
  val out = ByteArrayOutputStream()
  val buffer = ByteArray(8 * 1024)
  var total = 0L
  while (true) {
    val read = read(buffer)
    if (read < 0) break
    total += read
    if (total > maxBytes) throw IOException("File exceeds the $maxBytes-byte limit.")
    out.write(buffer, 0, read)
  }
  return out.toByteArray()
}

private fun ByteArray.toHex(): String = buildString(size * 2) {
  for (b in this@toHex) append("%02x".format(b.toInt() and 0xff))
}

/**
 * Validates the APK-bundled manifest against the settled mobile contract and
 * returns the four fixed assets in copy order. Only the fixed file names are
 * accepted, so asset names can never carry paths or traversal sequences.
 */
internal fun validateManifest(manifest: MobileModelManifest): List<ModelAsset> {
  if (manifest.schemaVersion != 1) throw IllegalArgumentException("Unsupported model version")
  if (manifest.modelId != BundledModelRepository.EXPECTED_MODEL_ID) {
    throw IllegalArgumentException("Unexpected modelId")
  }
  if (manifest.releaseYear != 2026) throw IllegalArgumentException("Unexpected releaseYear")
  if (manifest.exportKind != "ctc-int8") throw IllegalArgumentException("Unexpected exportKind")
  if (manifest.sourceCheckpointSha256 != BundledModelRepository.EXPECTED_SOURCE_SHA) {
    throw IllegalArgumentException("Unexpected checkpoint")
  }
  if (manifest.fps != 25 || manifest.cropSize != 88 || manifest.maxFrames != 200) {
    throw IllegalArgumentException("Unexpected frame geometry")
  }
  if (manifest.blankId != 0 || manifest.eosId != 1048) throw IllegalArgumentException("Unexpected token IDs")
  val names = manifest.files.map { it.name }
  if (names.size != BundledModelRepository.REQUIRED_FILES.size ||
    names.toSet() != BundledModelRepository.REQUIRED_FILES.toSet()
  ) {
    throw IllegalArgumentException("Unexpected model files")
  }
  for (asset in manifest.files) {
    val bounds = BundledModelRepository.SIZE_BOUNDS[asset.name]
      ?: throw IllegalArgumentException("Unexpected model file: ${asset.name}")
    if (asset.sizeBytes !in bounds) {
      throw IllegalArgumentException("Model file ${asset.name} has an unexpected size")
    }
    if (!isLowerHexSha256(asset.sha256)) {
      throw IllegalArgumentException("Model file ${asset.name} has an invalid SHA-256")
    }
  }
  val byName = manifest.files.associateBy { it.name }
  return BundledModelRepository.REQUIRED_FILES.map { byName.getValue(it) }
}

/**
 * Copies the signed APK assets under assets/recognition/ into app-private
 * models/recognition/. The APK manifest is the trusted artifact descriptor:
 * every file is size- and hash-verified while copying into a temp file that
 * is renamed only after verification, and the installed.json marker is
 * written last. Existing valid files survive failed copies. prepareBundled is
 * explicit (called after UI consent), never opens the camera or installs on
 * launch, and never claims readiness when assets are missing. All hashing and
 * copying runs on Dispatchers.IO under one installer mutex; retry is safe.
 */
class BundledModelRepository(context: Context) : ModelRepository {
  companion object {
    internal const val ASSET_DIR = "recognition"
    internal const val MANIFEST_NAME = "manifest.json"
    internal const val MARKER_NAME = "installed.json"
    internal const val EXPECTED_MODEL_ID = "usr2-baseplus-2026-ctc-int8"
    internal const val EXPECTED_SOURCE_SHA =
      "4ab1f3f230aa6a724bcc698db8a17f2f54919136d2885c1a07a6b69247012773"
    internal val REQUIRED_FILES =
      listOf("usr2-ctc-int8.onnx", "tokens.json", "yunet-2026may.onnx", "mean-face.json")

    /** Plausible on-disk bounds guarding against truncated or runaway files. */
    internal val SIZE_BOUNDS = mapOf(
      "usr2-ctc-int8.onnx" to 1_000_000L..2_000_000_000L,
      "tokens.json" to 100L..1_000_000L,
      "yunet-2026may.onnx" to 10_000L..50_000_000L,
      "mean-face.json" to 10L..1_000_000L,
    )
    internal const val MAX_MANIFEST_BYTES = 1024 * 1024L
  }

  private val appContext = context.applicationContext
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val installerMutex = Mutex()
  private val manifestJson = Json { ignoreUnknownKeys = true }

  private val _status = MutableStateFlow(ModelStatus())
  override val status: StateFlow<ModelStatus> = _status

  init {
    scope.launch { refreshInitialStatus() }
  }

  internal fun modelsDir(): File = File(appContext.filesDir, "models/recognition")

  override fun installedModels(): InstalledModels? {
    val dir = modelsDir()
    val marker = File(dir, MARKER_NAME)
    if (!marker.isFile || marker.length() > MAX_MANIFEST_BYTES) return null
    val manifest = try {
      manifestJson.decodeFromString(MobileModelManifest.serializer(), marker.readText(Charsets.UTF_8))
    } catch (_: Exception) {
      return null
    }
    try {
      validateManifest(manifest)
    } catch (_: IllegalArgumentException) {
      return null
    }
    for (asset in manifest.files) {
      // Names are restricted to the fixed allowlist by validateManifest.
      val file = File(dir, asset.name)
      if (!file.isFile || file.length() != asset.sizeBytes) return null
    }
    return InstalledModels(dir, manifest)
  }

  override suspend fun prepareBundled() {
    installerMutex.withLock {
      withContext(Dispatchers.IO) {
        _status.value = _status.value.copy(
          preparing = true,
          progress = 0f,
          message = "Preparing model",
          error = null,
        )
        try {
          val manifest = readManifestAsset()
          val assets = validateManifest(manifest)
          val total = assets.sumOf { it.sizeBytes }
          _status.value = _status.value.copy(totalBytes = total, message = "Copying model files")
          modelsDir().mkdirs()
          var done = 0L
          for (asset in assets) {
            val dest = File(modelsDir(), asset.name)
            if (!isValidFile(dest, asset)) {
              copyAssetVerified(asset) { bytes ->
                _status.value = _status.value.copy(
                  progress = ((done + bytes).toFloat() / total.toFloat()).coerceIn(0f, 1f),
                )
              }
            }
            done += asset.sizeBytes
            _status.value = _status.value.copy(
              progress = (done.toFloat() / total.toFloat()).coerceIn(0f, 1f),
            )
          }
          writeMarker(manifest)
          _status.value = ModelStatus(
            bundled = true,
            installed = true,
            preparing = false,
            progress = 1f,
            totalBytes = total,
            message = "Model ready",
            error = null,
          )
        } catch (e: CancellationException) {
          _status.value = _status.value.copy(preparing = false, message = "Model preparation cancelled")
          throw e
        } catch (e: Exception) {
          val stillInstalled = try {
            installedModels() != null
          } catch (_: Exception) {
            false
          }
          _status.value = _status.value.copy(
            preparing = false,
            installed = stillInstalled,
            progress = if (stillInstalled) 1f else 0f,
            message = if (stillInstalled) "Model ready" else "Model preparation failed",
            error = e.message ?: "Model preparation failed",
          )
          // The failure is recorded on status for the UI; no rethrow so an
          // unobserved call cannot crash the app. Calling again retries safely.
        }
      }
    }
  }

  private suspend fun refreshInitialStatus() {
    val bundled = try {
      appContext.assets.list(ASSET_DIR)?.contains(MANIFEST_NAME) == true
    } catch (_: Exception) {
      false
    }
    var total = 0L
    var manifestOk = false
    if (bundled) {
      try {
        total = validateManifest(readManifestAsset()).sumOf { it.sizeBytes }
        manifestOk = true
      } catch (_: Exception) {
        manifestOk = false
      }
    }
    val installed = try {
      installedModels() != null
    } catch (_: Exception) {
      false
    }
    _status.value = ModelStatus(
      bundled = bundled,
      installed = installed,
      preparing = false,
      progress = if (installed) 1f else 0f,
      totalBytes = total,
      message = when {
        installed -> "Model ready"
        bundled && manifestOk -> "Model bundled — ready to prepare"
        bundled -> "Bundled model manifest is unreadable"
        else -> "Recognition model is not bundled in this build"
      },
      error = if (bundled && !manifestOk) "Bundled model manifest is unreadable" else null,
    )
  }

  private fun readManifestAsset(): MobileModelManifest {
    val bytes = try {
      appContext.assets.open("$ASSET_DIR/$MANIFEST_NAME").use { it.readBounded(MAX_MANIFEST_BYTES) }
    } catch (e: IOException) {
      throw IOException("Recognition model is not bundled in this build.", e)
    }
    try {
      return manifestJson.decodeFromString(MobileModelManifest.serializer(), bytes.decodeToString())
    } catch (e: IllegalArgumentException) {
      throw IOException("Bundled model manifest is unreadable: ${e.message}", e)
    }
  }

  private fun copyAssetVerified(asset: ModelAsset, onBytes: (Long) -> Unit) {
    val dir = modelsDir()
    val dest = File(dir, asset.name)
    val tmp = File(dir, asset.name + ".part")
    tmp.delete()
    val digest = MessageDigest.getInstance("SHA-256")
    var total = 0L
    try {
      appContext.assets.open("$ASSET_DIR/${asset.name}").use { input ->
        FileOutputStream(tmp).use { output ->
          val buffer = ByteArray(64 * 1024)
          while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            total += read
            if (total > asset.sizeBytes) {
              throw IOException("Model file ${asset.name} exceeds its expected size.")
            }
            digest.update(buffer, 0, read)
            output.write(buffer, 0, read)
            onBytes(total)
          }
          output.fd.sync()
        }
      }
    } catch (e: IOException) {
      tmp.delete()
      throw e
    }
    if (total != asset.sizeBytes) {
      tmp.delete()
      throw IOException("Model file ${asset.name} size does not match its manifest.")
    }
    if (digest.digest().toHex() != asset.sha256) {
      tmp.delete()
      throw IOException("Model file ${asset.name} SHA-256 does not match its manifest.")
    }
    if (dest.isFile && !dest.delete()) {
      tmp.delete()
      throw IOException("Could not replace model file ${asset.name}.")
    }
    if (!tmp.renameTo(dest)) {
      tmp.delete()
      throw IOException("Could not install model file ${asset.name}.")
    }
  }

  private fun writeMarker(manifest: MobileModelManifest) {
    val dir = modelsDir()
    val marker = File(dir, MARKER_NAME)
    val tmp = File(dir, "$MARKER_NAME.part")
    tmp.delete()
    try {
      FileOutputStream(tmp).use { output ->
        output.write(Json.encodeToString(MobileModelManifest.serializer(), manifest).toByteArray(Charsets.UTF_8))
        output.fd.sync()
      }
    } catch (e: IOException) {
      tmp.delete()
      throw e
    }
    if (marker.isFile && !marker.delete()) {
      tmp.delete()
      throw IOException("Could not finalize the model installation.")
    }
    if (!tmp.renameTo(marker)) {
      tmp.delete()
      throw IOException("Could not finalize the model installation.")
    }
  }

  private fun sha256Of(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    FileInputStream(file).use { input ->
      val buffer = ByteArray(64 * 1024)
      while (true) {
        val read = input.read(buffer)
        if (read < 0) break
        digest.update(buffer, 0, read)
      }
    }
    return digest.digest().toHex()
  }

  private fun isValidFile(file: File, asset: ModelAsset): Boolean {
    if (!file.isFile || file.length() != asset.sizeBytes) return false
    return try {
      sha256Of(file) == asset.sha256
    } catch (_: IOException) {
      false
    }
  }
}
