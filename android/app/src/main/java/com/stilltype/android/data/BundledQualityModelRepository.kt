package com.stilltype.android.data

import android.app.ActivityManager
import android.content.Context
import com.stilltype.android.core.InstalledQualityModels
import com.stilltype.android.core.ModelAsset
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.MIN_QUALITY_MEMORY_BYTES
import com.stilltype.android.core.QUALITY_MODEL_ID
import com.stilltype.android.core.QUALITY_MEMORY_REQUIREMENT
import com.stilltype.android.core.QualityCacheManifest
import com.stilltype.android.core.QualityModelManifest
import com.stilltype.android.core.QualityModelRepository
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
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
private data class QualityMarker(val model: QualityModelManifest, val cache: QualityCacheManifest)

private fun ByteArray.hexDigest(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }

internal data class QualityMemoryCheck(val unavailableReason: String?, val lowMemory: Boolean = false)

private const val MEMORY_INFO_UNAVAILABLE =
  "Quality is unavailable because device memory information could not be obtained. Use Fast on this device."

internal fun checkQualityMemory(context: Context): QualityMemoryCheck {
  val info = try {
    val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
      ?: return QualityMemoryCheck(MEMORY_INFO_UNAVAILABLE)
    ActivityManager.MemoryInfo().also { manager.getMemoryInfo(it) }
  } catch (_: Exception) {
    return QualityMemoryCheck(MEMORY_INFO_UNAVAILABLE)
  }
  return when {
    info.totalMem <= 0 -> QualityMemoryCheck(MEMORY_INFO_UNAVAILABLE)
    info.totalMem < MIN_QUALITY_MEMORY_BYTES -> QualityMemoryCheck(QUALITY_MEMORY_REQUIREMENT, info.lowMemory)
    else -> QualityMemoryCheck(null, info.lowMemory)
  }
}

internal fun copyQualityVerified(input: InputStream, dest: File, asset: ModelAsset, onBytes: (Long) -> Unit = {}) {
  val part = File(dest.parentFile, dest.name + ".part")
  val digest = MessageDigest.getInstance("SHA-256")
  var total = 0L
  try {
    FileOutputStream(part).use { output ->
      val buffer = ByteArray(64 * 1024)
      while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        total += count
        if (total > asset.sizeBytes) throw IOException("${asset.name} exceeds its expected size")
        digest.update(buffer, 0, count)
        output.write(buffer, 0, count)
        onBytes(total)
      }
      output.fd.sync()
    }
    if (total != asset.sizeBytes || digest.digest().hexDigest() != asset.sha256) {
      throw IOException("${asset.name} failed size or SHA-256 verification")
    }
    if (!part.renameTo(dest)) throw IOException("Could not install ${asset.name}")
  } finally {
    part.delete()
  }
}

/** Explicitly prepares the optional APK-bundled pack; construction never copies models. */
class BundledQualityModelRepository(context: Context) : QualityModelRepository {
  private val appContext = context.applicationContext
  private val mutex = Mutex()
  private val json = Json { ignoreUnknownKeys = true }
  private val initialMemoryReason = checkQualityMemory(appContext).unavailableReason
  private val _status = MutableStateFlow(ModelStatus(bundled = hasBundledPack(), modelId = QUALITY_MODEL_ID,
    message = initialMemoryReason ?: "Checking Quality model", unavailableReason = initialMemoryReason))
  override val status: StateFlow<ModelStatus> = _status
  private val dir: File get() = File(appContext.filesDir, "models/quality")
  private val marker: File get() = File(dir, "installed.json")

  companion object {
    private const val MAX_METADATA_BYTES = 1024 * 1024L
    private const val ASSET_DIR = "quality"
    private val requiredAssets = setOf("metadata.json", "cache.json", "encoder.onnx", "cross.onnx",
      "decoder-beam.onnx", "cache-gather.onnx", "cache-merge.onnx")
  }

  init {
    CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
      mutex.withLock { refreshInitialStatus() }
    }
  }

  override fun installedModels(): InstalledQualityModels? {
    if (checkQualityMemory(appContext).unavailableReason != null) return null
    return try {
      val saved = marker.inputStream().use { json.decodeFromString(QualityMarker.serializer(), it.readBounded(2 * MAX_METADATA_BYTES).decodeToString()) }
      val files = validateQualityManifests(saved.model, saved.cache)
      if (files.any { !File(dir, it.name).isFile || File(dir, it.name).length() != it.sizeBytes }) return null
      InstalledQualityModels(dir, saved.model, saved.cache)
    } catch (_: Exception) {
      null
    }
  }

  override suspend fun prepareBundled() {
    mutex.withLock {
      withContext(Dispatchers.IO) {
        val memoryReason = checkQualityMemory(appContext).unavailableReason
        if (memoryReason != null) {
          _status.value = _status.value.copy(bundled = hasBundledPack(), installed = false, preparing = false,
            progress = 0f, message = memoryReason, error = memoryReason, unavailableReason = memoryReason)
          return@withContext
        }
        _status.value = _status.value.copy(preparing = true, progress = 0f, message = "Preparing Quality model",
          error = null, unavailableReason = null)
        try {
          if (!hasBundledPack()) throw IOException("Quality model pack is incomplete or not bundled")
          val pack = readPack()
          val files = validateQualityManifests(pack.model, pack.cache)
          val total = files.sumOf { it.sizeBytes }
          _status.value = _status.value.copy(bundled = true, totalBytes = total, message = "Copying Quality model files")
          if (!dir.isDirectory && !dir.mkdirs()) throw IOException("Could not create Quality model directory")
          // A prior marker cannot describe a partially replaced version of the pack.
          val old = installedModels()
          if (marker.exists() && (old == null || old.manifest != pack.model || old.cache != pack.cache) && !marker.delete()) {
            throw IOException("Could not replace Quality installation marker")
          }
          var done = 0L
          for (asset in files) {
            val dest = File(dir, asset.name)
            if (!validFile(dest, asset)) {
              if (marker.exists() && !marker.delete()) throw IOException("Could not invalidate Quality installation marker")
              appContext.assets.open("$ASSET_DIR/${asset.name}").use { input ->
                copyQualityVerified(input, dest, asset) { bytes ->
                  _status.value = _status.value.copy(progress = ((done + bytes).toFloat() / total).coerceIn(0f, 1f))
                }
              }
            }
            done += asset.sizeBytes
            _status.value = _status.value.copy(progress = (done.toFloat() / total).coerceIn(0f, 1f))
          }
          writeMarker(pack)
          _status.value = ModelStatus(bundled = true, installed = true, progress = 1f, totalBytes = total,
            modelId = QUALITY_MODEL_ID, message = "Quality model ready")
        } catch (e: CancellationException) {
          val installed = installedModels() != null
          _status.value = _status.value.copy(preparing = false, installed = installed,
            progress = if (installed) 1f else 0f, message = "Quality model preparation cancelled")
          throw e
        } catch (e: Exception) {
          val installed = installedModels() != null
          _status.value = _status.value.copy(preparing = false, installed = installed,
            progress = if (installed) 1f else 0f,
            message = if (installed) "Quality model ready" else "Quality model preparation failed",
            error = e.message ?: "Quality model preparation failed")
        }
      }
    }
  }

  private fun hasBundledPack(): Boolean = try {
    appContext.assets.list(ASSET_DIR)?.toSet()?.containsAll(requiredAssets) == true
  } catch (_: IOException) { false }

  private fun refreshInitialStatus() {
    val bundled = hasBundledPack()
    val memoryReason = checkQualityMemory(appContext).unavailableReason
    var total = 0L
    var valid = false
    if (bundled) {
      try {
        val pack = readPack()
        total = validateQualityManifests(pack.model, pack.cache).sumOf { it.sizeBytes }
        valid = true
      } catch (_: Exception) { }
    }
    val installed = memoryReason == null && installedModels() != null
    _status.value = ModelStatus(bundled = bundled, installed = installed, progress = if (installed) 1f else 0f,
      totalBytes = total, modelId = QUALITY_MODEL_ID,
      message = when {
        memoryReason != null -> memoryReason
        installed -> "Quality model ready"
        valid -> "Quality model bundled — ready to prepare"
        bundled -> "Bundled Quality model metadata is unreadable"
        else -> "Quality model is not bundled in this build"
      }, error = memoryReason ?: if (bundled && !valid) "Bundled Quality model metadata is unreadable" else null,
      unavailableReason = memoryReason)
  }

  private fun readPack(): QualityMarker {
    val model = appContext.assets.open("$ASSET_DIR/metadata.json").use {
      json.decodeFromString(QualityModelManifest.serializer(), it.readBounded(MAX_METADATA_BYTES).decodeToString())
    }
    val cache = appContext.assets.open("$ASSET_DIR/cache.json").use {
      json.decodeFromString(QualityCacheManifest.serializer(), it.readBounded(MAX_METADATA_BYTES).decodeToString())
    }
    return QualityMarker(model, cache)
  }

  private fun validFile(file: File, asset: ModelAsset): Boolean {
    if (!file.isFile || file.length() != asset.sizeBytes) return false
    return try {
      val digest = MessageDigest.getInstance("SHA-256")
      FileInputStream(file).use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
          val count = input.read(buffer)
          if (count < 0) break
          digest.update(buffer, 0, count)
        }
      }
      digest.digest().hexDigest() == asset.sha256
    } catch (_: IOException) { false }
  }

  private fun writeMarker(pack: QualityMarker) {
    val part = File(dir, "installed.json.part")
    try {
      FileOutputStream(part).use {
        it.write(json.encodeToString(QualityMarker.serializer(), pack).toByteArray(Charsets.UTF_8))
        it.fd.sync()
      }
      if (!part.renameTo(marker)) throw IOException("Could not finalize Quality model installation")
    } finally {
      part.delete()
    }
  }
}
