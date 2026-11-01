package com.stilltype.android.data

import android.content.Context
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.datastore.core.CorruptionException
import androidx.datastore.core.DataStore
import androidx.datastore.core.DataStoreFactory
import androidx.datastore.core.Serializer
import com.stilltype.android.core.AppData
import com.stilltype.android.core.AppRepository
import com.stilltype.android.core.AppSettings
import com.stilltype.android.core.ExportFormat
import com.stilltype.android.core.HistoryEntry
import com.stilltype.android.core.VocabularyWord
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.Json

/**
 * FileProvider authority. The platform manifest must declare a provider with
 * this authority and an xml/file-paths entry for the "exports" path.
 */
internal const val EXPORT_AUTHORITY = "com.stilltype.android.files"

private const val MAX_TEXT_CHARS = 100_000
private const val MAX_DURATION_MS = 120_000L
private const val MAX_WORD_CHARS = 256
private const val MIN_RETENTION_DAYS = 7
private const val MAX_RETENTION_DAYS = 365
private const val MIN_CLIP_SECONDS = 3
private const val MAX_CLIP_SECONDS = 120

private val EXPORT_NAME = Regex("""stilltype-\d{4}-\d{2}-\d{2}\.(txt|json|csv)""")

/** Version 1 store. Unknown fields are ignored; malformed files surface an error. */
internal object AppDataSerializer : Serializer<AppData> {
  override val defaultValue: AppData = AppData()
  private val json = Json { ignoreUnknownKeys = true }

  override suspend fun readFrom(input: InputStream): AppData {
    val text = try {
      input.readBytes().decodeToString()
    } catch (e: IOException) {
      throw CorruptionException("Could not read saved data.", e)
    }
    if (text.isBlank()) throw CorruptionException("Saved data is empty.")
    try {
      return json.decodeFromString(AppData.serializer(), text)
    } catch (e: IllegalArgumentException) {
      // SerializationException extends IllegalArgumentException. Never silently
      // discard a file we cannot parse.
      throw CorruptionException(
        "Saved data is unreadable and was left untouched: ${e.message}",
        e,
      )
    }
  }

  override suspend fun writeTo(t: AppData, output: OutputStream) {
    output.write(json.encodeToString(AppData.serializer(), t).toByteArray(Charsets.UTF_8))
  }
}

internal fun validateSettings(settings: AppSettings) {
  if (settings.retentionDays !in MIN_RETENTION_DAYS..MAX_RETENTION_DAYS ||
    settings.maxClipSeconds !in MIN_CLIP_SECONDS..MAX_CLIP_SECONDS
  ) {
    throw IllegalArgumentException("Invalid settings")
  }
}

internal fun validId(id: String): Boolean = try {
  UUID.fromString(id)
  true
} catch (_: IllegalArgumentException) {
  false
}

internal fun validateEntry(entry: HistoryEntry) {
  if (!validId(entry.id) || entry.createdAt <= 0 ||
    entry.text.length > MAX_TEXT_CHARS || entry.raw.length > MAX_TEXT_CHARS ||
    entry.durationMs !in 0..MAX_DURATION_MS || entry.inferenceMs < 0
  ) {
    throw IllegalArgumentException("Invalid history entry")
  }
}

internal fun validateWord(word: VocabularyWord) {
  if (!validId(word.id) || word.word.trim().isEmpty() ||
    word.word.length > MAX_WORD_CHARS || word.replacement.length > MAX_WORD_CHARS
  ) {
    throw IllegalArgumentException("Invalid vocabulary word")
  }
}

/** Drop entries older than the retention window. retentionDays is always >= 7. */
internal fun AppData.pruned(nowMillis: Long): AppData {
  val cutoff = nowMillis - settings.retentionDays * 24L * 60L * 60L * 1000L
  val kept = history.filter { it.createdAt >= cutoff }
  return if (kept.size == history.size) this else copy(history = kept)
}

internal fun readableError(e: Throwable): String {
  val detail = e.message?.take(300)?.trim().orEmpty()
  return if (detail.isEmpty()) "Something went wrong." else detail
}

internal fun utcDate(millis: Long = System.currentTimeMillis()): String =
  SimpleDateFormat("yyyy-MM-dd", Locale.US).apply {
    timeZone = TimeZone.getTimeZone("UTC")
  }.format(Date(millis))

internal fun utcInstant(millis: Long): String =
  SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
    timeZone = TimeZone.getTimeZone("UTC")
  }.format(Date(millis))

internal fun exportBody(format: ExportFormat, data: AppData): String = when (format) {
  ExportFormat.JSON -> Json { prettyPrint = true }.encodeToString(AppData.serializer(), data)
  ExportFormat.TXT -> buildTxt(data)
  ExportFormat.CSV -> buildCsv(data)
}

internal fun buildTxt(data: AppData): String {
  val lines = ArrayList<String>()
  lines += "Stilltype history export"
  lines += "Exported: ${utcInstant(System.currentTimeMillis())}"
  lines += "Entries: ${data.history.size}"
  for (entry in data.history) {
    lines += "---"
    lines += utcInstant(entry.createdAt) + if (entry.favorite) " ★" else ""
    lines += entry.text
    lines += "Raw: ${entry.raw}"
    lines += "Duration: ${entry.durationMs} ms, inference: ${entry.inferenceMs} ms"
  }
  return lines.joinToString("\n") + "\n"
}

internal fun buildCsv(data: AppData): String {
  val rows = ArrayList<String>(data.history.size + 1)
  rows += "id,createdAt,text,raw,favorite,words,durationMs,inferenceMs"
  for (entry in data.history) {
    val words = entry.text.split(Regex("\\s+")).count { it.isNotEmpty() }
    rows += listOf(
      entry.id,
      entry.createdAt.toString(),
      entry.text,
      entry.raw,
      entry.favorite.toString(),
      words.toString(),
      entry.durationMs.toString(),
      entry.inferenceMs.toString(),
    ).joinToString(",") { csvCell(it) }
  }
  return rows.joinToString("\r\n") + "\r\n"
}

/** Neutralise spreadsheet formula prefixes, then quote and double any quotes. */
internal fun csvCell(value: String): String {
  val first = value.firstOrNull { !it.isWhitespace() && !it.isISOControl() }
  val guarded = if (first == '=' || first == '+' || first == '-' || first == '@') "'$value" else value
  return "\"" + guarded.replace("\"", "\"\"") + "\""
}

/**
 * Local app persistence in app-private storage. Writes are serialized through
 * DataStore; history is newest-first with in-place updates by id, expired
 * entries are pruned on load and on every write, and keepHistory=false skips
 * new entries without blocking explicit edits or deletes of existing ones.
 * Exports are local temporary files in cacheDir; the UI invokes the share
 * chooser. No network, no telemetry.
 */
class LocalAppRepository(context: Context) : AppRepository {
  private val appContext = context.applicationContext
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val exportMutex = Mutex()
  private val store: DataStore<AppData> = DataStoreFactory.create(
    serializer = AppDataSerializer,
    produceFile = { File(appContext.filesDir, "stilltype/state.json") },
  )

  private val _loading = MutableStateFlow(true)
  private val _error = MutableStateFlow<String?>(null)

  override val loading: StateFlow<Boolean> = _loading
  override val error: StateFlow<String?> = _error

  override val state: StateFlow<AppData> = store.data
    .catch { e ->
      _error.value = readableError(e)
      emit(AppData())
    }
    .map { sanitize(it).pruned(System.currentTimeMillis()) }
    .onEach { _loading.value = false }
    .stateIn(scope, SharingStarted.Eagerly, AppData())

  init {
    scope.launch {
      try {
        val current = store.data.first()
        val pruned = sanitize(current).pruned(System.currentTimeMillis())
        if (pruned.history.size != current.history.size) {
          store.updateData { pruned }
        }
      } catch (e: CancellationException) {
        throw e
      } catch (e: Exception) {
        _error.value = readableError(e)
      }
    }
  }

  override suspend fun updateSettings(settings: AppSettings) {
    mutate { current ->
      validateSettings(settings)
      current.copy(settings = settings)
    }
  }

  override suspend fun saveEntry(entry: HistoryEntry) {
    mutate { current ->
      validateEntry(entry)
      val history = current.history.toMutableList()
      val existing = history.indexOfFirst { it.id == entry.id }
      if (existing >= 0) {
        history[existing] = entry
      } else if (current.settings.keepHistory) {
        history.add(0, entry)
      }
      current.copy(history = history)
    }
  }

  override suspend fun deleteEntry(id: String) {
    if (!validId(id)) throw IllegalArgumentException("Invalid ID")
    mutate { current -> current.copy(history = current.history.filter { it.id != id }) }
  }

  override suspend fun clearHistory() {
    mutate { current -> current.copy(history = emptyList()) }
  }

  override suspend fun saveWord(word: VocabularyWord) {
    mutate { current ->
      validateWord(word)
      val vocabulary = current.vocabulary.toMutableList()
      val existing = vocabulary.indexOfFirst { it.id == word.id }
      if (existing >= 0) {
        vocabulary[existing] = word
      } else {
        vocabulary.add(word)
      }
      current.copy(vocabulary = vocabulary)
    }
  }

  override suspend fun deleteWord(id: String) {
    if (!validId(id)) throw IllegalArgumentException("Invalid ID")
    mutate { current -> current.copy(vocabulary = current.vocabulary.filter { it.id != id }) }
  }

  override suspend fun export(format: ExportFormat): Uri {
    return exportMutex.withLock {
      try {
        val current = try {
          store.data.first()
        } catch (e: Exception) {
          throw IOException(readableError(e), e)
        }
        val data = sanitize(current).pruned(System.currentTimeMillis())
        if (data.history.size != current.history.size) {
          // Best effort: keep serving the pruned in-memory result even if the
          // file cannot be rewritten right now.
          try {
            store.updateData { data }
          } catch (_: Exception) {
            // File left as-is; the served export is still pruned.
          }
        }
        val extension = when (format) {
          ExportFormat.TXT -> "txt"
          ExportFormat.JSON -> "json"
          ExportFormat.CSV -> "csv"
        }
        val dir = File(appContext.cacheDir, "exports")
        if (!dir.isDirectory && !dir.mkdirs()) throw IOException("Could not prepare the export folder.")
        val file = File(dir, "stilltype-${utcDate()}.${extension}")
        file.writeText(exportBody(format, data), Charsets.UTF_8)
        pruneStaleExports(dir, file.name)
        _error.value = null
        FileProvider.getUriForFile(appContext, EXPORT_AUTHORITY, file)
      } catch (e: CancellationException) {
        throw e
      } catch (e: Exception) {
        _error.value = readableError(e)
        throw e
      }
    }
  }

  private suspend fun mutate(transform: (AppData) -> AppData) {
    try {
      store.updateData { current -> transform(sanitize(current)).pruned(System.currentTimeMillis()) }
      _error.value = null
    } catch (e: CancellationException) {
      throw e
    } catch (e: Exception) {
      _error.value = readableError(e)
      throw e
    }
  }

  /** Invalid stored data surfaces an error and falls back to defaults in memory. */
  private fun sanitize(data: AppData): AppData {
    try {
      if (data.version != 1) throw IllegalArgumentException("Unsupported state version")
      validateSettings(data.settings)
      data.history.forEach(::validateEntry)
      data.vocabulary.forEach(::validateWord)
      return data
    } catch (e: IllegalArgumentException) {
      _error.value = readableError(e)
      return AppData()
    }
  }

  private fun pruneStaleExports(dir: File, keepName: String) {
    val files = dir.listFiles() ?: return
    for (file in files) {
      if (file.name != keepName && EXPORT_NAME.matches(file.name)) file.delete()
    }
  }
}
