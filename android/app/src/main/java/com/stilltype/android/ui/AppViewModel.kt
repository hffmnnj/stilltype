package com.stilltype.android.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.stilltype.android.StilltypeApplication
import com.stilltype.android.core.AppData
import com.stilltype.android.core.AppSettings
import com.stilltype.android.core.ExportFormat
import com.stilltype.android.core.HistoryEntry
import com.stilltype.android.core.ModelStatus
import com.stilltype.android.core.VocabularyWord
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * Thin, failure-aware wrapper over the shared repositories. Nothing here owns a
 * second store; every mutation goes through [AppRepository]. [notify] receives
 * short outcome messages for the shell's snackbar.
 */
class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as StilltypeApplication
    private val repository = app.repository
    private val models = app.models

    val data: StateFlow<AppData> = repository.state
    val loading: StateFlow<Boolean> = repository.loading
    val storeError: StateFlow<String?> = repository.error
    val modelStatus: StateFlow<ModelStatus> = models.status

    private fun run(onFailure: (String) -> Unit = {}, block: suspend () -> Unit) {
        viewModelScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                onFailure(e.message ?: "That change could not be saved.")
            }
        }
    }

    fun updateSettings(notify: (String) -> Unit, transform: (AppSettings) -> AppSettings) = run(notify) {
        repository.updateSettings(transform(repository.state.value.settings))
    }

    fun setFavorite(entry: HistoryEntry, favorite: Boolean, notify: (String) -> Unit) = run(notify) {
        repository.saveEntry(entry.copy(favorite = favorite))
    }

    /** Edits keep the original ID, raw output and timing. */
    fun editEntry(entry: HistoryEntry, text: String, notify: (String) -> Unit) = run(notify) {
        repository.saveEntry(entry.copy(text = text))
    }

    fun deleteEntry(id: String, notify: (String) -> Unit) = run(notify) { repository.deleteEntry(id) }

    fun clearHistory(notify: (String) -> Unit) = run(notify) {
        repository.clearHistory()
        notify("History cleared")
    }

    fun saveWord(existing: VocabularyWord?, word: String, replacement: String, notify: (String) -> Unit) = run(notify) {
        repository.saveWord(
            VocabularyWord(
                id = existing?.id ?: UUID.randomUUID().toString(),
                word = word.trim(),
                replacement = replacement.trim(),
            ),
        )
    }

    fun deleteWord(id: String, notify: (String) -> Unit) = run(notify) { repository.deleteWord(id) }

    fun export(format: ExportFormat, onReady: (Uri) -> Unit, onFailure: (String) -> Unit) = run({ onFailure(it) }) {
        onReady(repository.export(format))
    }

    fun prepareModel() {
        viewModelScope.launch {
            try {
                models.prepareBundled()
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // The repository publishes the failure on ModelStatus.error.
            }
        }
    }
}
