package com.stilltype.android.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowForward
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.DeleteOutline
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.stilltype.android.core.VocabularyWord
import com.stilltype.android.ui.capture.fieldColors
import com.stilltype.android.ui.components.EmptyState
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.Rule
import com.stilltype.android.ui.components.SecondaryButton
import com.stilltype.android.ui.theme.StilltypeTheme

private const val MAX_CHARS = 256

private fun normalized(value: String) = value.trim().lowercase().split(Regex("\\s+")).joinToString(" ")

@Composable
fun VocabularyScreen(
    padding: PaddingValues,
    words: List<VocabularyWord>,
    onSave: (VocabularyWord?, String, String) -> Unit,
    onDelete: (VocabularyWord) -> Unit,
) {
    // "new" opens a blank dialog; an ID edits that entry.
    var editing by rememberSaveable { mutableStateOf<String?>(null) }

    ScreenColumn(padding) {
        Row(Modifier.fillMaxWidth().padding(bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(
                "Exact phrases are replaced after recognition. Matching ignores case.",
                style = MaterialTheme.typography.bodyMedium,
                color = StilltypeTheme.colors.muted,
                modifier = Modifier.weight(1f).padding(end = 12.dp),
            )
            SecondaryButton("Add", { editing = "new" }, icon = Icons.Outlined.Add)
        }
        if (words.isEmpty()) {
            EmptyState(
                title = "No custom words",
                detail = "Add names, jargon or spellings the model gets wrong. Nothing is guessed or fuzzy-matched.",
            )
        } else {
            Rule()
            words.forEach { word ->
                Row(
                    Modifier.fillMaxWidth().clickable(onClickLabel = "Edit ${word.word}") { editing = word.id }.padding(start = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Row(
                        Modifier.weight(1f).padding(vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(word.word, style = MaterialTheme.typography.bodyLarge, maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                        if (word.replacement.isNotBlank() && word.replacement != word.word) {
                            Icon(Icons.AutoMirrored.Outlined.ArrowForward, contentDescription = "becomes", tint = StilltypeTheme.colors.muted)
                            Text(word.replacement, style = MaterialTheme.typography.bodyLarge, maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                        }
                    }
                    IconButton(onClick = { onDelete(word) }) {
                        Icon(Icons.Outlined.DeleteOutline, contentDescription = "Delete ${word.word}")
                    }
                }
                Rule()
            }
        }
    }

    val key = editing
    if (key != null) {
        val existing = words.firstOrNull { it.id == key }
        WordDialog(
            existing = existing,
            others = words.filter { it.id != existing?.id },
            onDismiss = { editing = null },
            onSave = { word, replacement -> onSave(existing, word, replacement); editing = null },
        )
    }
}

@Composable
private fun WordDialog(
    existing: VocabularyWord?,
    others: List<VocabularyWord>,
    onDismiss: () -> Unit,
    onSave: (String, String) -> Unit,
) {
    var word by rememberSaveable { mutableStateOf(existing?.word.orEmpty()) }
    var replacement by rememberSaveable { mutableStateOf(existing?.replacement.orEmpty()) }
    var touched by remember { mutableStateOf(false) }
    val error = when {
        word.isBlank() -> "Enter the word or phrase to match"
        word.length > MAX_CHARS || replacement.length > MAX_CHARS -> "Keep each entry under $MAX_CHARS characters"
        others.any { normalized(it.word) == normalized(word) } -> "This phrase is already in your list"
        else -> null
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        shape = MaterialTheme.shapes.large,
        title = { Text(if (existing == null) "Add word" else "Edit word", style = MaterialTheme.typography.titleMedium) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(top = 4.dp)) {
                OutlinedTextField(
                    value = word,
                    onValueChange = { word = it; touched = true },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Heard as") },
                    singleLine = true,
                    isError = touched && error != null,
                    supportingText = { if (touched && error != null) Text(error) },
                    shape = MaterialTheme.shapes.small,
                    colors = fieldColors(),
                )
                OutlinedTextField(
                    value = replacement,
                    onValueChange = { replacement = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Write as (optional)") },
                    supportingText = { Text("Blank keeps the spelling above, which fixes capitalization.") },
                    singleLine = true,
                    shape = MaterialTheme.shapes.small,
                    colors = fieldColors(),
                )
            }
        },
        confirmButton = {
            QuietButton("Save", { if (error == null) onSave(word.trim(), replacement.trim()) else touched = true })
        },
        dismissButton = { QuietButton("Cancel", onDismiss) },
    )
}
