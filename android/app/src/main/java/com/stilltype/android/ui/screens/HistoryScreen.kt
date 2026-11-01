package com.stilltype.android.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.calculateEndPadding
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.StarBorder
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.stilltype.android.core.HistoryEntry
import com.stilltype.android.ui.Sharing
import com.stilltype.android.ui.capture.DetailLine
import com.stilltype.android.ui.capture.fieldColors
import com.stilltype.android.ui.components.EmptyState
import com.stilltype.android.ui.components.QuietButton
import com.stilltype.android.ui.components.Rule
import com.stilltype.android.ui.theme.StilltypeTheme
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

private val TimeFormat: DateTimeFormatter = DateTimeFormatter.ofLocalizedTime(FormatStyle.SHORT)
private val DateFormat: DateTimeFormatter = DateTimeFormatter.ofLocalizedDate(FormatStyle.MEDIUM)

private fun HistoryEntry.localDate(): LocalDate = Instant.ofEpochMilli(createdAt).atZone(ZoneId.systemDefault()).toLocalDate()

private fun dayLabel(date: LocalDate): String {
    val today = LocalDate.now()
    return when (date) {
        today -> "Today"
        today.minusDays(1) -> "Yesterday"
        else -> DateFormat.format(date)
    }
}

private sealed interface Row2 {
    data class Header(val label: String) : Row2
    data class Item(val entry: HistoryEntry) : Row2
}

@Composable
fun HistoryScreen(
    padding: PaddingValues,
    history: List<HistoryEntry>,
    keepHistory: Boolean,
    onFavorite: (HistoryEntry, Boolean) -> Unit,
    onEdit: (HistoryEntry, String) -> Unit,
    onDelete: (HistoryEntry) -> Unit,
    onGoDictate: () -> Unit,
    notify: (String) -> Unit,
) {
    val context = LocalContext.current
    val dir = LocalLayoutDirection.current
    var query by rememberSaveable { mutableStateOf("") }
    var favoritesOnly by rememberSaveable { mutableStateOf(false) }
    var editingId by rememberSaveable { mutableStateOf<String?>(null) }

    val rows by remember(history, query, favoritesOnly) {
        derivedStateOf {
            val q = query.trim().lowercase()
            history
                .filter { (!favoritesOnly || it.favorite) && (q.isEmpty() || it.text.lowercase().contains(q) || it.raw.lowercase().contains(q)) }
                .sortedByDescending { it.createdAt }
                .groupBy { it.localDate() }
                .flatMap { (date, entries) -> listOf<Row2>(Row2.Header(dayLabel(date))) + entries.map { Row2.Item(it) } }
        }
    }
    val editing = history.firstOrNull { it.id == editingId }

    Box(
        Modifier
            .fillMaxSize()
            .padding(start = padding.calculateStartPadding(dir), end = padding.calculateEndPadding(dir), top = padding.calculateTopPadding()),
        contentAlignment = Alignment.TopCenter,
    ) {
        LazyColumn(
            Modifier.widthIn(max = 680.dp).fillMaxWidth().fillMaxSize(),
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = padding.calculateBottomPadding() + 16.dp),
        ) {
            if (history.isNotEmpty()) {
                item("search") {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = query,
                            onValueChange = { query = it },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                            placeholder = { Text("Search transcripts") },
                            leadingIcon = { Icon(Icons.Outlined.Search, contentDescription = null) },
                            trailingIcon = {
                                if (query.isNotEmpty()) {
                                    IconButton(onClick = { query = "" }) { Icon(Icons.Outlined.Close, contentDescription = "Clear search") }
                                }
                            },
                            shape = MaterialTheme.shapes.small,
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                            colors = fieldColors(),
                        )
                        FilterChip(
                            selected = favoritesOnly,
                            onClick = { favoritesOnly = !favoritesOnly },
                            label = { Text("Favorites") },
                            shape = MaterialTheme.shapes.small,
                            modifier = Modifier.padding(bottom = 4.dp),
                        )
                    }
                }
            }
            if (history.isEmpty()) {
                item("empty") {
                    EmptyState(
                        title = if (keepHistory) "No transcripts yet" else "History is off",
                        detail = if (keepHistory) "Recordings you make are saved here, on this device only." else "Turn on Keep history in Settings to save new transcripts.",
                        actionLabel = "Go to Dictate",
                        onAction = onGoDictate,
                    )
                }
            } else if (rows.isEmpty()) {
                item("none") {
                    EmptyState(
                        title = "No matches",
                        detail = "Nothing in your history fits this search.",
                        actionLabel = "Clear search",
                        onAction = { query = ""; favoritesOnly = false },
                    )
                }
            } else {
                items(rows, key = { row -> when (row) { is Row2.Header -> "h:${row.label}"; is Row2.Item -> row.entry.id } }) { row ->
                    when (row) {
                        is Row2.Header -> Text(
                            row.label,
                            style = MaterialTheme.typography.labelLarge,
                            color = StilltypeTheme.colors.muted,
                            modifier = Modifier.padding(top = 20.dp, bottom = 4.dp),
                        )
                        is Row2.Item -> {
                            HistoryRow(
                                row.entry,
                                onOpen = { editingId = row.entry.id },
                                onFavorite = { onFavorite(row.entry, !row.entry.favorite) },
                                onCopy = { notify(if (Sharing.copy(context, row.entry.text)) "Copied" else "Could not copy") },
                                onShare = { if (!Sharing.shareText(context, row.entry.text)) notify("No app can share this") },
                                onDelete = { onDelete(row.entry) },
                            )
                            Rule()
                        }
                    }
                }
            }
        }
    }

    editing?.let { entry ->
        EditDialog(
            entry,
            onDismiss = { editingId = null },
            onSave = { text -> onEdit(entry, text); editingId = null },
            onDelete = { onDelete(entry); editingId = null },
            onCopy = { text -> notify(if (Sharing.copy(context, text)) "Copied" else "Could not copy") },
            onShare = { text -> if (!Sharing.shareText(context, text)) notify("No app can share this") },
        )
    }
}

@Composable
private fun HistoryRow(
    entry: HistoryEntry,
    onOpen: () -> Unit,
    onFavorite: () -> Unit,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onDelete: () -> Unit,
) {
    var menu by remember { mutableStateOf(false) }
    val time = TimeFormat.format(Instant.ofEpochMilli(entry.createdAt).atZone(ZoneId.systemDefault()))
    Row(
        Modifier.fillMaxWidth().clickable(onClickLabel = "Edit transcript", onClick = onOpen).padding(vertical = 4.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Column(Modifier.weight(1f).padding(vertical = 10.dp)) {
            Text(entry.text, style = MaterialTheme.typography.bodyLarge, maxLines = 4, overflow = TextOverflow.Ellipsis)
            Text(
                "$time · ${"%.1f".format(entry.durationMs / 1000.0)} s clip",
                style = MaterialTheme.typography.bodySmall,
                color = StilltypeTheme.colors.muted,
                modifier = Modifier.padding(top = 4.dp),
            )
        }
        IconButton(onClick = onFavorite) {
            Icon(
                if (entry.favorite) Icons.Filled.Star else Icons.Outlined.StarBorder,
                contentDescription = if (entry.favorite) "Remove from favorites" else "Add to favorites",
            )
        }
        Box {
            IconButton(onClick = { menu = true }) { Icon(Icons.Outlined.MoreVert, contentDescription = "More actions") }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                DropdownMenuItem(text = { Text("Edit") }, onClick = { menu = false; onOpen() })
                DropdownMenuItem(text = { Text("Copy") }, onClick = { menu = false; onCopy() })
                DropdownMenuItem(text = { Text("Share") }, onClick = { menu = false; onShare() })
                DropdownMenuItem(
                    text = { Text("Delete", color = MaterialTheme.colorScheme.error) },
                    onClick = { menu = false; onDelete() },
                )
            }
        }
    }
}

@Composable
private fun EditDialog(
    entry: HistoryEntry,
    onDismiss: () -> Unit,
    onSave: (String) -> Unit,
    onDelete: () -> Unit,
    onCopy: (String) -> Unit,
    onShare: (String) -> Unit,
) {
    var text by rememberSaveable(entry.id) { mutableStateOf(entry.text) }
    AlertDialog(
        onDismissRequest = onDismiss,
        shape = MaterialTheme.shapes.large,
        title = { Text("Edit transcript", style = MaterialTheme.typography.titleMedium) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.padding(top = 4.dp)) {
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Transcript") },
                    minLines = 3,
                    maxLines = 8,
                    shape = MaterialTheme.shapes.small,
                    colors = fieldColors(),
                )
                DetailLine("Raw output", entry.raw.ifBlank { "(empty)" }, mono = false)
                DetailLine("Inference", "${entry.inferenceMs} ms · clip ${"%.1f".format(entry.durationMs / 1000.0)} s")
                Row {
                    QuietButton("Copy", { onCopy(text) })
                    QuietButton("Share", { onShare(text) })
                    QuietButton("Delete", onDelete, destructive = true)
                }
            }
        },
        confirmButton = { QuietButton("Save", { onSave(text.trim().ifEmpty { entry.text }) }, enabled = text.trim() != entry.text) },
        dismissButton = { QuietButton("Cancel", onDismiss) },
    )
}
