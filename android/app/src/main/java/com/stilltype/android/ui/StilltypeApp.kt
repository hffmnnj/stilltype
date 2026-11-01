package com.stilltype.android.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Spellcheck
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.Spellcheck
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.width
import androidx.compose.ui.platform.LocalConfiguration
import com.stilltype.android.core.ExportFormat
import com.stilltype.android.ui.capture.CaptureViewModel
import com.stilltype.android.ui.components.SeamMark
import com.stilltype.android.ui.screens.DictateScreen
import com.stilltype.android.ui.screens.HistoryScreen
import com.stilltype.android.ui.screens.ModelsScreen
import com.stilltype.android.ui.screens.RAIL_DP
import com.stilltype.android.ui.screens.SettingsScreen
import com.stilltype.android.ui.screens.VocabularyScreen
import com.stilltype.android.ui.theme.StilltypeTheme
import com.stilltype.android.ui.theme.WordmarkStyle
import kotlinx.coroutines.launch

enum class Destination(val label: String, val selected: ImageVector?, val unselected: ImageVector?) {
    DICTATE("Dictate", Icons.Filled.Videocam, Icons.Outlined.Videocam),
    HISTORY("History", Icons.Filled.History, Icons.Outlined.History),
    VOCABULARY("Vocabulary", Icons.Filled.Spellcheck, Icons.Outlined.Spellcheck),
    SETTINGS("Settings", Icons.Filled.Settings, Icons.Outlined.Settings),

    /** Reached from readiness and Settings; not a bottom item. */
    MODELS("Models", null, null),
}

private val Tabs = Destination.entries.filter { it.selected != null }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StilltypeApp(app: AppViewModel, capture: CaptureViewModel, start: Destination) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    var destination by rememberSaveable { mutableStateOf(start) }
    var cameFrom by rememberSaveable { mutableStateOf(Destination.DICTATE) }
    var resumed by remember { mutableStateOf(false) }

    val data by app.data.collectAsStateWithLifecycle()
    val model by app.modelStatus.collectAsStateWithLifecycle()
    val loading by app.loading.collectAsStateWithLifecycle()
    val storeError by app.storeError.collectAsStateWithLifecycle()

    // ON_PAUSE, not ON_STOP: a translucent capture sheet leaves this activity STARTED.
    LifecycleResumeEffect(capture) {
        resumed = true
        onPauseOrDispose {
            resumed = false
            capture.onPaused()
        }
    }
    LaunchedEffect(destination) { if (destination != Destination.DICTATE) capture.cancel() }

    fun go(next: Destination) {
        if (next == destination) return
        cameFrom = destination
        destination = next
    }
    BackHandler(enabled = destination == Destination.MODELS) { destination = cameFrom.takeIf { it != Destination.MODELS } ?: Destination.SETTINGS }
    BackHandler(enabled = destination != Destination.DICTATE && destination != Destination.MODELS) { destination = Destination.DICTATE }

    val notify: (String) -> Unit = { message -> scope.launch { snackbar.showSnackbar(message) } }
    LaunchedEffect(storeError) { storeError?.let { snackbar.showSnackbar(it) } }

    val rail = LocalConfiguration.current.screenWidthDp >= RAIL_DP

    Box(Modifier.fillMaxSize()) {
    Row(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        if (rail) {
            NavigationRail(containerColor = MaterialTheme.colorScheme.background) {
                Tabs.forEach { tab ->
                    NavigationRailItem(
                        selected = tab == destination,
                        onClick = { go(tab) },
                        icon = { Icon(if (tab == destination) tab.selected!! else tab.unselected!!, contentDescription = null) },
                        label = { Text(tab.label) },
                    )
                }
            }
        }
        Scaffold(
            modifier = Modifier.weight(1f),
            containerColor = MaterialTheme.colorScheme.background,
            snackbarHost = { SnackbarHost(snackbar) },
            topBar = {
                TopAppBar(
                    title = {
                        if (destination == Destination.DICTATE) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                SeamMark(size = 28.dp)
                                Spacer(Modifier.width(10.dp))
                                Text("Stilltype", style = WordmarkStyle)
                            }
                        } else {
                            Text(destination.label, style = MaterialTheme.typography.headlineMedium)
                        }
                    },
                    navigationIcon = {
                        if (destination == Destination.MODELS) {
                            IconButton(onClick = { destination = cameFrom.takeIf { it != Destination.MODELS } ?: Destination.SETTINGS }) {
                                Icon(Icons.AutoMirrored.Outlined.ArrowBack, contentDescription = "Back")
                            }
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
                )
            },
            bottomBar = {
                if (!rail) {
                    NavigationBar(containerColor = MaterialTheme.colorScheme.background) {
                        Tabs.forEach { tab ->
                            NavigationBarItem(
                                selected = tab == destination,
                                onClick = { go(tab) },
                                icon = { Icon(if (tab == destination) tab.selected!! else tab.unselected!!, contentDescription = null) },
                                label = { Text(tab.label, style = MaterialTheme.typography.labelMedium) },
                            )
                        }
                    }
                }
            },
        ) { padding ->
            when (destination) {
                Destination.DICTATE -> DictateScreen(padding, app, capture, resumed, onOpenModels = { go(Destination.MODELS) }, notify = notify)
                Destination.HISTORY -> HistoryScreen(
                    padding,
                    history = data.history,
                    keepHistory = data.settings.keepHistory,
                    onFavorite = { entry, favorite -> app.setFavorite(entry, favorite, notify) },
                    onEdit = { entry, text -> app.editEntry(entry, text, notify) },
                    onDelete = { entry -> app.deleteEntry(entry.id, notify) },
                    onGoDictate = { go(Destination.DICTATE) },
                    notify = notify,
                )
                Destination.VOCABULARY -> VocabularyScreen(
                    padding,
                    words = data.vocabulary,
                    onSave = { existing, word, replacement -> app.saveWord(existing, word, replacement, notify) },
                    onDelete = { word -> app.deleteWord(word.id, notify) },
                )
                Destination.SETTINGS -> SettingsScreen(
                    padding,
                    settings = data.settings,
                    historyCount = data.history.size,
                    model = model,
                    onChange = { transform -> app.updateSettings(notify, transform) },
                    onExport = { format: ExportFormat ->
                        app.export(
                            format,
                            onReady = { uri -> if (!Sharing.shareExport(context, uri, format)) notify("No app can receive this file") },
                            onFailure = { notify(it) },
                        )
                    },
                    onClearHistory = { app.clearHistory(notify) },
                    onOpenModels = { go(Destination.MODELS) },
                    notify = notify,
                )
                Destination.MODELS -> ModelsScreen(
                    padding,
                    status = model,
                    compute = data.settings.computeMode,
                    onPrepare = app::prepareModel,
                    onOpenSettings = { go(Destination.SETTINGS) },
                )
            }
        }
    }
    }
    // Hold the first frame until the store resolves so the saved theme and history do not flash.
    if (loading) Spacer(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background))
}
