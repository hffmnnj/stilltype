package com.stilltype.android.platform

import android.app.Activity
import android.app.StatusBarManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.drawable.Icon
import android.os.Build
import com.stilltype.android.R
import com.stilltype.android.core.ACTION_CAPTURE

/** UI-owned targets, referenced by name so platform compiles before UI lands. */
internal const val QUICK_CAPTURE_ACTIVITY_CLASS = "com.stilltype.android.QuickCaptureActivity"
internal const val MAIN_ACTIVITY_CLASS = "com.stilltype.android.MainActivity"

/** Explicit capture intent with no extras; only the capture activity may read it. */
internal fun captureIntent(context: Context): Intent =
    Intent(ACTION_CAPTURE).setClassName(context.packageName, QUICK_CAPTURE_ACTIVITY_CLASS)

/** Honest outcome of a tile-add request; never report success unless the system did. */
enum class TileAddResult {
    ADDED,
    ALREADY_ADDED,
    DISMISSED,
    UNSUPPORTED,
    ERROR,
}

/**
 * User-initiated shortcuts into quick capture. Call only from a visible user
 * action (settings button, onboarding affordance); never automatically on
 * launch. These helpers own no UI state and never touch clipboard or paste.
 */
object QuickAccess {

    /**
     * Asks the system to add [StilltypeTileService] to Quick Settings.
     * API 33+ only; older devices get UNSUPPORTED so the UI can show manual
     * add guidance instead of pretending the tile was added.
     */
    fun requestTile(activity: Activity, result: (TileAddResult) -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            result(TileAddResult.UNSUPPORTED)
            return
        }
        try {
            val manager = activity.getSystemService(StatusBarManager::class.java)
                ?: return result(TileAddResult.ERROR)
            val component = ComponentName(activity, StilltypeTileService::class.java)
            val icon = Icon.createWithResource(activity, android.R.drawable.ic_menu_camera)
            manager.requestAddTileService(component, TILE_LABEL, icon, activity.mainExecutor) { code ->
                result(
                    when (code) {
                        StatusBarManager.TILE_ADD_REQUEST_RESULT_TILE_ADDED -> TileAddResult.ADDED
                        StatusBarManager.TILE_ADD_REQUEST_RESULT_TILE_ALREADY_ADDED -> TileAddResult.ALREADY_ADDED
                        StatusBarManager.TILE_ADD_REQUEST_RESULT_TILE_NOT_ADDED -> TileAddResult.DISMISSED
                        else -> TileAddResult.ERROR
                    },
                )
            }
        } catch (_: Exception) {
            result(TileAddResult.ERROR)
        }
    }

    /**
     * Requests a pinnable launcher shortcut that opens quick capture via
     * MainActivity. Returns the system's acceptance, not the user's choice;
     * false means pinning is unsupported or unavailable here.
     */
    fun requestPinnedShortcut(context: Context): Boolean {
        return try {
            val manager = context.getSystemService(ShortcutManager::class.java) ?: return false
            if (!manager.isRequestPinShortcutSupported) return false
            val intent = Intent(ACTION_CAPTURE)
                .setClassName(context.packageName, MAIN_ACTIVITY_CLASS)
            val info = ShortcutInfo.Builder(context, PIN_SHORTCUT_ID)
                .setShortLabel(SHORTCUT_LABEL)
                .setIcon(Icon.createWithResource(context, R.drawable.ic_stilltype))
                .setIntent(intent)
                .build()
            manager.requestPinShortcut(info, null)
        } catch (_: Exception) {
            false
        }
    }

    /** Opens the explicit capture activity; adds NEW_TASK for non-Activity callers. */
    fun openCapture(context: Context) {
        try {
            val intent = captureIntent(context)
            if (context !is Activity) intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        } catch (_: Exception) {
            // Invoked from visible UI only; never crash the caller.
        }
    }

    private const val TILE_LABEL = "Stilltype"
    private const val PIN_SHORTCUT_ID = "stilltype-dictate"
    private const val SHORTCUT_LABEL = "Dictate"
}
