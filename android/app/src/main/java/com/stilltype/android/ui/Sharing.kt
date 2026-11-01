package com.stilltype.android.ui

import android.content.ClipData
import android.content.ClipDescription
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PersistableBundle
import com.stilltype.android.core.ExportFormat

/** Explicit user-action helpers only. Nothing here pastes into other apps. */
object Sharing {
    fun copy(context: Context, text: String): Boolean = try {
        val clip = ClipData.newPlainText("Stilltype", text)
        if (Build.VERSION.SDK_INT >= 33) {
            clip.description.extras = PersistableBundle().apply { putBoolean(ClipDescription.EXTRA_IS_SENSITIVE, true) }
        }
        context.getSystemService(ClipboardManager::class.java).setPrimaryClip(clip)
        true
    } catch (_: Exception) {
        false
    }

    fun shareText(context: Context, text: String): Boolean = launch(
        context,
        Intent.createChooser(
            Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text),
            "Share transcript",
        ),
    )

    fun shareExport(context: Context, uri: Uri, format: ExportFormat): Boolean {
        val mime = when (format) {
            ExportFormat.TXT -> "text/plain"
            ExportFormat.JSON -> "application/json"
            ExportFormat.CSV -> "text/csv"
        }
        val send = Intent(Intent.ACTION_SEND)
            .setType(mime)
            .putExtra(Intent.EXTRA_STREAM, uri)
            .apply { clipData = ClipData.newRawUri("Stilltype export", uri) }
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        return launch(context, Intent.createChooser(send, "Export history").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
    }

    private fun launch(context: Context, intent: Intent): Boolean = try {
        context.startActivity(intent)
        true
    } catch (_: Exception) {
        false
    }
}
