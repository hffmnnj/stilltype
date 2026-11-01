package com.stilltype.android

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.stilltype.android.core.ACTION_CAPTURE
import com.stilltype.android.ui.AppViewModel
import com.stilltype.android.ui.Destination
import com.stilltype.android.ui.StilltypeApp
import com.stilltype.android.ui.capture.CaptureViewModel
import com.stilltype.android.ui.theme.StilltypeTheme

class MainActivity : ComponentActivity() {
    private val appModel: AppViewModel by viewModels()
    private val capture: CaptureViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A shortcut launch goes straight to the capture sheet and leaves no main screen behind it.
        if (intent?.action == ACTION_CAPTURE) {
            startActivity(
                Intent(this, QuickCaptureActivity::class.java)
                    .setAction(ACTION_CAPTURE)
                    .addFlags(Intent.FLAG_ACTIVITY_NO_ANIMATION),
            )
            finish()
            return
        }
        enableEdgeToEdge()
        val start = if (intent?.getStringExtra(EXTRA_DESTINATION) == "models") Destination.MODELS else Destination.DICTATE
        setContent {
            val data by appModel.data.collectAsStateWithLifecycle()
            StilltypeTheme(data.settings.theme) {
                StilltypeApp(appModel, capture, start)
            }
        }
    }

    companion object {
        const val EXTRA_DESTINATION = "destination"
    }
}
