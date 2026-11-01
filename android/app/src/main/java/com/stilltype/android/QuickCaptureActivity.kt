package com.stilltype.android

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.stilltype.android.ui.AppViewModel
import com.stilltype.android.ui.capture.CaptureViewModel
import com.stilltype.android.ui.capture.QuickCaptureSheet
import com.stilltype.android.ui.theme.StilltypeTheme

/** Private, visible capture sheet over the previous app. Not an overlay window. */
class QuickCaptureActivity : ComponentActivity() {
    private val appModel: AppViewModel by viewModels()
    private val capture: CaptureViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val data by appModel.data.collectAsStateWithLifecycle()
            StilltypeTheme(data.settings.theme) {
                QuickCaptureSheet(
                    app = appModel,
                    capture = capture,
                    onFinish = { finish() },
                    onOpenModels = {
                        capture.cancel()
                        startActivity(
                            Intent(this, MainActivity::class.java).putExtra(MainActivity.EXTRA_DESTINATION, "models"),
                        )
                        finish()
                    },
                )
            }
        }
    }
}
