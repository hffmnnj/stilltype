package com.stilltype.android

import android.app.Application
import com.stilltype.android.core.AppRepository
import com.stilltype.android.core.ModelRepository
import com.stilltype.android.core.QualityModelRepository
import com.stilltype.android.core.VisualRecognizer
import com.stilltype.android.data.BundledModelRepository
import com.stilltype.android.data.BundledQualityModelRepository
import com.stilltype.android.data.LocalAppRepository
import com.stilltype.android.engine.AndroidRecognizer

/**
 * Lazily wired repositories. Nothing heavy (models, recognizer, camera)
 * is initialized at launch; the Quick Settings tile binding in particular
 * must not trigger model or camera setup.
 */
class StilltypeApplication : Application() {
    val repository: AppRepository by lazy { LocalAppRepository(this) }
    val models: ModelRepository by lazy { BundledModelRepository(this) }
    val qualityModels: QualityModelRepository by lazy { BundledQualityModelRepository(this) }
    val recognizer: VisualRecognizer by lazy { AndroidRecognizer(this, models, qualityModels) }
}
