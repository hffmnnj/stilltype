plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

// Bundled model source lives outside Git. Either pass
// -PmobileModelDir=/var/tmp/james/opencode/stilltype-mobile-model
// or export STILLTYPE_MOBILE_MODEL_DIR. When neither is set the task
// below is a no-op and the APK builds without model assets; the data
// layer then truthfully reports the model as not bundled.
val mobileModelDir: String = providers.gradleProperty("mobileModelDir")
    .orElse(providers.environmentVariable("STILLTYPE_MOBILE_MODEL_DIR"))
    .orElse("")
    .get()

val hasBundledModels = mobileModelDir.isNotBlank()

// Assets root is build/generated/modelAssets so APK entries land under
// assets/recognition/. A lightweight build without a model directory uses a
// distinct empty root so previously staged weights are never reused.
val stagedAssetsRoot = if (hasBundledModels) {
    layout.buildDirectory.dir("generated/modelAssets")
} else {
    layout.buildDirectory.dir("generated/modelAssetsEmpty")
}

val stagedRecognitionDir = layout.buildDirectory.dir("generated/modelAssets/recognition")

val stageBundledModels = tasks.register<Copy>("stageBundledModels") {
    onlyIf { mobileModelDir.isNotBlank() }
    doFirst {
        if (!file(mobileModelDir).isDirectory) {
            throw org.gradle.api.GradleException(
                "mobileModelDir '$mobileModelDir' does not exist or is not a directory. " +
                    "Pass -PmobileModelDir=<dir> or set STILLTYPE_MOBILE_MODEL_DIR to a directory " +
                    "containing manifest.json.",
            )
        }
    }
    from(mobileModelDir) {
        include(
            "manifest.json",
            "usr2-ctc-int8.onnx",
            "tokens.json",
            "yunet-2026may.onnx",
            "mean-face.json",
        )
    }
    into(stagedRecognitionDir)
}

android {
    namespace = "com.stilltype.android"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.stilltype.android"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {
            // ORT and OpenCV ship arm64-v8a and x86_64 binaries.
            ndk {
                abiFilters += listOf("arm64-v8a", "x86_64")
            }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    packaging {
        jniLibs {
            // ORT and OpenCV each bundle libc++_shared.so.
            pickFirsts += listOf("lib/**/libc++_shared.so")
        }
    }
    sourceSets {
        named("main") {
            assets.srcDir(stagedAssetsRoot)
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

tasks.named("preBuild") {
    dependsOn(stageBundledModels)
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2025.08.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.3")
    implementation("androidx.navigation:navigation-compose:2.9.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")
    implementation("androidx.datastore:datastore-preferences:1.1.7")

    implementation("androidx.camera:camera-core:1.5.1")
    implementation("androidx.camera:camera-camera2:1.5.1")
    implementation("androidx.camera:camera-lifecycle:1.5.1")
    implementation("androidx.camera:camera-view:1.5.1")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
    implementation("org.opencv:opencv:4.11.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")

    debugImplementation("androidx.compose.ui:ui-tooling")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
}
