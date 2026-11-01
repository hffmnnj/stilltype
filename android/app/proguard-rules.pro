# Release shrinking is currently disabled in app/build.gradle.kts.
# These rules apply if minification is enabled later.
-keepattributes Signature, InnerClasses, EnclosingMethod
-keep class com.stilltype.android.** { *; }
-keep class ai.onnxruntime.** { *; }
-keep class org.opencv.** { *; }
