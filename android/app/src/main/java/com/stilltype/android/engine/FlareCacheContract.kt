package com.stilltype.android.engine

import java.io.File

/** Fixed layout of the pinned 2026 Base decoder; not a generic model format. */
internal object FlareCacheLayout {
    const val LAYERS = 6
    const val HEADS = 4
    const val HEAD_DIM = 192
    const val MAX_NATIVE_ROWS = 5
    const val MAX_PREFIX = 128
    const val VOCABULARY = 1000
    const val EOS = 2
    const val PAD = 1
    val suffixes: List<String> = (0 until LAYERS).flatMap { listOf("${it}_key", "${it}_value") }
}

/** The caller verifies artifact provenance, file sizes and hashes before construction. */
internal data class FlareGraphFiles(
    val encoder: File,
    val cross: File,
    val decoder: File,
    val cacheGather: File,
    val cacheMerge: File,
)

internal data class FlareBeamTimings(
    val encoderMs: Double,
    val crossMs: Double,
    val decoderRunMs: Double,
    val cacheGatherMs: Double,
    val cacheMergeMs: Double,
    val outputCopyMs: Double,
    val candidateSelectionMs: Double,
    val completeDecoderLoopMs: Double,
    val decoderCalls: Int,
    val maximumNativeRows: Int,
)

internal data class FlareBeamOutput(
    /** Includes the initial EOS prefix and naturally finalized EOS. */
    val tokenIds: List<Long>,
    val timings: FlareBeamTimings,
    val backend: String = "CPU encoder; CPU decoder",
)
