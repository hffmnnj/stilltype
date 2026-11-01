package com.stilltype.android.core

data class ImagePoint(val x: Float, val y: Float)

/** A single accepted detector frame. Buffers are owned by the capture session. */
data class AlignmentFrame(
    val grayscale: ByteArray,
    val width: Int,
    val height: Int,
    val landmarks: List<ImagePoint>,
)

/** Temporally aligned 96x96 crops, separate from the unchanged activity-gate ROI. */
data class QualityCrops(val face: ByteArray, val lip: ByteArray)
