package com.stilltype.android.engine

import com.stilltype.android.core.AlignmentFrame
import com.stilltype.android.core.FrameAnalysis
import com.stilltype.android.core.ImagePoint
import com.stilltype.android.core.MouthFrame
import com.stilltype.android.core.QualityCrops
import com.stilltype.android.core.QualityFrameAccumulator
import java.util.ArrayDeque
import kotlin.math.round
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Centered desktop alignment for accepted observations, with six frames of lookahead.
 * A missing or invalid pair ends a segment; unlike desktop, gaps are not interpolated.
 * All calls are serialized by the owning capture session, which must not mutate buffers.
 */
internal class CenteredFaceAligner(reference: List<ImagePoint>) : QualityFrameAccumulator {
    init {
        require(reference.size == 4 && reference.all { it.x.isFinite() && it.y.isFinite() })
        check(OpenCVLoader.initLocal()) { "OpenCV initialization failed" }
    }

    private data class Entry(val index: Int, val mouth: MouthFrame, val alignment: AlignmentFrame)

    private val target = MatOfPoint2f(*reference.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
    private val gray = Mat()
    private val aligned = Mat()
    private val face = Mat()
    private val frames = ArrayDeque<Entry>()
    private var previousTransform: Mat? = null
    private var received = 0
    private var next = 0
    private var closed = false

    val retainedFrameCount: Int get() = frames.size

    override fun offer(analysis: FrameAnalysis): List<MouthFrame> {
        check(!closed)
        val mouth = analysis.mouth
        val alignment = analysis.alignment
        if (mouth == null || alignment == null || !valid(alignment)) return finish()
        frames.addLast(Entry(received++, mouth, alignment))
        val output = mutableListOf<MouthFrame>()
        while (next < received && received - 1 - next >= minOf(6, next)) {
            output.add(crop(next, minOf(6, next)))
            next++
            discardOldFrames()
        }
        check(frames.size <= 13)
        return output
    }

    override fun finish(): List<MouthFrame> {
        check(!closed)
        val output = mutableListOf<MouthFrame>()
        while (next < received) {
            output.add(crop(next, minOf(6, next, received - 1 - next)))
            next++
            discardOldFrames()
        }
        reset()
        return output
    }

    override fun reset() {
        check(!closed)
        frames.clear()
        previousTransform?.release()
        previousTransform = null
        received = 0
        next = 0
    }

    private fun valid(frame: AlignmentFrame): Boolean =
        frame.width > 0 && frame.height > 0 &&
            frame.width.toLong() * frame.height == frame.grayscale.size.toLong() &&
            frame.landmarks.size == 4 && frame.landmarks.all { it.x.isFinite() && it.y.isFinite() }

    private fun discardOldFrames() {
        while (frames.isNotEmpty() && frames.first.index < next - 6) frames.removeFirst()
    }

    private fun crop(index: Int, radius: Int): MouthFrame {
        val window = frames.filter { it.index in index - radius..index + radius }
        check(window.size == radius * 2 + 1)
        val current = window[radius]
        val xs = FloatArray(4)
        val ys = FloatArray(4)
        for (entry in window) for (i in 0..3) {
            xs[i] += entry.alignment.landmarks[i].x
            ys[i] += entry.alignment.landmarks[i].y
        }
        for (i in 0..3) {
            xs[i] /= window.size.toFloat()
            ys[i] /= window.size.toFloat()
        }
        var currentX = 0f; var currentY = 0f; var smoothX = 0f; var smoothY = 0f
        for (i in 0..3) {
            currentX += current.alignment.landmarks[i].x
            currentY += current.alignment.landmarks[i].y
            smoothX += xs[i]; smoothY += ys[i]
        }
        val dx = currentX / 4f - smoothX / 4f
        val dy = currentY / 4f - smoothY / 4f
        for (i in 0..3) { xs[i] += dx; ys[i] += dy }

        val source = MatOfPoint2f(*Array(4) { Point(xs[it].toDouble(), ys[it].toDouble()) })
        val inliers = Mat()
        var estimated: Mat? = null
        try {
            estimated = Calib3d.estimateAffinePartial2D(source, target, inliers, Calib3d.LMEDS)
            if (estimated.empty() && previousTransform == null) {
                estimated.release()
                estimated = Calib3d.estimateAffinePartial2D(source, target)
            }
            val transform = if (estimated.empty()) previousTransform else estimated
            check(transform != null && !transform.empty()) { "Unable to estimate face alignment transform" }
            if (transform === estimated) {
                previousTransform?.release()
                previousTransform = transform.clone()
            }
            gray.create(current.alignment.height, current.alignment.width, CvType.CV_8UC1)
            gray.put(0, 0, current.alignment.grayscale)
            Imgproc.warpAffine(gray, aligned, transform, Size(256.0, 256.0), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT)
            val row0 = DoubleArray(3).also { transform.get(0, 0, it) }
            val row1 = DoubleArray(3).also { transform.get(1, 0, it) }
            val x = round((row0[0] * xs[3] + row0[1] * ys[3] + row0[2] - 48).coerceIn(0.0, 160.0)).toInt()
            val y = round((row1[0] * xs[3] + row1[1] * ys[3] + row1[2] - 48).coerceIn(0.0, 160.0)).toInt()
            val lipRoi = aligned.submat(Rect(x, y, 96, 96))
            try {
                val faceRoi = aligned.submat(Rect(32, 32, 192, 192))
                try {
                    val lipPixels = ByteArray(96 * 96).also { lipRoi.get(0, 0, it) }
                    Imgproc.resize(faceRoi, face, Size(96.0, 96.0), 0.0, 0.0, Imgproc.INTER_LINEAR)
                    val facePixels = ByteArray(96 * 96).also { face.get(0, 0, it) }
                    return current.mouth.copy(quality = QualityCrops(facePixels, lipPixels))
                } finally {
                    faceRoi.release()
                }
            } finally {
                lipRoi.release()
            }
        } finally {
            source.release(); inliers.release(); estimated?.release(); gray.release()
        }
    }

    override fun close() {
        if (closed) return
        reset()
        target.release(); gray.release(); aligned.release(); face.release()
        closed = true
    }
}
