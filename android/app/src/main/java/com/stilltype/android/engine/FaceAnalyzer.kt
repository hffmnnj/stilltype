package com.stilltype.android.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import com.stilltype.android.core.FrameAnalysis
import com.stilltype.android.core.MouthFrame
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.round
import kotlin.math.sqrt
import org.json.JSONArray
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** One CPU YuNet session, reused across frames. All calls are serialized by the owner. */
internal class FaceAnalyzer(directory: File, private val env: OrtEnvironment) : AutoCloseable {
    init {
        check(OpenCVLoader.initLocal()) { "OpenCV initialization failed" }
    }

    private val session: OrtSession
    private val reference: Array<Point>
    private val source = Mat()
    private val rgba = Mat()
    private val resized = Mat()
    private val gray = Mat()
    private val warped = Mat()
    private val detectorBytes = ByteArray(WIDTH * HEIGHT * 4)
    private val detectorInput = FloatArray(WIDTH * HEIGHT * 3)
    private var previousMouth: ByteArray? = null

    init {
        val points = JSONArray(File(directory, "mean-face.json").readText())
        require(points.length() == 4) { "Invalid alignment reference" }
        reference = Array(4) { index ->
            val point = points.getJSONArray(index)
            Point(point.getDouble(0), point.getDouble(1))
        }
        session = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)
            env.createSession(File(directory, "yunet-2026may.onnx").absolutePath, options)
        }
    }

    fun analyze(bitmap: Bitmap, timestampMs: Long): FrameAnalysis {
        Utils.bitmapToMat(bitmap, source)
        rgba.create(480, 640, CvType.CV_8UC4)
        rgba.setTo(Scalar(0.0, 0.0, 0.0, 255.0))
        val scale = minOf(640.0 / bitmap.width, 480.0 / bitmap.height)
        val width = round(bitmap.width * scale).toInt().coerceIn(1, 640)
        val height = round(bitmap.height * scale).toInt().coerceIn(1, 480)
        val fitted = rgba.submat(Rect((640 - width) / 2, (480 - height) / 2, width, height))
        try { Imgproc.resize(source, fitted, Size(width.toDouble(), height.toDouble())) } finally { fitted.release() }
        Imgproc.resize(rgba, resized, Size(WIDTH.toDouble(), HEIGHT.toDouble()))
        resized.get(0, 0, detectorBytes)
        val area = WIDTH * HEIGHT
        var brightness = 0.0
        for (i in 0 until area) {
            val offset = i * 4
            val r = detectorBytes[offset].toInt() and 255
            val g = detectorBytes[offset + 1].toInt() and 255
            val b = detectorBytes[offset + 2].toInt() and 255
            detectorInput[i] = b.toFloat()
            detectorInput[area + i] = g.toFloat()
            detectorInput[2 * area + i] = r.toFloat()
            brightness += 0.2126 * r + 0.7152 * g + 0.0722 * b
        }
        val light = (brightness / area / 255.0).toFloat().coerceIn(0f, 1f)
        val points = OnnxTensor.createTensor(env, FloatBuffer.wrap(detectorInput), longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong())).use { tensor ->
            session.run(mapOf("input" to tensor)).use { outputs -> decode(outputs, 640, 480) }
        }
        if (points == null) {
            previousMouth = null
            return FrameAnalysis(brightness = light)
        }
        val pixels = align(points) ?: run {
            previousMouth = null
            return FrameAnalysis(faceDetected = true, brightness = light, hint = "Face the camera")
        }
        val previous = previousMouth
        var motion = 0f
        if (previous != null) {
            var sum = 0.0
            for (i in pixels.indices) sum += kotlin.math.abs((pixels[i].toInt() and 255) - (previous[i].toInt() and 255))
            motion = (sum / pixels.size / 32.0).toFloat().coerceIn(0f, 1f)
        }
        previousMouth = pixels
        return FrameAnalysis(MouthFrame(pixels, timestampMs), true, light, motion,
            if (light < 0.25f) "More light" else "Ready to dictate")
    }

    private fun decode(result: OrtSession.Result, width: Int, height: Int): Array<Point>? {
        var best = 0.75
        var bestPoints: Array<Point>? = null
        for (stride in intArrayOf(8, 16, 32)) {
            val columns = WIDTH / stride
            val count = columns * (HEIGHT / stride)
            fun output(name: String): FloatArray {
                val tensor = result.get(name).orElseThrow { IllegalStateException("Missing YuNet output $name") } as OnnxTensor
                val data = tensor.floatBuffer
                return FloatArray(data.remaining()).also { data.get(it) }
            }
            val cls = output("cls_$stride")
            val obj = output("obj_$stride")
            val bbox = output("bbox_$stride")
            val kps = output("kps_$stride")
            require(cls.size >= count && obj.size >= count && bbox.size >= count * 4 && kps.size >= count * 10)
            for (i in 0 until count) {
                val score = sqrt(cls[i].coerceIn(0f, 1f).toDouble() * obj[i].coerceIn(0f, 1f))
                if (!score.isFinite() || score < 0.75 || score <= best) continue
                val bw = exp(bbox[4 * i + 2].toDouble()) * stride
                val bh = exp(bbox[4 * i + 3].toDouble()) * stride
                if (!bw.isFinite() || !bh.isFinite() || bw <= 0 || bh <= 0) continue
                val pts = Array(5) { j ->
                    Point((i % columns + kps[i * 10 + j * 2]) * stride.toDouble(),
                        (i / columns + kps[i * 10 + j * 2 + 1]) * stride.toDouble())
                }
                if (pts.any { !it.x.isFinite() || !it.y.isFinite() || it.x < -4 || it.y < -4 || it.x > WIDTH + 4 || it.y > HEIGHT + 4 }) continue
                if (hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) <= 4) continue
                val left = if (pts[0].x <= pts[1].x) pts[0] else pts[1]
                val right = if (pts[0].x <= pts[1].x) pts[1] else pts[0]
                val mouth = Point((pts[3].x + pts[4].x) / 2, (pts[3].y + pts[4].y) / 2)
                bestPoints = arrayOf(left, right, pts[2], mouth).map {
                    Point(it.x * width / WIDTH, it.y * height / HEIGHT)
                }.toTypedArray()
                best = score
            }
        }
        return bestPoints
    }

    private fun align(points: Array<Point>): ByteArray? {
        val source = MatOfPoint2f(*points)
        val target = MatOfPoint2f(*reference)
        val inliers = Mat()
        var transform: Mat? = null
        try {
            transform = Calib3d.estimateAffinePartial2D(source, target, inliers, Calib3d.LMEDS)
            if (transform.empty()) return null
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.warpAffine(gray, warped, transform, Size(256.0, 256.0), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT)
            val row0 = DoubleArray(3).also { transform.get(0, 0, it) }
            val row1 = DoubleArray(3).also { transform.get(1, 0, it) }
            val x = round((row0[0] * points[3].x + row0[1] * points[3].y + row0[2] - 48).coerceIn(0.0, 160.0)).toInt()
            val y = round((row1[0] * points[3].x + row1[1] * points[3].y + row1[2] - 48).coerceIn(0.0, 160.0)).toInt()
            val roi = warped.submat(Rect(x, y, 96, 96))
            return try { ByteArray(96 * 96).also { roi.get(0, 0, it) } } finally { roi.release() }
        } finally {
            source.release(); target.release(); inliers.release(); transform?.release()
        }
    }

    override fun close() {
        session.close()
        source.release(); rgba.release(); resized.release(); gray.release(); warped.release()
        previousMouth = null
    }

    private companion object { const val WIDTH = 320; const val HEIGHT = 256 }
}
