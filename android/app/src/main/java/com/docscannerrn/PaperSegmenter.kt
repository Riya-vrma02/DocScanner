package com.docscannerrn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * ML page segmentation using `paper_seg.onnx` from
 * https://github.com/sgaofen/paper-extractor (MIT) — a U-Net with a
 * mobileone_s0 encoder (~5M params) trained to find sheets of paper in real
 * handheld photos: arbitrary angles, hands over the page, cluttered
 * backgrounds. Reported val IoU 0.96.
 *
 * This is the primary detector, replacing classical CV (Canny/Otsu/HSV
 * contours) which kept locking onto a near-full-frame contour when the page
 * sat on a patterned blanket or beside bright cloth.
 *
 * Contract (mirrors the reference extractor.py):
 *   input : [1, 3, 320, 320] float32, RGB, (px/255 - ImageNet mean) / std
 *   output: [1, 1, 320, 320] float32, min-max normalized to [0,1] as the mask
 *
 * Returns null when the model is unavailable or the mask can't be fit to one
 * quad, so callers can fall back to [DocumentDetector] or the full frame.
 */
class PaperSegmenter(private val context: Context) {

    companion object {
        private const val INPUT_SIZE = 320
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val MODEL_ASSET = "paper_seg.onnx"

        /**
         * Corner expansion about the centroid. The upstream reference uses 1.10 to
         * keep a sliver of background (it feeds an LLM prompt), but for a scanner
         * that margin is harmful: the quad no longer sits on the page corners, so
         * the perspective warp doesn't exactly rectify the page and the result
         * looks slanted with background wedges around it. 1.0 = crop tight on the
         * detected corners, which makes the warp straighten the page properly.
         * Nudge slightly above 1.0 only if you see edges being shaved off.
         */
        private const val EXPAND_SCALE = 1.0

        /** A plain 4-corner shape is kept only if it matches the outline this closely (IoU). */
        private const val QUAD_IOU = 0.985
    }

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var unavailable = false

    /**
     * Why the model produced no result, for diagnostics. Without this, a model
     * load failure or an output-shape mismatch is indistinguishable from
     * "no page found" — both silently fall back to the CV detector.
     */
    @Volatile
    var lastError: String? = null
        private set

    /** "loaded" | "not_loaded" | "unavailable" */
    val status: String
        get() = when {
            unavailable -> "unavailable"
            session != null -> "loaded"
            else -> "not_loaded"
        }

    @Synchronized
    private fun getSession(): OrtSession? {
        if (unavailable) return null
        session?.let { return it }
        return try {
            val e = OrtEnvironment.getEnvironment()
            val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            val s = e.createSession(bytes, OrtSession.SessionOptions())
            env = e
            session = s
            lastError = null
            s
        } catch (t: Throwable) {
            unavailable = true
            lastError = "load ${t::class.java.simpleName}: ${t.message}"
            null
        }
    }

    /**
     * Detects the page in [rgba] (an RGBA Mat of any size) and returns its 4
     * corners ordered [TL, TR, BR, BL] in **[rgba]'s own pixel coordinates**,
     * or null if unavailable / no single quad fits.
     */
    fun findPageQuad(rgba: Mat): Array<Point>? = withMask(rgba) { quadFromMask(it) }

    /**
     * Same segmentation mask, but returns the paper's actual outline (4..24 points,
     * in [rgba]'s pixel coordinates) so torn edges, cut corners and curved sides
     * survive. A clean rectangular page still comes back as exactly 4 points.
     */
    fun findPagePolygon(rgba: Mat): Array<Point>? = withMask(rgba) { polygonFromMask(it) }

    private fun withMask(rgba: Mat, fit: (Mat) -> Array<Point>?): Array<Point>? {
        val s = getSession() ?: return null
        val e = env ?: return null

        val fullW = rgba.cols()
        val fullH = rgba.rows()
        if (fullW <= 0 || fullH <= 0) return null

        val rgb = Mat()
        val resized = Mat()
        val maskSmall = Mat()
        val maskFull = Mat()
        try {
            // ---- Preprocess: RGB, 320x320 (INTER_AREA), ImageNet normalize, CHW.
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.resize(
                rgb, resized, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
                0.0, 0.0, Imgproc.INTER_AREA
            )

            val total = INPUT_SIZE * INPUT_SIZE
            val pixels = ByteArray(total * 3)
            resized.get(0, 0, pixels)

            val buffer = FloatBuffer.allocate(3 * total)
            for (c in 0 until 3) {
                val mean = MEAN[c]
                val std = STD[c]
                var i = c
                for (p in 0 until total) {
                    val v = (pixels[i].toInt() and 0xFF) / 255f
                    buffer.put((v - mean) / std)
                    i += 3
                }
            }
            buffer.rewind()

            val inputName = s.inputNames.iterator().next()
            val raw: Array<FloatArray>
            OnnxTensor.createTensor(
                e, buffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            ).use { tensor ->
                s.run(mapOf(inputName to tensor)).use { results ->
                    @Suppress("UNCHECKED_CAST")
                    val out = results[0].value as Array<Array<Array<FloatArray>>>
                    raw = out[0][0] // [320][320]
                }
            }

            // ---- Min-max normalize to [0,1] (matches reference), then to 0..255.
            var mn = Float.MAX_VALUE
            var mx = -Float.MAX_VALUE
            for (row in raw) {
                for (v in row) {
                    if (v < mn) mn = v
                    if (v > mx) mx = v
                }
            }
            val range = (mx - mn) + 1e-8f
            val maskBytes = ByteArray(total)
            var k = 0
            for (y in 0 until INPUT_SIZE) {
                val row = raw[y]
                for (x in 0 until INPUT_SIZE) {
                    val norm = (row[x] - mn) / range
                    maskBytes[k++] = (norm * 255f).toInt().coerceIn(0, 255).toByte()
                }
            }
            maskSmall.create(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
            maskSmall.put(0, 0, maskBytes)

            // ---- Upscale the mask to this image's resolution, then fit a quad.
            Imgproc.resize(
                maskSmall, maskFull, Size(fullW.toDouble(), fullH.toDouble()),
                0.0, 0.0, Imgproc.INTER_LINEAR
            )

            val result = fit(maskFull)
            lastError = if (result == null) "no_shape (mask empty or too irregular)" else null
            return result
        } catch (t: Throwable) {
            lastError = "infer ${t::class.java.simpleName}: ${t.message}"
            return null
        } finally {
            rgb.release(); resized.release(); maskSmall.release(); maskFull.release()
        }
    }

    /**
     * Mask -> 4-corner quad; a port of the reference `_quad_from_mask`:
     * threshold at 127, 5x5 open+close, keep contours >= 3% of area, convex
     * hull over the union of them, epsilon sweep for a 4-gon with a
     * minAreaRect fallback, reject when the mask fills < 55% of the quad
     * (too irregular), then expand corners outward 10%.
     */
    private fun quadFromMask(mask: Mat): Array<Point>? {
        val bin = Mat()
        try {
            Imgproc.threshold(mask, bin, 127.0, 255.0, Imgproc.THRESH_BINARY)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, kernel)
            Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel)

            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
            )
            hierarchy.release()
            if (contours.isEmpty()) return null

            val imgArea = (mask.rows() * mask.cols()).toDouble()
            var big = contours.filter { Imgproc.contourArea(it) > 0.03 * imgArea }
            if (big.isEmpty()) {
                val biggest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return null
                if (Imgproc.contourArea(biggest) < 0.01 * imgArea) return null
                big = listOf(biggest)
            }

            // Convex hull over the union of accepted contours, so one quad can
            // cover multiple sheets without concavities skewing the fit.
            val unionPoints = mutableListOf<Point>()
            for (c in big) unionPoints.addAll(c.toList())
            val unionMat = MatOfPoint(*unionPoints.toTypedArray())
            val hullIdx = MatOfInt()
            Imgproc.convexHull(unionMat, hullIdx)
            val unionArr = unionMat.toArray()
            val hullPts = hullIdx.toArray().map { unionArr[it] }.toTypedArray()
            hullIdx.release(); unionMat.release()
            if (hullPts.size < 4) return null

            val hull2f = MatOfPoint2f(*hullPts)
            var quad: Array<Point>? = null
            val peri = Imgproc.arcLength(hull2f, true)
            for (eps in listOf(0.02, 0.03, 0.05, 0.08, 0.12)) {
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(hull2f, approx, eps * peri, true)
                val isQuad = approx.total() == 4L
                if (isQuad) quad = approx.toArray()
                approx.release()
                if (isQuad) break
            }
            if (quad == null) {
                // Fallback: minimum rotated rectangle around the hull.
                val rect = Imgproc.minAreaRect(hull2f)
                val boxPts = arrayOfNulls<Point>(4)
                rect.points(boxPts)
                quad = boxPts.map { it ?: Point(0.0, 0.0) }.toTypedArray()
            }
            hull2f.release()

            // Fill-ratio sanity: don't force an L/U-shaped mask into a rectangle.
            val q2f = MatOfPoint2f(*quad)
            val quadArea = Imgproc.contourArea(q2f)
            q2f.release()
            val maskPixels = Core.countNonZero(bin).toDouble()
            if (quadArea > 0 && maskPixels / quadArea < 0.55) return null

            return DocumentDetector.orderCorners(expandQuad(quad))
        } catch (t: Throwable) {
            return null
        } finally {
            bin.release()
        }
    }

    /**
     * Mask -> irregular polygon. Unlike [quadFromMask] there is NO convex hull and
     * NO forcing to 4 corners, so concave tears and curved edges are preserved.
     * If a plain 4-corner shape matches the outline almost exactly (IoU >= QUAD_IOU),
     * that quad is returned instead, so clean flat pages keep the perspective-crop path.
     */
    private fun polygonFromMask(mask: Mat): Array<Point>? {
        val bin = Mat()
        try {
            Imgproc.threshold(mask, bin, 127.0, 255.0, Imgproc.THRESH_BINARY)
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
            Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, kernel)
            Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel)

            // Grow the mask a hair so the outline sits ON the paper edge rather than just
            // inside it (replaces the centroid-scaling expandQuad, which would distort
            // an irregular shape).
            val r = max(2, (min(bin.rows(), bin.cols()) * 0.006).toInt())
            val grow = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE, Size((2 * r + 1).toDouble(), (2 * r + 1).toDouble())
            )
            Imgproc.dilate(bin, bin, grow)

            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
            )
            hierarchy.release()

            val imgArea = (mask.rows() * mask.cols()).toDouble()
            val biggest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return null
            if (Imgproc.contourArea(biggest) < 0.03 * imgArea) return null

            // Small epsilon keeps bends and tears; loosen only if there are too many
            // points to edit comfortably.
            val c2f = MatOfPoint2f(*biggest.toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            var eps = 0.004
            do {
                Imgproc.approxPolyDP(c2f, approx, eps * peri, true)
                eps *= 1.4
            } while (approx.total() > 24 && eps < 0.05)
            if (approx.total() < 4) {
                approx.release(); c2f.release()
                return null
            }
            val poly = approx.toArray()
            approx.release()

            // Clean page? Prefer the plain quad when it hugs the outline almost exactly.
            for (qe in listOf(0.02, 0.03, 0.05)) {
                val q = MatOfPoint2f()
                Imgproc.approxPolyDP(c2f, q, qe * peri, true)
                val isQuad = q.total() == 4L
                val quadPts = if (isQuad) q.toArray() else null
                q.release()
                if (quadPts != null) {
                    if (iou(bin.size(), quadPts, poly) >= QUAD_IOU) {
                        c2f.release()
                        return quadPts
                    }
                    break
                }
            }
            c2f.release()
            return poly
        } catch (t: Throwable) {
            return null
        } finally {
            bin.release()
        }
    }

    /** Intersection-over-union of two filled polygons, rasterised at [size]. */
    private fun iou(size: Size, a: Array<Point>, b: Array<Point>): Double {
        val ma = Mat.zeros(size, CvType.CV_8UC1)
        val mb = Mat.zeros(size, CvType.CV_8UC1)
        Imgproc.fillPoly(ma, listOf(MatOfPoint(*a)), org.opencv.core.Scalar(255.0))
        Imgproc.fillPoly(mb, listOf(MatOfPoint(*b)), org.opencv.core.Scalar(255.0))
        val inter = Mat()
        val union = Mat()
        Core.bitwise_and(ma, mb, inter)
        Core.bitwise_or(ma, mb, union)
        val i = Core.countNonZero(inter).toDouble()
        val u = Core.countNonZero(union).toDouble()
        ma.release(); mb.release(); inter.release(); union.release()
        return if (u > 0) i / u else 0.0
    }

    /** Pushes corners outward from the centroid so a little background remains. */
    private fun expandQuad(quad: Array<Point>): Array<Point> {
        var cx = 0.0
        var cy = 0.0
        for (p in quad) {
            cx += p.x
            cy += p.y
        }
        cx /= quad.size
        cy /= quad.size
        return quad.map {
            Point(cx + (it.x - cx) * EXPAND_SCALE, cy + (it.y - cy) * EXPAND_SCALE)
        }.toTypedArray()
    }
}
