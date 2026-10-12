package com.docscannerrn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * ML page segmentation using `paper_seg.onnx` (U-Net, mobileone_s0 encoder).
 *
 * Contract:
 *   input : [1, 3, 320, 320] float32, RGB, (px/255 - ImageNet mean) / std
 *   output: [1, 1, 320, 320] float32, min-max normalized to [0,1] as the mask
 *
 * Improvements over the first version:
 *  1. Non-paper suppression: pixels much darker than the paper core (laptop bezel,
 *     keyboard, shadows) are removed from the mask before contour extraction.
 *  2. Pages are treated as convex (convex hull) so notches / leaks don't bend the quad.
 *  3. Quad corners are refined by robust line fitting on the real contour points and
 *     intersecting the 4 lines -> corners sit exactly on the page edges.
 *  4. Polygon fallback is capped (MAX_POLY_POINTS) so the corner editor never gets
 *     clusters of dots.
 */
class PaperSegmenter(private val context: Context) {

    companion object {
        private const val INPUT_SIZE = 320
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val MODEL_ASSET = "paper_seg.onnx"

        /** Fallback / clamp range for the per-frame (Otsu) mask threshold, 0..255. */
        private const val MASK_THRESHOLD = 150.0
        private const val THRESH_MIN = 110.0
        private const val THRESH_MAX = 175.0

        /**
         * Pixels whose Lab colour is farther than this from the paper's median colour are
         * treated as background (bezel, keyboard, table). Lower = more aggressive.
         */
        private const val COLOR_DISTANCE = 70f

        /**
         * If suppression would delete more than this fraction of the mask, the page itself
         * is colourful/patterned (not a leak), so suppression is skipped for that frame.
         */
        private const val MAX_SUPPRESS_FRACTION = 0.20

        /** Quad is accepted if it overlaps the page outline at least this well (IoU). */
        private const val QUAD_IOU = 0.93

        private const val TAG = "PaperSegmenter"

        /** Working size (long edge, px) for splitting overlapping sheets. */
        private const val SPLIT_WORK_SIZE = 640.0

        /** Min length of an edge line (fraction of the short side) that may cut the mask (Hough mode). */
        private const val SPLIT_MIN_LINE = 0.25

        /**
         * A blob whose convex hull fills at least this fraction of its min-area rectangle is
         * treated as a single clean sheet and is NOT split. Lower it if stacked sheets that
         * line up neatly are not being split; raise it if single pages get altered.
         */
        private const val SPLIT_RECT_GATE = 0.90

        /** Max points returned when the page is not a clean quad. */
        private const val MAX_POLY_POINTS = 16

        /** Weight of the newest frame when averaging live masks (lower = steadier, laggier). */
        private const val TEMPORAL_ALPHA = 0.7
    }

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var unavailable = false
    private var accMask: Mat? = null
    @Volatile private var maskThr = MASK_THRESHOLD

    /** Forget the averaged live mask (call when the page is lost). */
    fun resetSmoothing() {
        accMask?.release()
        accMask = null
    }

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

    /** 4 corners [TL, TR, BR, BL] in [rgba]'s own pixel coordinates, or null. */
    fun findPageQuad(rgba: Mat): Array<Point>? = withMask(rgba) { m, img -> quadFromMask(m, img) }

    /**
     * Page outline (4..MAX_POLY_POINTS points). A clean page comes back as exactly 4
     * ordered corners.
     */
    fun findPagePolygon(rgba: Mat, live: Boolean = false): Array<Point>? =
        withMask(rgba, live) { m, img -> polygonFromMask(m, img) }

    private fun withMask(rgba: Mat, temporal: Boolean = false, fit: (Mat, Mat) -> Array<Point>?): Array<Point>? {
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
                    raw = out[0][0]
                }
            }

            var mn = Float.MAX_VALUE
            var mx = -Float.MAX_VALUE
            for (row in raw) for (v in row) {
                if (v < mn) mn = v
                if (v > mx) mx = v
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

            if (temporal) {
                val acc = accMask
                if (acc == null || acc.rows() != INPUT_SIZE) {
                    val a = Mat()
                    maskSmall.convertTo(a, CvType.CV_32F)
                    accMask = a
                } else {
                    Imgproc.accumulateWeighted(maskSmall, acc, TEMPORAL_ALPHA)
                    acc.convertTo(maskSmall, CvType.CV_8U)
                }
            }

            // Remove laptop bezel / keyboard / shadow regions the model wrongly marked as paper.
            suppressNonPaper(resized, maskSmall)

            // Per-frame threshold: colored pages give lower-confidence masks than white ones.
            val tmp = Mat()
            val otsu = Imgproc.threshold(maskSmall, tmp, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
            tmp.release()
            maskThr = otsu.coerceIn(THRESH_MIN, THRESH_MAX)

            Imgproc.resize(
                maskSmall, maskFull, Size(fullW.toDouble(), fullH.toDouble()),
                0.0, 0.0, Imgproc.INTER_LINEAR
            )

            val result = fit(maskFull, rgba)
            lastError = if (result == null) "no_shape (mask empty or too irregular)" else null
            return result
        } catch (t: Throwable) {
            lastError = "infer ${t::class.java.simpleName}: ${t.message}"
            return null
        } finally {
            rgb.release(); resized.release(); maskSmall.release(); maskFull.release()
        }
    }

    // ------------------------------------------------------------------ mask cleanup

    /**
     * Zeroes mask pixels whose colour is far from the paper's median colour (laptop bezel,
     * keyboard, dark table...). Colour-aware so coloured pages are not cut up, and it
     * backs off entirely when it would remove a big part of the page.
     */
    private fun suppressNonPaper(rgb320: Mat, mask: Mat) {
        val lab = Mat()
        val blur = Mat()
        val core = Mat()
        val f = Mat()
        val dist = Mat()
        val far32 = Mat()
        val far = Mat()
        val maskBin = Mat()
        val removed = Mat()
        val chans = ArrayList<Mat>()
        try {
            Imgproc.cvtColor(rgb320, lab, Imgproc.COLOR_RGB2Lab)
            Imgproc.GaussianBlur(lab, blur, Size(9.0, 9.0), 0.0)

            val total = INPUT_SIZE * INPUT_SIZE
            Imgproc.threshold(mask, core, 200.0, 255.0, Imgproc.THRESH_BINARY)
            if (Core.countNonZero(core) < total * 0.02) {
                Imgproc.threshold(mask, core, MASK_THRESHOLD, 255.0, Imgproc.THRESH_BINARY)
            }
            val er = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
            Imgproc.erode(core, core, er)
            er.release()
            if (Core.countNonZero(core) < total * 0.01) return

            // Median Lab colour of the confident paper core.
            val lb = ByteArray(total * 3)
            val cb = ByteArray(total)
            blur.get(0, 0, lb)
            core.get(0, 0, cb)
            val med = DoubleArray(3)
            for (ch in 0 until 3) {
                val hist = IntArray(256)
                var n = 0
                for (i in 0 until total) {
                    if (cb[i].toInt() != 0) {
                        hist[lb[i * 3 + ch].toInt() and 0xFF]++
                        n++
                    }
                }
                var acc = 0
                for (v in 0..255) {
                    acc += hist[v]
                    if (acc >= n / 2) { med[ch] = v.toDouble(); break }
                }
            }

            // Euclidean Lab distance to the paper colour.
            blur.convertTo(f, CvType.CV_32FC3)
            Core.subtract(f, Scalar(med[0], med[1], med[2]), f)
            Core.multiply(f, f, f)
            Core.split(f, chans)
            Core.add(chans[0], chans[1], dist)
            Core.add(dist, chans[2], dist)
            Core.sqrt(dist, dist)

            Imgproc.threshold(dist, far32, COLOR_DISTANCE.toDouble(), 255.0, Imgproc.THRESH_BINARY)
            far32.convertTo(far, CvType.CV_8U)

            // Safety: if this would eat a big chunk of the mask, the page is just colourful.
            Imgproc.threshold(mask, maskBin, MASK_THRESHOLD, 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_and(far, maskBin, removed)
            val maskPx = Core.countNonZero(maskBin)
            if (maskPx == 0 || Core.countNonZero(removed) > MAX_SUPPRESS_FRACTION * maskPx) return

            mask.setTo(Scalar(0.0), far)
        } finally {
            lab.release(); blur.release(); core.release(); f.release(); dist.release()
            far32.release(); far.release(); maskBin.release(); removed.release()
            chans.forEach { it.release() }
        }
    }

    // ------------------------------------------------------------------ shape fitting

    /** Binarise + clean [mask] into [bin]; return the largest contour (dense points) or null. */
    private fun largestContour(mask: Mat, bin: Mat, rgba: Mat): Array<Point>? {
        Imgproc.threshold(mask, bin, maskThr, 255.0, Imgproc.THRESH_BINARY)

        var ks = (min(bin.rows(), bin.cols()) * 0.008).toInt()
        ks = max(5, ks) or 1 // odd, >= 5
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(ks.toDouble(), ks.toDouble()))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, kernel)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        // Several overlapping sheets come out of the model as one blob: keep only the top one.
        isolateTopSheet(bin, rgba)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        // CHAIN_APPROX_NONE: dense points along straight edges are needed for line fitting.
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        hierarchy.release()

        val imgArea = (bin.rows() * bin.cols()).toDouble()
        val biggest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return null
        if (Imgproc.contourArea(biggest) < 0.03 * imgArea) return null
        return biggest.toArray()
    }

    private class SplitCfg(val name: String, val hough: Boolean, val low: Double, val high: Double, val dil: Int = 5)

    private val splitConfigs = listOf(
        SplitCfg("hough", true, 15.0, 45.0),
        SplitCfg("edges-lo", false, 8.0, 24.0),
        SplitCfg("edges-mid", false, 15.0, 45.0),
        SplitCfg("edges-hi", false, 30.0, 90.0),
        SplitCfg("edges-wide-lo", false, 8.0, 24.0, 9),
        SplitCfg("edges-wide-mid", false, 15.0, 45.0, 9)
    )

    private class Piece(val mask: Mat, val quality: Double, var grow: Int = 5)

    /** hull area / min-area-rect area of the largest blob: ~1 for a clean sheet, lower for piles. */
    private fun rectangularity(maskSmall: Mat): Double {
        val tmp = maskSmall.clone()
        val contours = mutableListOf<MatOfPoint>()
        val hier = Mat()
        try {
            Imgproc.findContours(tmp, contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val big = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return 0.0
            val hull = hullOf(big.toArray())
            if (hull.size < 3) return 0.0
            val h2 = MatOfPoint2f(*hull)
            val hullArea = Imgproc.contourArea(h2)
            val rect = Imgproc.minAreaRect(h2)
            h2.release()
            val ra = rect.size.width * rect.size.height
            return if (ra > 0) (hullArea / ra).coerceIn(0.0, 1.0) else 0.0
        } finally {
            tmp.release(); hier.release(); contours.forEach { it.release() }
        }
    }

    /** How many of the 4 image borders the mask touches substantially (0..4). */
    private fun bordersTouched(m: Mat): Int {
        val w = m.cols()
        val h = m.rows()
        val t = 3
        val strips = listOf(
            m.submat(0, t, 0, w), m.submat(h - t, h, 0, w),
            m.submat(0, h, 0, t), m.submat(0, h, w - t, w)
        )
        val lens = intArrayOf(w, w, h, h)
        var n = 0
        for (i in 0 until 4) {
            if (Core.countNonZero(strips[i]) > 0.15 * lens[i] * t) n++
            strips[i].release()
        }
        return n
    }

    /** From the pieces left after cutting, pick the large, central, rectangular one. */
    private fun pickPiece(pieces: Mat, total: Double, ww: Int, wh: Int): Piece? {
        val labels = Mat()
        val stats = Mat()
        val cent = Mat()
        try {
            val n = Imgproc.connectedComponentsWithStats(pieces, labels, stats, cent, 8, CvType.CV_32S)
            if (n <= 1 || total <= 0) return null
            val halfDiag = 0.5 * hypot(ww.toDouble(), wh.toDouble())
            var best = -1
            var bestScore = 0.0
            var bestArea = 0.0
            var bestD = 0.0
            for (i in 1 until n) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                if (area < 0.08 * total) continue
                val c = cent.get(i, 0)
                val d = hypot(c[0] - ww / 2.0, c[1] - wh / 2.0) / halfDiag
                val score = area / (1.0 + 2.0 * d)
                if (score > bestScore) { bestScore = score; best = i; bestArea = area; bestD = d }
            }
            Log.d(TAG, "    pieces=%d bestFrac=%.2f".format(n - 1, bestArea / total))
            if (best < 0 || bestArea > 0.92 * total || bestArea < 0.12 * total) return null
            val chosen = Mat()
            Core.compare(labels, Scalar(best.toDouble()), chosen, Core.CMP_EQ)
            val rect = rectangularity(chosen)
            val quality = rect * rect * (bestArea / total) / (1.0 + 2.0 * bestD)
            return Piece(chosen, quality)
        } finally {
            labels.release(); stats.release(); cent.release()
        }
    }

    private fun trySplit(cfg: SplitCfg, binS: Mat, grayS: Mat, total: Double, ww: Int, wh: Int): Piece? {
        val edges = Mat()
        val cut = Mat.zeros(wh, ww, CvType.CV_8UC1)
        val notCut = Mat()
        val pieces = Mat()
        val region = Mat()
        val lines = Mat()
        try {
            Imgproc.Canny(grayS, edges, cfg.low, cfg.high)
            if (cfg.hough) {
                // Long straight edges strictly inside the blob, extended to reach its boundary.
                val er = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0))
                Imgproc.erode(binS, region, er)
                er.release()
                Core.bitwise_and(edges, region, edges)
                val shortSide = min(ww, wh).toDouble()
                Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 40, shortSide * SPLIT_MIN_LINE, 6.0)
                if (lines.rows() == 0) return null
                val ext = shortSide * 0.06
                for (i in 0 until lines.rows()) {
                    val l = lines.get(i, 0)
                    val dx = l[2] - l[0]
                    val dy = l[3] - l[1]
                    val len = hypot(dx, dy)
                    if (len < 1.0) continue
                    val ux = dx / len
                    val uy = dy / len
                    Imgproc.line(
                        cut,
                        Point(l[0] - ux * ext, l[1] - uy * ext),
                        Point(l[2] + ux * ext, l[3] + uy * ext),
                        Scalar(255.0), 3
                    )
                }
            } else {
                // Every edge (incl. faint shadows) inside/at the blob, thickened to close gaps.
                val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(cfg.dil.toDouble(), cfg.dil.toDouble()))
                Imgproc.dilate(binS, region, k)
                Core.bitwise_and(edges, region, edges)
                Imgproc.dilate(edges, cut, k)
                k.release()
            }
            Core.bitwise_not(cut, notCut)
            Core.bitwise_and(binS, notCut, pieces)
            return pickPiece(pieces, total, ww, wh)?.also { it.grow = if (cfg.hough) 5 else max(5, cfg.dil) }
        } finally {
            edges.release(); cut.release(); notCut.release(); pieces.release()
            region.release(); lines.release()
        }
    }

    /**
     * When sheets overlap, the mask is one irregular blob. Try several edge-based ways of
     * cutting it into sheets and keep the piece that is large, central and most
     * rectangular. Clean single sheets (rectangular blob) are left alone. Edits [bin]
     * in place; leaves it untouched if no sensible split is found.
     * Debug: `adb logcat -s PaperSegmenter` shows which strategy won.
     */
    private fun isolateTopSheet(bin: Mat, rgba: Mat) {
        val fullW = bin.cols()
        val fullH = bin.rows()
        val scale = min(1.0, SPLIT_WORK_SIZE / max(fullW, fullH).toDouble())
        val ww = max(32, (fullW * scale).toInt())
        val wh = max(32, (fullH * scale).toInt())

        val binS = Mat()
        val gray = Mat()
        val grayS = Mat()
        val up = Mat()
        var best: Piece? = null
        try {
            Imgproc.resize(bin, binS, Size(ww.toDouble(), wh.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            val rect0 = rectangularity(binS)
            // A blob that runs off the frame or fills most of it looks rectangular only
            // because the frame clips it (e.g. a pile of sheets) -> never treat it as clean.
            val areaFrac = Core.countNonZero(binS).toDouble() / (ww.toDouble() * wh)
            val sides = bordersTouched(binS)
            val clipped = sides >= 2 || areaFrac > 0.45
            if (rect0 >= SPLIT_RECT_GATE && !clipped) {
                Log.d(TAG, "split skipped: rectangularity=%.2f, clean sheet".format(rect0))
                return
            }
            Log.d(TAG, "split check: rectangularity=%.2f area=%.2f sides=%d".format(rect0, areaFrac, sides))

            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.resize(gray, grayS, Size(ww.toDouble(), wh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.GaussianBlur(grayS, grayS, Size(5.0, 5.0), 0.0)

            val total = Core.countNonZero(binS).toDouble()
            var bestName = ""
            for (cfg in splitConfigs) {
                val piece0 = trySplit(cfg, binS, grayS, total, ww, wh)
                Log.d(TAG, "  %s -> %s".format(cfg.name, piece0?.let { "q=%.3f".format(it.quality) } ?: "none"))
                val piece = piece0 ?: continue
                val cur = best
                if (cur == null || piece.quality > cur.quality) {
                    cur?.mask?.release()
                    best = piece
                    bestName = cfg.name
                } else {
                    piece.mask.release()
                }
            }

            val chosen = best
            if (chosen == null) {
                Log.d(TAG, "split: no strategy found a sheet (blob rectangularity=%.2f)".format(rect0))
                return
            }
            Log.d(TAG, "split: using %s (quality=%.3f, blob rectangularity=%.2f)".format(bestName, chosen.quality, rect0))

            // Give back the pixels eaten by the cut lines, but stay inside the blob.
            val grow = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(chosen.grow.toDouble(), chosen.grow.toDouble()))
            Imgproc.dilate(chosen.mask, chosen.mask, grow)
            grow.release()
            Imgproc.resize(chosen.mask, up, Size(fullW.toDouble(), fullH.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
            Core.bitwise_and(bin, up, bin)
        } finally {
            best?.mask?.release()
            binS.release(); gray.release(); grayS.release(); up.release()
        }
    }

    private fun hullOf(pts: Array<Point>): Array<Point> {
        val m = MatOfPoint(*pts)
        val idx = MatOfInt()
        Imgproc.convexHull(m, idx)
        val arr = m.toArray()
        val hull = idx.toArray().map { arr[it] }.toTypedArray()
        m.release(); idx.release()
        return hull
    }

    /** Hull -> best 4-gon (epsilon sweep, minAreaRect fallback). Unordered. */
    private fun quadFromHull(hull: Array<Point>): Array<Point> {
        val hull2f = MatOfPoint2f(*hull)
        try {
            val peri = Imgproc.arcLength(hull2f, true)
            for (eps in listOf(0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12)) {
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(hull2f, approx, eps * peri, true)
                val isQuad = approx.total() == 4L
                val pts = if (isQuad) approx.toArray() else null
                approx.release()
                if (pts != null) return pts
            }
            val rect = Imgproc.minAreaRect(hull2f)
            val box = arrayOfNulls<Point>(4)
            rect.points(box)
            return box.map { it ?: Point(0.0, 0.0) }.toTypedArray()
        } finally {
            hull2f.release()
        }
    }

    /**
     * Snap each quad side onto the real page edge: collect contour points lying near
     * that side, fit a robust line, then intersect neighbouring lines for the corners.
     * [quad] must be ordered TL, TR, BR, BL.
     */
    private fun refineQuad(quad: Array<Point>, contour: Array<Point>): Array<Point> {
        val diag = hypot(
            quad.maxOf { it.x } - quad.minOf { it.x },
            quad.maxOf { it.y } - quad.minOf { it.y }
        )
        val tol = diag * 0.025
        // line = [vx, vy, x0, y0]
        val lines = arrayOfNulls<DoubleArray>(4)

        for (i in 0 until 4) {
            val a = quad[i]
            val b = quad[(i + 1) % 4]
            val dx = b.x - a.x
            val dy = b.y - a.y
            val len = hypot(dx, dy)
            if (len < 1e-3) return quad
            val ux = dx / len
            val uy = dy / len

            val near = ArrayList<Point>()
            for (p in contour) {
                val rx = p.x - a.x
                val ry = p.y - a.y
                val t = (rx * ux + ry * uy) / len
                val d = abs(rx * uy - ry * ux)
                if (t in 0.1..0.9 && d < tol) near.add(p)
            }
            if (near.size < 8) {
                lines[i] = doubleArrayOf(ux, uy, a.x, a.y)
                continue
            }
            val pm = MatOfPoint2f(*near.toTypedArray())
            val l = Mat()
            Imgproc.fitLine(pm, l, Imgproc.DIST_HUBER, 0.0, 0.01, 0.01)
            val v = FloatArray(4)
            l.get(0, 0, v)
            pm.release(); l.release()
            lines[i] = doubleArrayOf(v[0].toDouble(), v[1].toDouble(), v[2].toDouble(), v[3].toDouble())
        }

        val out = Array(4) { quad[it] }
        for (i in 0 until 4) {
            // corner i = intersection of side (i-1) and side i
            val l1 = lines[(i + 3) % 4] ?: return quad
            val l2 = lines[i] ?: return quad
            val det = l1[0] * l2[1] - l1[1] * l2[0]
            if (abs(det) < 1e-6) continue // near parallel, keep original corner
            val t = ((l2[2] - l1[2]) * l2[1] - (l2[3] - l1[3]) * l2[0]) / det
            val p = Point(l1[2] + t * l1[0], l1[3] + t * l1[1])
            if (hypot(p.x - quad[i].x, p.y - quad[i].y) < diag * 0.05) out[i] = p
        }
        return out
    }

    /** Mask -> 4 refined corners (ordered TL, TR, BR, BL). */
    private fun quadFromMask(mask: Mat, rgba: Mat): Array<Point>? {
        val bin = Mat()
        try {
            val contour = largestContour(mask, bin, rgba) ?: return null
            val hull = hullOf(contour)
            if (hull.size < 3) return null

            val quad = DocumentDetector.orderCorners(quadFromHull(hull))

            // Don't force an L/U-shaped mask into a rectangle.
            val q2f = MatOfPoint2f(*quad)
            val quadArea = Imgproc.contourArea(q2f)
            q2f.release()
            if (quadArea > 0 && Core.countNonZero(bin).toDouble() / quadArea < 0.55) return null

            return refineQuad(quad, contour)
        } catch (t: Throwable) {
            return null
        } finally {
            bin.release()
        }
    }

    /**
     * Mask -> outline, polygon first: the real (possibly concave) shape of the page is
     * kept, simplified to at most MAX_POLY_POINTS well-spaced vertices. Only when a plain
     * 4-corner page matches that outline (IoU >= QUAD_IOU) is it snapped to a rectangle
     * with refined corners.
     */
    private fun polygonFromMask(mask: Mat, rgba: Mat): Array<Point>? {
        val bin = Mat()
        try {
            val contour = largestContour(mask, bin, rgba) ?: return null
            val hull = hullOf(contour)
            val quad = if (hull.size >= 3) DocumentDetector.orderCorners(quadFromHull(hull)) else null
            val poly = simplifyOutline(contour)

            if (poly == null || poly.size < 4) {
                return quad?.let { refineQuad(it, contour) }
            }
            if (quad != null && iou(bin.size(), quad, poly) >= QUAD_IOU) {
                return refineQuad(quad, contour)
            }
            return if (poly.size == 4) DocumentDetector.orderCorners(poly) else poly
        } catch (t: Throwable) {
            return null
        } finally {
            bin.release()
        }
    }

    /** Dense contour -> few, well-spaced vertices (no clusters), concavities preserved. */
    private fun simplifyOutline(contour: Array<Point>): Array<Point>? {
        val c2f = MatOfPoint2f(*contour)
        val approx = MatOfPoint2f()
        try {
            val peri = Imgproc.arcLength(c2f, true)
            var eps = 0.004
            do {
                Imgproc.approxPolyDP(c2f, approx, eps * peri, true)
                eps *= 1.35
            } while (approx.total() > MAX_POLY_POINTS && eps < 0.1)

            val minGap = 0.02 * peri
            val merged = ArrayList<Point>()
            for (p in approx.toArray()) {
                val last = merged.lastOrNull()
                if (last != null && hypot(p.x - last.x, p.y - last.y) < minGap) {
                    merged[merged.size - 1] = Point((p.x + last.x) / 2, (p.y + last.y) / 2)
                } else {
                    merged.add(p)
                }
            }
            if (merged.size > 2) {
                val f = merged.first()
                val l = merged.last()
                if (hypot(f.x - l.x, f.y - l.y) < minGap) {
                    merged[0] = Point((f.x + l.x) / 2, (f.y + l.y) / 2)
                    merged.removeAt(merged.size - 1)
                }
            }
            return if (merged.size >= 3) merged.toTypedArray() else null
        } finally {
            c2f.release(); approx.release()
        }
    }

    /** Intersection-over-union of two filled polygons, rasterised at [size]. */
    private fun iou(size: Size, a: Array<Point>, b: Array<Point>): Double {
        val ma = Mat.zeros(size, CvType.CV_8UC1)
        val mb = Mat.zeros(size, CvType.CV_8UC1)
        Imgproc.fillPoly(ma, listOf(MatOfPoint(*a)), Scalar(255.0))
        Imgproc.fillPoly(mb, listOf(MatOfPoint(*b)), Scalar(255.0))
        val inter = Mat()
        val union = Mat()
        Core.bitwise_and(ma, mb, inter)
        Core.bitwise_or(ma, mb, union)
        val i = Core.countNonZero(inter).toDouble()
        val u = Core.countNonZero(union).toDouble()
        ma.release(); mb.release(); inter.release(); union.release()
        return if (u > 0) i / u else 0.0
    }
}