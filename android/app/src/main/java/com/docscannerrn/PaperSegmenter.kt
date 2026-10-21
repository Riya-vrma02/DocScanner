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
import kotlin.math.atan2
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
 *
 * Changes in this revision (search for "CHANGED" to find each one):
 *  A. The outline is a real polygon: it is no longer snapped to a rectangle
 *     (FORCE_POLYGON), keeps up to 32 vertices and is simplified less aggressively.
 *  B. Pile / top-sheet mode has hysteresis, so a pile sitting near the solidity gate
 *     does not flip between "top sheet" and "whole pile" every frame.
 *  C. The top-sheet quad search prefers last frame's quad, and accepts a sheet with
 *     one weakly-supported side (the side lying against the pile is usually the
 *     faint one).
 *  D. The top-sheet quad is dilated slightly before it is ANDed with the mask, so a
 *     curved/torn top edge survives instead of being cut to a straight line.
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

        /**
         * Colour gate (see suppressNonPaper). Max weighted Lab distance from the page colour; lower
         * = tighter outline, higher = more tolerant of shading and colour gradients on the page.
         */
        private const val COLOR_GATE_T = 22f

        /** Lightness counts this much compared with chroma, so shadows don't read as "other paper". */
        private const val COLOR_GATE_L_WEIGHT = 0.5f

        /** A pixel must have at least this fraction of the mask's 99th percentile to be kept at all. */
        private const val GATE_CONF_FRACTION = 0.35

        /** If the gate would keep less than this fraction of the model's paper, it is skipped. */
        private const val GATE_MIN_KEPT = 0.4

        /**
         * CHANGED (A): was 0.93. A 4-corner quad is accepted in place of the polygon only if
         * it overlaps the page outline at least this well (IoU). Only used when
         * [FORCE_POLYGON] is false.
         */
        private const val QUAD_IOU = 0.985

        /**
         * CHANGED (A): true = never snap the outline to a rectangle; always return the real
         * (possibly curved / torn / concave) outline. A clean flat page still comes back as
         * 4 refined corners because the simplified outline has 4 vertices. Set to false to
         * get the old "snap to rectangle when it fits well" behaviour back.
         */
        private const val FORCE_POLYGON = true

        /**
         * The top-sheet / pile logic (isolateTopSheet) treats a low-solidity blob as a pile of
         * sheets and replaces the mask with the best-fitting QUAD. A curved, crumpled or torn
         * single page ALSO has low solidity, so with this on it gets forced to a quadrilateral.
         * false = the outline is the raw mask contour (true polygon). Turn it back on once
         * single-page polygons look right, to see what it does to piles.
         */
        private const val ENABLE_TOP_SHEET_LOGIC = true

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

        /**
         * Below this solidity (paper area / hull area) the frame is reported as
         * overlapping sheets. A single sheet is convex and sits around 0.95+; sheets
         * at angles leave background wedges inside the hull and fall well below.
         * Lower it if piles are missed, raise it if lone pages are flagged.
         */
        private const val SOLIDITY_GATE = 0.88

        /**
         * CHANGED (B): once in pile mode, solidity must rise this far above [SOLIDITY_GATE]
         * before the frame is treated as a clean single sheet again.
         */
        private const val PILE_HYSTERESIS = 0.04

        /**
         * Lines kept per orientation family. C(n,2)^2 candidates, so keep it small.
         * PAPER-ON-PAPER: was 5. A sheet lying on others has its border close to the borders of the
         * sheets below it, so a few more lines must survive for the right pair to be among them.
         */
        private const val QUAD_LINES_PER_FAMILY = 6

        /**
         * PAPER-ON-PAPER: two lines closer than this (fraction of the short side) with the same angle
         * count as one edge. Was 6%, which merged the top sheet's border with the border of the sheet
         * just under it whenever they were offset by a few millimetres, so the top sheet could never
         * be formed.
         */
        private const val LINE_DEDUPE = 0.025

        /** PAPER-ON-PAPER: black top-hat value (text-free paper minus image) above which a pixel is ink. */
        private const val INK_THR = 28

        /** PAPER-ON-PAPER: band (fraction of the short side) checked for ink beside a candidate border. */
        private const val INK_BAND = 0.025

        /**
         * PAPER-ON-PAPER: the darkest the line ON a sheet border may be (top-hat, grey levels). A border
         * between two white sheets is at most a faint shadow line; anything darker is printed ink.
         */
        private const val BORDER_LINE_MAX = 60

        /** PAPER-ON-PAPER: paper on both sides of a border must be at least this bright (x paper level). */
        private const val PAPER_BRIGHT = 0.80

        /** PAPER-ON-PAPER: for a brightness step both sides must be at least this bright (not a photo). */
        private const val STEP_BRIGHT = 0.60

        /**
         * PAPER-ON-PAPER: a nested quad whose two OPPOSITE sides each have only a narrow (< this fraction
         * of the short side), similar-width strip of paper beyond them is a printed frame on one page,
         * not a sheet lying on another.
         */
        private const val FRAME_MARGIN = 0.08

        /** A candidate quad's interior must be at least this fraction paper. */
        private const val QUAD_MIN_FILL = 0.90

        /**
         * CHANGED (C): the second-weakest side of a candidate quad must lie on real edges by
         * at least this much (the weakest side may be weaker, see below).
         */
        private const val QUAD_MIN_SIDE_SUPPORT = 0.45

        /** CHANGED (C): the single weakest side of a candidate quad must still reach this. */
        private const val QUAD_WEAKEST_SIDE_SUPPORT = 0.20

        /**
         * A quad side that has paper on BOTH sides (an edge inside the paper blob: the border of the
         * top sheet over a lower sheet, or just a printed line) only counts if the paper brightness
         * (text removed) steps by at least INTERIOR_STEP grey levels across it for at least
         * INTERIOR_STEP_FRAC of its length. A real sheet border casts a shadow / sits at a different
         * tilt, so it has a step; a printed line inside one sheet does not. Sides on the outer
         * boundary of the blob (paper against table) are exempt.
         */
        private const val INTERIOR_STEP = 5
        private const val INTERIOR_STEP_FRAC = 0.35

        /** Distance (fraction of the short side) at which brightness is sampled on each side of a quad side. */
        private const val INTERIOR_OFFSET = 0.02

        /**
         * CHANGED (D): how far (fraction of the short image side) the top-sheet quad is grown
         * before it is ANDed with the mask, so curved/torn edges are not cut flat.
         */
        private const val TOP_QUAD_GROW = 0.01

        /**
         * The model's mask has rounded corners, so simplification turns every corner into a short
         * chamfer edge (two vertices) and corners look cut. A chamfer is restored to a sharp
         * corner when it is shorter than this fraction of the outline's perimeter and both
         * neighbouring edges are at least twice as long.
         */
        private const val CHAMFER_MAX_FRACTION = 0.07

        /** CHANGED (A): was 16. Max points returned when the page is not a clean quad. */
        private const val MAX_POLY_POINTS = 32

        /** Weight of the newest frame when averaging live masks (lower = steadier, laggier). */
        private const val TEMPORAL_ALPHA = 0.7
    }

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var unavailable = false
    private var accMask: Mat? = null
    @Volatile private var maskThr = MASK_THRESHOLD

    /** CHANGED (B): current pile/top-sheet mode, with hysteresis. */
    private var pileMode = false

    /** CHANGED (C): last frame's top-sheet quad, normalised to 0..1 of the working size. */
    private var prevTopQuadN: Array<Point>? = null

    /** Short human-readable summary of the last outline decision (shown by the on-screen HUD). */
    @Volatile
    var lastDebug: String = ""
        private set

    /** Forget the averaged live mask (call when the page is lost). */
    fun resetSmoothing() {
        accMask?.release()
        accMask = null
        pileMode = false
        prevTopQuadN = null
    }

    @Volatile
    var lastError: String? = null
        private set

    /** Exception text from the last fit step, so a crash isn't reported as a vague "no_shape". */
    @Volatile private var fitError: String? = null

    /**
     * What the most recent mask looked like, sheet-wise. Read this right after a
     * findPageQuad/findPagePolygon call.
     *
     * [overlapping] means the model returned ONE blob that had to be cut apart,
     * i.e. sheets are stacked/overlapping and only the top one was kept — so the
     * others are silently excluded from the scan and the user should be told.
     *
     * [separateSheets] counts clearly detached paper blobs (pages lying side by
     * side). That's not overlap, but it does mean more than one page is in frame.
     */
    class SheetInfo(
        val overlapping: Boolean,
        val separateSheets: Int,
        /** Hull vs min-area rect (1.0 = hull is rectangular). Diagnostic only. */
        val blobRectangularity: Double,
        /** Paper area / hull area. This is what drives [overlapping]. */
        val blobSolidity: Double,
        /** True when a cut was actually applied to isolate the top sheet. */
        val splitApplied: Boolean
    )

    @Volatile
    var lastSheetInfo: SheetInfo = SheetInfo(false, 0, 1.0, 1.0, false)
        private set

    // Set by isolateTopSheet() while a frame is being processed.
    private var frameSplitApplied = false
    private var frameRect = 1.0

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
     * Page outline (4..MAX_POLY_POINTS points). A clean flat page comes back as exactly 4
     * ordered corners; a curved or torn page comes back as a polygon that follows its edge.
     */
    fun findPagePolygon(rgba: Mat, live: Boolean = false): Array<Point>? =
        withMask(rgba, live) { m, img -> polygonFromMask(m, img) }

    private fun withMask(rgba: Mat, temporal: Boolean = false, fit: (Mat, Mat) -> Array<Point>?): Array<Point>? {
        val s = getSession() ?: return null
        val e = env ?: return null
        lastDebug = "" // cleared per call so a stale line can't be mistaken for this frame

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

            fitError = null
            val result = fit(maskFull, rgba)
            lastError = if (result == null) (fitError ?: "no_shape (mask empty or too irregular)") else null
            return result
        } catch (t: Throwable) {
            lastError = "infer ${t::class.java.simpleName}: ${t.message}"
            Log.w(TAG, "withMask failed: $lastError")
            return null
        } finally {
            rgb.release(); resized.release(); maskSmall.release(); maskFull.release()
        }
    }

    // ------------------------------------------------------------------ mask cleanup

    /**
     * Colour gate: removes mask pixels that are not the same colour as the page.
     *
     * Works for white AND coloured paper. A coloured page gets a weaker, leakier mask from the model
     * (it bleeds onto the table, other sheets, the desk), which made the outline too big. The page's
     * own colour is the most reliable cue, so:
     *
     *  1. take the model's most confident region as the page core (relative to THIS frame's mask, so a
     *     weak mask on coloured paper still has a core) and read its median Lab colour;
     *  2. keep only mask pixels whose colour is close to that (lightness is weighted down, so shadows
     *     and shading don't count as "different paper");
     *  3. keep just the connected region that contains the core, and fill its holes, so a printed
     *     picture, a colour band or a stamp INSIDE the page is not cut out of it.
     *
     * Nothing is ever added to the mask, only removed, and the gate backs off if it would throw away
     * most of what the model found (e.g. the core sits on one colour of a multi-colour page).
     * Tuned on tinted pages with simulated leaks: IoU with the true page 0.64 -> 0.94.
     */
    private fun suppressNonPaper(rgb320: Mat, mask: Mat) {
        val total = INPUT_SIZE * INPUT_SIZE
        val lab = Mat()
        val blur = Mat()
        val tmp = Mat()
        val core = Mat()
        val f = Mat()
        val dist = Mat()
        val near32 = Mat()
        val near = Mat()
        val conf = Mat()
        val keep = Mat()
        val labels = Mat()
        val comp = Mat(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
        val filled = Mat.zeros(INPUT_SIZE, INPUT_SIZE, CvType.CV_8UC1)
        val notFilled = Mat()
        val fg = Mat()
        val kept = Mat()
        val chans = ArrayList<Mat>()
        try {
            // --- this frame's own idea of "confident paper"
            val otsu = Imgproc.threshold(mask, tmp, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
            val maskBytes = ByteArray(total)
            mask.get(0, 0, maskBytes)
            val p99 = percentile(maskBytes, 0.99)
            if (p99 < 20.0) return // model saw nothing

            val coreThr = max(otsu, 0.75 * p99)
            Imgproc.threshold(mask, core, coreThr - 0.5, 255.0, Imgproc.THRESH_BINARY)
            val er = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
            Imgproc.erode(core, core, er)
            er.release()
            if (Core.countNonZero(core) < total * 0.01) return

            // --- median Lab colour of the core
            Imgproc.cvtColor(rgb320, lab, Imgproc.COLOR_RGB2Lab)
            Imgproc.GaussianBlur(lab, blur, Size(9.0, 9.0), 0.0)
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

            // --- weighted Lab distance to the page colour (lightness counts less than chroma)
            blur.convertTo(f, CvType.CV_32FC3)
            Core.subtract(f, Scalar(med[0], med[1], med[2]), f)
            Core.multiply(f, f, f)
            Core.split(f, chans)
            Core.addWeighted(
                chans[0], (COLOR_GATE_L_WEIGHT * COLOR_GATE_L_WEIGHT).toDouble(),
                chans[1], 1.0, 0.0, dist
            )
            Core.add(dist, chans[2], dist)
            Core.sqrt(dist, dist)

            // near = same colour as the page; conf = the model thinks it is at least vaguely paper
            Imgproc.threshold(dist, near32, COLOR_GATE_T.toDouble(), 255.0, Imgproc.THRESH_BINARY_INV)
            near32.convertTo(near, CvType.CV_8U)
            Imgproc.threshold(mask, conf, GATE_CONF_FRACTION * p99, 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_and(near, conf, keep)

            val open = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0))
            Imgproc.morphologyEx(keep, keep, Imgproc.MORPH_OPEN, open)
            open.release()

            // --- keep only the region(s) that contain the core
            val nLabels = Imgproc.connectedComponents(keep, labels, 8, CvType.CV_32S)
            if (nLabels <= 1) return
            val ids = IntArray(total)
            labels.get(0, 0, ids)
            val wanted = BooleanArray(nLabels)
            for (i in 0 until total) if (cb[i].toInt() != 0) wanted[ids[i]] = true
            wanted[0] = false
            val compBytes = ByteArray(total)
            for (i in 0 until total) if (wanted[ids[i]]) compBytes[i] = 255.toByte()
            comp.put(0, 0, compBytes)

            // --- fill holes so colour inside the page (pictures, bands, stamps) survives
            val contours = mutableListOf<MatOfPoint>()
            val hier = Mat()
            val compCopy = comp.clone()
            Imgproc.findContours(compCopy, contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            compCopy.release(); hier.release()
            Imgproc.drawContours(filled, contours, -1, Scalar(255.0), Imgproc.FILLED)
            contours.forEach { it.release() }

            // --- back off if the gate would throw away most of what the model found
            Imgproc.threshold(mask, fg, otsu - 0.5, 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_and(fg, filled, kept)
            val fgPx = Core.countNonZero(fg)
            if (fgPx == 0 || Core.countNonZero(kept) < GATE_MIN_KEPT * fgPx) {
                Log.d(TAG, "colour gate skipped: would keep only %d of %d px".format(Core.countNonZero(kept), fgPx))
                return
            }

            Core.bitwise_not(filled, notFilled)
            mask.setTo(Scalar(0.0), notFilled)
        } finally {
            lab.release(); blur.release(); tmp.release(); core.release(); f.release(); dist.release()
            near32.release(); near.release(); conf.release(); keep.release(); labels.release()
            comp.release(); filled.release(); notFilled.release(); fg.release(); kept.release()
            chans.forEach { it.release() }
        }
    }

    /** [q]-quantile (0..1) of an 8-bit mask given as raw bytes. */
    private fun percentile(bytes: ByteArray, q: Double): Double {
        val hist = IntArray(256)
        for (b in bytes) hist[b.toInt() and 0xFF]++
        val target = (bytes.size * q).toInt()
        var acc = 0
        for (v in 0..255) {
            acc += hist[v]
            if (acc > target) return v.toDouble()
        }
        return 255.0
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

        val imgArea = (bin.rows() * bin.cols()).toDouble()

        // Count clearly detached sheets BEFORE splitting: pages lying side by side
        // show up here as several separate blobs. (Overlapping sheets, by contrast,
        // arrive as a single blob and are counted by isolateTopSheet instead.)
        val separate = countSignificantBlobs(bin, imgArea)

        // Measured BEFORE the split: afterwards only the top sheet is left, which is
        // convex, so solidity would read high no matter how big the pile was.
        val solid = solidity(bin)

        // Several overlapping sheets come out of the model as one blob: keep only the top one.
        frameSplitApplied = false
        frameRect = 1.0
        isolateTopSheet(bin, rgba)

        // Two ways a stack of sheets shows up:
        //  - the cut succeeded, so we know we dropped something (splitApplied), or
        //  - the blob is one mass whose hull is full of background gaps, i.e. low
        //    solidity, which is what a pile of angled sheets looks like even when no
        //    cut could divide it.
        // CHANGED (B): uses the hysteresis-filtered pileMode instead of the raw gate, so the
        // overlap flag does not flicker for a pile hovering around SOLIDITY_GATE.
        lastSheetInfo = SheetInfo(
            overlapping = frameSplitApplied || pileMode,
            separateSheets = separate,
            blobRectangularity = frameRect,
            blobSolidity = solid,
            splitApplied = frameSplitApplied
        )

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        // CHAIN_APPROX_NONE: dense points along straight edges are needed for line fitting.
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        hierarchy.release()

        val biggest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return null
        if (Imgproc.contourArea(biggest) < 0.03 * imgArea) return null
        return biggest.toArray()
    }

    /** Number of separate blobs each covering >= 3% of the frame. */
    private fun countSignificantBlobs(bin: Mat, imgArea: Double): Int {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        var n = 0
        for (c in contours) {
            if (Imgproc.contourArea(c) >= 0.03 * imgArea) n++
            c.release()
        }
        return n
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

    private class QuadResult(val quad: Array<Point>, val score: Double)

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

    /**
     * Paper area / convex-hull area of the largest blob.
     *
     * This is the one that actually sees overlap. [rectangularity] compares the hull
     * to its min-area rect, and the hull of several scattered sheets is itself a
     * near-perfect rectangle, so it reads ~1.0 for a pile. Solidity instead asks how
     * much of that hull is really covered in paper: a single sheet is convex and
     * fills its own hull (~0.95+), while sheets lying at angles leave big wedges of
     * background inside the hull and drop well below.
     */
    private fun solidity(maskSmall: Mat): Double {
        val tmp = maskSmall.clone()
        val contours = mutableListOf<MatOfPoint>()
        val hier = Mat()
        try {
            Imgproc.findContours(tmp, contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val big = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return 1.0
            val area = Imgproc.contourArea(big)
            val hull = hullOf(big.toArray())
            if (hull.size < 3) return 1.0
            val h2 = MatOfPoint2f(*hull)
            val hullArea = Imgproc.contourArea(h2)
            h2.release()
            return if (hullArea > 0) (area / hullArea).coerceIn(0.0, 1.0) else 1.0
        } finally {
            tmp.release(); hier.release(); contours.forEach { it.release() }
        }
    }

    /** Segment orientation in degrees, folded to [0,180). */
    private fun segAngle(s: DoubleArray): Double {
        var a = Math.toDegrees(atan2(s[3] - s[1], s[2] - s[0]))
        if (a < 0) a += 180.0
        if (a >= 180.0) a -= 180.0
        return a
    }

    /** Smallest angle between two orientations, 0..90. */
    private fun angleDiff(a: Double, b: Double): Double {
        var d = abs(a - b) % 180.0
        if (d > 90.0) d = 180.0 - d
        return d
    }

    /** Perpendicular distance from a point to the infinite line through [s]. */
    private fun pointLineDist(px: Double, py: Double, s: DoubleArray): Double {
        val dx = s[2] - s[0]
        val dy = s[3] - s[1]
        val len = hypot(dx, dy)
        if (len < 1e-6) return hypot(px - s[0], py - s[1])
        return abs(dy * (px - s[0]) - dx * (py - s[1])) / len
    }

    /** Intersection of the infinite lines through two segments, or null if parallel. */
    private fun intersect(a: DoubleArray, b: DoubleArray): Point? {
        val x1 = a[0]; val y1 = a[1]; val x2 = a[2]; val y2 = a[3]
        val x3 = b[0]; val y3 = b[1]; val x4 = b[2]; val y4 = b[3]
        val d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (abs(d) < 1e-6) return null
        val c1 = x1 * y2 - y1 * x2
        val c2 = x3 * y4 - y3 * x4
        return Point(
            (c1 * (x3 - x4) - (x1 - x2) * c2) / d,
            (c1 * (y3 - y4) - (y1 - y2) * c2) / d
        )
    }

    /** Bilinear point inside a quad ordered as a cycle. */
    private fun quadPoint(c: Array<Point>, u: Double, v: Double): Point {
        val tx = c[0].x + (c[1].x - c[0].x) * u
        val ty = c[0].y + (c[1].y - c[0].y) * u
        val bx = c[3].x + (c[2].x - c[3].x) * u
        val by = c[3].y + (c[2].y - c[3].y) * u
        return Point(tx + (bx - tx) * v, ty + (by - ty) * v)
    }

    private fun isConvex(c: Array<Point>): Boolean {
        var sign = 0
        for (i in 0 until 4) {
            val a = c[i]
            val b = c[(i + 1) % 4]
            val d = c[(i + 2) % 4]
            val cross = (b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x)
            if (abs(cross) < 1e-9) continue
            val s = if (cross > 0) 1 else -1
            if (sign == 0) sign = s else if (s != sign) return false
        }
        return sign != 0
    }

    /**
     * Positive search for the sheet lying on top of a pile, in [binS] coordinates.
     *
     * Cutting the blob apart (see [trySplit]) works only when a sheet border shows up
     * as one long line that happens to divide the blob cleanly. On a stack of angled
     * sheets the borders are white-on-white and the cuts just fragment it, which is
     * why nothing is returned and the whole pile survives as one contour.
     *
     * This looks for the sheet instead of against it. Being on top is exactly the
     * property of having its borders unoccluded, so a quadrilateral whose sides sit on
     * detected edges and whose interior is solid paper is the top sheet. Candidates
     * come from pairs of lines in two perpendicular families; the best-scoring one wins.
     *
     * CHANGED (C): one side may be weakly supported (the side against the pile is often
     * the faint one), and a candidate overlapping last frame's winner scores higher so
     * the choice stays stable from frame to frame.
     *
     * Returns null when no candidate is convincing, leaving the caller to fall back.
     */
    private fun findTopSheetQuadPass(
        binS: Mat, grayS: Mat, lumS: Mat, tophatS: Mat, ww: Int, wh: Int,
        cannyLo: Double, cannyHi: Double, houghThresh: Int
    ): QuadResult? {
        val edges = Mat()
        val region = Mat()
        val edgesDil = Mat()
        val lines = Mat()
        try {
            Imgproc.Canny(grayS, edges, cannyLo, cannyHi)
            // A patterned tablecloth produces more lines than the page does, so keep
            // only edges on or just inside the paper blob.
            val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
            Imgproc.dilate(binS, region, k)
            Core.bitwise_and(edges, region, edges)
            // Tolerate a couple of px of misalignment when testing side support.
            Imgproc.dilate(edges, edgesDil, k)
            k.release()

            val shortSide = min(ww, wh).toDouble()
            val minLen = shortSide * 0.18
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, houghThresh, minLen, 12.0)
            if (lines.rows() < 4) {
                Log.d(TAG, "  quad: only %d lines".format(lines.rows()))
                return null
            }

            val segs = ArrayList<DoubleArray>()
            for (i in 0 until lines.rows()) {
                val l = lines.get(i, 0)
                if (hypot(l[2] - l[0], l[3] - l[1]) >= minLen) segs.add(l)
            }
            if (segs.size < 4) return null
            segs.sortByDescending { hypot(it[2] - it[0], it[3] - it[1]) }

            // Split into two perpendicular families around the dominant orientation,
            // dropping lines that duplicate one already kept (same edge found twice).
            val base = segAngle(segs[0])
            val famA = ArrayList<DoubleArray>()
            val famB = ArrayList<DoubleArray>()
            val dedupeDist = shortSide * LINE_DEDUPE
            for (s in segs) {
                val off = angleDiff(segAngle(s), base)
                val fam: ArrayList<DoubleArray>? =
                    if (off <= 30.0) famA else if (off >= 60.0) famB else null
                if (fam == null || fam.size >= QUAD_LINES_PER_FAMILY) continue
                val dup = fam.any {
                    angleDiff(segAngle(s), segAngle(it)) < 8.0 &&
                        pointLineDist(s[0], s[1], it) < dedupeDist
                }
                if (!dup) fam.add(s)
            }
            if (famA.size < 2 || famB.size < 2) {
                Log.d(TAG, "  quad: families %d/%d".format(famA.size, famB.size))
                return null
            }

            // Pixel lookups dominate the scoring, so read both masks once into arrays
            // instead of touching the Mats per candidate.
            val n = ww * wh
            val eBuf = ByteArray(n)
            val bBuf = ByteArray(n)
            edgesDil.get(0, 0, eBuf)
            binS.get(0, 0, bBuf)
            val lBuf = ByteArray(n)
            lumS.get(0, 0, lBuf)

            fun paperAt(x: Int, y: Int): Boolean =
                x in 0 until ww && y in 0 until wh && bBuf[y * ww + x] != 0.toByte()

            fun edgeAt(x: Int, y: Int): Boolean =
                x in 0 until ww && y in 0 until wh && eBuf[y * ww + x] != 0.toByte()

            fun lumAt(x: Int, y: Int): Int =
                if (x in 0 until ww && y in 0 until wh) (lBuf[y * ww + x].toInt() and 0xFF) else -1

            // ---------------- PAPER-ON-PAPER evidence ----------------
            val tBuf = ByteArray(n)
            tophatS.get(0, 0, tBuf)

            fun tophatAt(x: Int, y: Int): Int =
                (tBuf[y.coerceIn(0, wh - 1) * ww + x.coerceIn(0, ww - 1)].toInt() and 0xFF)

            // Brightness of the paper itself: 90th percentile of the text-free image over the blob.
            val paperLevel: Double = run {
                val hist = IntArray(256)
                var cnt = 0
                for (i in 0 until n) if (bBuf[i] != 0.toByte()) { hist[lBuf[i].toInt() and 0xFF]++; cnt++ }
                var acc = 0
                var lvl = 255
                for (v in 0..255) { acc += hist[v]; if (acc >= 0.9 * cnt) { lvl = v; break } }
                if (cnt == 0) 255.0 else lvl.toDouble()
            }
            val band = max(5, (shortSide * INK_BAND).toInt())
            val off0 = max(4.0, shortSide * INTERIOR_OFFSET)

            /** Ink pixels along the normal at the given offsets from (px, py). */
            fun inkNear(px: Double, py: Double, nx: Double, ny: Double, from: Int, to: Int): Int {
                var h = 0
                for (d in from..to) {
                    val xx = (px + nx * d).toInt()
                    val yy = (py + ny * d).toInt()
                    if (xx in 0 until ww && yy in 0 until wh && (tBuf[yy * ww + xx].toInt() and 0xFF) > INK_THR) h++
                }
                return h
            }

            /**
             * Fraction of segment a->b (outward normal nx,ny) that looks like the border of one sheet
             * lying on another: an edge, text-free paper-bright paper on both sides, at most a faint
             * line on it (shadow, not ink) and blank paper on the INSIDE (the top sheet's own margin).
             *
             * [insideOnly]: the outside may carry the lower sheet's text, because the top sheet cuts
             * it off. The edge must then also be present where BOTH sides are blank: a real border is
             * continuous, while the edge of a printed text block exists only where letters are.
             */
            fun edgeEvidence(ax: Double, ay: Double, bx: Double, by: Double,
                             nx: Double, ny: Double, insideOnly: Boolean): Double {
                val dx = bx - ax
                val dy = by - ay
                val len = hypot(dx, dy)
                if (len < 1.0) return 0.0
                val steps = max(8, (len / 2.0).toInt())
                var good = 0; var cnt = 0; var gap = 0; var gapEdge = 0
                for (t in 0..steps) {
                    val f = t.toDouble() / steps
                    val px = ax + dx * f
                    val py = ay + dy * f
                    val lo = lumAt((px + nx * off0).toInt(), (py + ny * off0).toInt())
                    val li = lumAt((px - nx * off0).toInt(), (py - ny * off0).toInt())
                    if (lo < 0 || li < 0) continue
                    cnt++
                    val e = edgeAt(px.toInt(), py.toInt())
                    var insideMin = 255
                    for (d in -band..-3) {
                        val v = lumAt((px + nx * d).toInt(), (py + ny * d).toInt())
                        if (v < insideMin) insideMin = v
                    }
                    val inBlank = inkNear(px, py, nx, ny, -band, -3) <= 1 && insideMin > PAPER_BRIGHT * paperLevel
                    val outBlank = inkNear(px, py, nx, ny, 3, band) <= 1
                    if (inBlank && outBlank) {
                        gap++
                        if (e) gapEdge++
                    }
                    if (!e || !inBlank) continue
                    if (!insideOnly && !outBlank) continue
                    if (lo < PAPER_BRIGHT * paperLevel || li < PAPER_BRIGHT * paperLevel) continue
                    var on = 0
                    for (d in -2..2) on = max(on, tophatAt((px + nx * d).toInt(), (py + ny * d).toInt()))
                    if (on > BORDER_LINE_MAX) continue
                    good++
                }
                if (cnt == 0) return 0.0
                if (insideOnly && (gap < 0.15 * cnt || gapEdge < 0.6 * gap)) return 0.0
                return good.toDouble() / cnt
            }

            // Per detected line: does it look like a sheet border? Used to reject a quad that has a
            // full-length sheet border running through its middle (it would span two sheets).
            val segEv = HashMap<DoubleArray, Double>()
            for (sg in famA + famB) {
                val dx = sg[2] - sg[0]
                val dy = sg[3] - sg[1]
                val len = hypot(dx, dy)
                if (len < 1.0) continue
                segEv[sg] = edgeEvidence(sg[0], sg[1], sg[2], sg[3], dy / len, -dx / len, false)
            }
            var bestNestedQuad: Array<Point>? = null
            var bestNestedScore = 0.0

            val frameArea = ww.toDouble() * wh
            val halfDiag = 0.5 * hypot(ww.toDouble(), wh.toDouble())
            val workSize = Size(ww.toDouble(), wh.toDouble())

            // CHANGED (C): last frame's winner, scaled into this working size.
            val priorQuad: Array<Point>? = prevTopQuadN?.let { pn ->
                Array(4) { Point(pn[it].x * ww, pn[it].y * wh) }
            }

            var bestQuad: Array<Point>? = null
            var bestScore = 0.0
            var tried = 0

            for (i in 0 until famA.size - 1) for (j in i + 1 until famA.size) {
                for (p in 0 until famB.size - 1) for (q in p + 1 until famB.size) {
                    // Cycle A_i -> B_q -> A_j -> B_p gives a proper quad ordering.
                    val c0 = intersect(famA[i], famB[p]) ?: continue
                    val c1 = intersect(famA[i], famB[q]) ?: continue
                    val c2 = intersect(famA[j], famB[q]) ?: continue
                    val c3 = intersect(famA[j], famB[p]) ?: continue
                    val quad = arrayOf(c0, c1, c2, c3)

                    val margin = shortSide * 0.08
                    if (quad.any {
                            it.x < -margin || it.y < -margin ||
                                it.x > ww + margin || it.y > wh + margin
                        }
                    ) continue
                    if (!isConvex(quad)) continue

                    val sideLens = DoubleArray(4) {
                        hypot(
                            quad[(it + 1) % 4].x - quad[it].x,
                            quad[(it + 1) % 4].y - quad[it].y
                        )
                    }
                    if (sideLens.any { it < shortSide * 0.12 }) continue
                    val longest = sideLens.maxOrNull() ?: continue
                    val shortest = sideLens.minOrNull() ?: continue
                    if (longest / shortest > 6.0) continue

                    val m = MatOfPoint2f(*quad)
                    val area = abs(Imgproc.contourArea(m))
                    m.release()
                    val areaFrac = area / frameArea
                    if (areaFrac < 0.08 || areaFrac > 0.90) continue

                    tried++

                    // Interior fill: how much of the quad is really paper. Sampled just
                    // inside the border so edge pixels don't bias it.
                    var inside = 0
                    var paper = 0
                    var u = 0.08
                    while (u <= 0.92) {
                        var v = 0.08
                        while (v <= 0.92) {
                            val pt = quadPoint(quad, u, v)
                            inside++
                            if (paperAt(pt.x.toInt(), pt.y.toInt())) paper++
                            v += 0.084
                        }
                        u += 0.084
                    }
                    if (inside == 0) continue
                    val fill = paper.toDouble() / inside
                    if (fill < QUAD_MIN_FILL) continue

                    // CHANGED (C): side support per side. A quad spanning two sheets has
                    // several unsupported sides, but the true top sheet may have ONE faint
                    // side (the one lying against the pile), so only the two strongest
                    // groups are held to the full threshold.
                    val sups = DoubleArray(4)
                    val qcx = quad.sumOf { it.x } / 4.0
                    val qcy = quad.sumOf { it.y } / 4.0
                    val off = max(4.0, shortSide * INTERIOR_OFFSET)
                    var interiorOk = true
                    var interiorSides = 0
                    for (sIdx in 0 until 4) {
                        val a = quad[sIdx]
                        val b = quad[(sIdx + 1) % 4]
                        val dx = b.x - a.x
                        val dy = b.y - a.y
                        val len = sideLens[sIdx]
                        // Outward normal (pointing away from the quad's centre).
                        var nx = dy / len
                        var ny = -dx / len
                        if (nx * ((a.x + b.x) / 2 - qcx) + ny * ((a.y + b.y) / 2 - qcy) < 0) {
                            nx = -nx
                            ny = -ny
                        }

                        val steps = max(8, (len / 2.0).toInt())
                        var onEdge = 0
                        var outsidePaper = 0
                        var lumPairs = 0
                        var stepHits = 0
                        val total = steps + 1
                        for (t in 0..steps) {
                            val f = t.toDouble() / steps
                            val px = a.x + dx * f
                            val py = a.y + dy * f
                            if (edgeAt(px.toInt(), py.toInt())) onEdge++
                            val ox = (px + nx * off).toInt()
                            val oy = (py + ny * off).toInt()
                            if (paperAt(ox, oy)) outsidePaper++
                            val ix = (px - nx * off).toInt()
                            val iy = (py - ny * off).toInt()
                            val lo = lumAt(ox, oy)
                            val li = lumAt(ix, iy)
                            // PAPER-ON-PAPER: measure the step only where neither side has ink nearby (text
                            // leaves the text-free image a few levels darker), and only between two bright
                            // papers (a printed photo also makes a step).
                            val cleanPt = inkNear(ox.toDouble(), oy.toDouble(), nx, ny, -3, 3) == 0 &&
                                inkNear(ix.toDouble(), iy.toDouble(), nx, ny, -3, 3) == 0
                            if (lo >= 0 && li >= 0 && cleanPt) {
                                lumPairs++
                                if (abs(lo - li) >= INTERIOR_STEP && min(lo, li) > STEP_BRIGHT * paperLevel) stepHits++
                            }
                        }
                        var sup = onEdge.toDouble() / total
                        // Paper continues beyond this side => it is an edge INSIDE the paper blob. It
                        // must show a brightness step (shadow / different tilt), otherwise it is just a
                        // printed line or fold inside one sheet and the quad is not a sheet.
                        if (outsidePaper.toDouble() / total > 0.4) {
                            interiorSides++
                            var stepFrac = if (lumPairs > 0 && lumPairs >= 0.2 * total) stepHits.toDouble() / lumPairs else 0.0
                            // PAPER-ON-PAPER: or a faint, continuous border line with the top sheet's blank margin inside.
                            stepFrac = max(stepFrac, edgeEvidence(a.x, a.y, b.x, b.y, nx, ny, true))
                            if (stepFrac < INTERIOR_STEP_FRAC) interiorOk = false
                            sup = max(sup, stepFrac)
                        }
                        sups[sIdx] = sup
                    }
                    if (!interiorOk) continue
                    val sortedSup = sups.sortedArray()
                    if (sortedSup[0] < QUAD_WEAKEST_SIDE_SUPPORT) continue
                    if (sortedSup[1] < QUAD_MIN_SIDE_SUPPORT) continue
                    val meanSup = sups.average()

                    // PAPER-ON-PAPER: printed frame? The paper beyond two OPPOSITE sides is a narrow strip of
                    // about the same width on both (a page margin). A sheet lying offset on another shows its
                    // strip on adjacent sides instead, because it is shifted one way.
                    if (interiorSides >= 2) {
                        val gaps = IntArray(4)
                        for (sIdx in 0 until 4) {
                            val a = quad[sIdx]
                            val b = quad[(sIdx + 1) % 4]
                            val len = sideLens[sIdx]
                            var nx = (b.y - a.y) / len
                            var ny = -(b.x - a.x) / len
                            val mx = (a.x + b.x) / 2
                            val my = (a.y + b.y) / 2
                            if (nx * (mx - qcx) + ny * (my - qcy) < 0) { nx = -nx; ny = -ny }
                            var d = 0
                            while (d < shortSide && paperAt((mx + nx * d).toInt(), (my + ny * d).toInt())) d++
                            gaps[sIdx] = d
                        }
                        val lim = FRAME_MARGIN * shortSide
                        val framed = (0..1).any { i ->
                            val g1 = gaps[i]; val g2 = gaps[i + 2]
                            g1 >= 3 && g2 >= 3 && g1 < lim && g2 < lim && min(g1, g2) >= 0.6 * max(g1, g2)
                        }
                        if (framed) continue
                    }

                    // PAPER-ON-PAPER: reject a quad with a full-length sheet border running through its middle.
                    run {
                        val poly = MatOfPoint2f(*quad)
                        val minSide = sideLens.minOrNull() ?: 0.0
                        var innerBorder = false
                        for ((sg, ev) in segEv) {
                            if (ev < INTERIOR_STEP_FRAC) continue
                            if (hypot(sg[2] - sg[0], sg[3] - sg[1]) < 0.5 * minSide) continue
                            val mid = Point((sg[0] + sg[2]) / 2, (sg[1] + sg[3]) / 2)
                            if (Imgproc.pointPolygonTest(poly, mid, true) >= shortSide * 0.04) { innerBorder = true; break }
                        }
                        poly.release()
                        if (innerBorder) return@run null
                        Unit
                    } ?: continue

                    val cx = quad.sumOf { it.x } / 4.0
                    val cy = quad.sumOf { it.y } / 4.0
                    val dCenter = hypot(cx - ww / 2.0, cy - wh / 2.0) / halfDiag

                    // CHANGED (C): prefer the quad that matches last frame's winner.
                    val prior = priorQuad?.let { iou(workSize, it, quad) } ?: 0.0

                    // Favour a well-supported, paper-filled, centred quad, with a mild
                    // preference for the larger one when several qualify.
                    val score = fill * fill * sortedSup[1] * meanSup *
                        (1.0 / (1.0 + 1.2 * dCenter)) * Math.pow(areaFrac, 0.6) *
                        (0.7 + 0.3 * prior)

                    // PAPER-ON-PAPER: a quad with a side INSIDE the paper is a sheet lying on others; prefer it
                    // over the outline of the whole pile, which otherwise always wins by being bigger.
                    if (interiorSides > 0 && score > bestNestedScore) {
                        bestNestedScore = score
                        bestNestedQuad = quad
                    }
                    if (score > bestScore) {
                        bestScore = score
                        bestQuad = quad
                    }
                }
            }
            bestNestedQuad?.let {
                bestQuad = it
                bestScore = bestNestedScore
            }

            if (bestQuad == null) {
                Log.d(TAG, "  quad: %d candidates scored, none passed".format(tried))
            } else {
                Log.d(TAG, "  quad: picked score=%.4f from %d candidates".format(bestScore, tried))
            }
            return bestQuad?.let { QuadResult(it, bestScore) }
        } finally {
            edges.release(); region.release(); edgesDil.release(); lines.release()
        }
    }

    /**
     * Finds the top sheet's quad. Runs two edge passes and keeps the better-scoring quad:
     * pass 1 with ordinary edges, pass 2 after local-contrast equalisation (CLAHE) with much lower
     * thresholds, which picks up the faint shadow line between a white sheet and the sheets under
     * it. (Taking the first pass that returns something let a smaller quad inside the page win.)
     * Both passes share a "paper brightness" image with text removed, used to check that a side
     * lying inside the paper blob really is a sheet border (see INTERIOR_STEP).
     */
    private fun findTopSheetQuad(binS: Mat, grayS: Mat, ww: Int, wh: Int): Array<Point>? {
        val lum = Mat()
        val eq = Mat()
        val tophat = Mat()
        try {
            val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
            Imgproc.morphologyEx(grayS, lum, Imgproc.MORPH_CLOSE, k) // dark text strokes disappear
            k.release()
            // PAPER-ON-PAPER: black top-hat = how much darker than the local paper each pixel is,
            // i.e. an ink map. Taken from the ORIGINAL grey image and shared by both passes.
            Core.subtract(lum, grayS, tophat)

            val r1 = findTopSheetQuadPass(binS, grayS, lum, tophat, ww, wh, 40.0, 120.0, 35)
            Imgproc.createCLAHE(3.0, Size(8.0, 8.0)).apply(grayS, eq)
            val r2 = findTopSheetQuadPass(binS, eq, lum, tophat, ww, wh, 12.0, 36.0, 25)

            val best = listOfNotNull(r1, r2).maxByOrNull { it.score } ?: return null
            // Remember the winner (normalised) so next frame prefers a consistent choice.
            prevTopQuadN = Array(4) { Point(best.quad[it].x / ww, best.quad[it].y / wh) }
            return best.quad
        } finally {
            lum.release()
            eq.release()
            tophat.release()
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
            // Recorded even when we bail out below: a ragged blob that no cut could
            // divide is still evidence of stacked sheets, and largestContour() needs
            // the number to decide that.
            frameRect = rect0
            // A blob that runs off the frame or fills most of it looks rectangular only
            // because the frame clips it (e.g. a pile of sheets) -> never treat it as clean.
            val areaFrac = Core.countNonZero(binS).toDouble() / (ww.toDouble() * wh)
            val sides = bordersTouched(binS)
            val clipped = sides >= 2 || areaFrac > 0.45
            // Low solidity means background shows through inside the hull, which a lone
            // sheet cannot do — so it keeps a pile from being waved through here even
            // when it sits clear of the frame edges and looks rectangular.
            val solid = solidity(binS)

            // CHANGED (B): hysteresis. Enter pile mode below SOLIDITY_GATE, but only leave
            // it once solidity is clearly back above the gate, so a pile that hovers around
            // the threshold does not alternate between "top sheet" and "whole pile".
            pileMode = if (pileMode) solid < SOLIDITY_GATE + PILE_HYSTERESIS else solid < SOLIDITY_GATE

            if (!ENABLE_TOP_SHEET_LOGIC) {
                Log.d(TAG, "top-sheet logic OFF: solidity=%.2f rect=%.2f (outline = raw mask contour)".format(solid, rect0))
                return
            }

            if (rect0 >= SPLIT_RECT_GATE && !clipped && !pileMode) {
                Log.d(TAG, "split skipped: rect=%.2f solidity=%.2f, clean sheet".format(rect0, solid))
                return
            }
            Log.d(
                TAG,
                "split check: rect=%.2f solidity=%.2f area=%.2f sides=%d pile=%s"
                    .format(rect0, solid, areaFrac, sides, pileMode)
            )

            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.resize(gray, grayS, Size(ww.toDouble(), wh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.GaussianBlur(grayS, grayS, Size(5.0, 5.0), 0.0)

            // When the hull is full of background wedges we are looking at a pile of
            // angled sheets. Cutting does badly there, so try to find the top sheet
            // directly first and only fall back to cutting if that finds nothing.
            if (pileMode || clipped) {
                val quad = findTopSheetQuad(binS, grayS, ww, wh)
                if (quad != null) {
                    val poly = MatOfPoint(*quad)
                    val qMask = Mat.zeros(wh, ww, CvType.CV_8UC1)
                    Imgproc.fillConvexPoly(qMask, poly, Scalar(255.0))
                    poly.release()
                    Imgproc.resize(qMask, up, Size(fullW.toDouble(), fullH.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
                    qMask.release()

                    // CHANGED (D): grow the quad a little so a curved/torn edge of the top
                    // sheet survives the AND below instead of being cut to a straight line.
                    var gk = (min(fullW, fullH) * TOP_QUAD_GROW).toInt()
                    gk = max(3, gk) or 1 // odd, >= 3
                    val gKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(gk.toDouble(), gk.toDouble()))
                    Imgproc.dilate(up, up, gKernel)
                    gKernel.release()

                    // AND rather than replace, so the mask can never claim paper where
                    // the segmenter saw none.
                    val before = Core.countNonZero(bin)
                    Core.bitwise_and(bin, up, bin)
                    val after = Core.countNonZero(bin)
                    // Only report overlapping sheets if the quad really removed something; a lone
                    // page that merely fills the frame loses almost nothing.
                    frameSplitApplied = pileMode || (before > 0 && (before - after) > 0.08 * before)
                    Log.d(TAG, "split: top-sheet quad won (solidity=%.2f, removed %.0f%% of blob)"
                        .format(solid, if (before > 0) 100.0 * (before - after) / before else 0.0))
                    return
                }
                Log.d(TAG, "split: quad search empty (solidity=%.2f, pile=%s)".format(solid, pileMode))
                // A single page that merely fills the frame must not be chopped up by the cuts below.
                if (!pileMode) return
            }

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

            // A split actually succeeded: the single blob really was multiple
            // overlapping sheets, and everything except the top one is about to
            // be dropped. Surfaced to JS so the user can be warned.
            frameSplitApplied = true

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
     * CHANGED (A): Mask -> outline, polygon first. The real (possibly concave) shape of the
     * page is kept, simplified to at most MAX_POLY_POINTS well-spaced vertices.
     *
     * With [FORCE_POLYGON] (default) the outline is never replaced by a rectangle; a flat
     * page still ends up as 4 refined corners because its simplified outline has 4
     * vertices. With FORCE_POLYGON off, a plain 4-corner page that matches the outline
     * (IoU >= QUAD_IOU) is snapped to a rectangle with refined corners, as before.
     */
    private fun polygonFromMask(mask: Mat, rgba: Mat): Array<Point>? {
        val bin = Mat()
        try {
            val contour = largestContour(mask, bin, rgba) ?: run {
                lastDebug = "no contour (mask empty / blob < 3% of frame) thr=${maskThr.toInt()}"
                return null
            }
            val hull = hullOf(contour)
            val quad = if (hull.size >= 3) DocumentDetector.orderCorners(quadFromHull(hull)) else null
            val poly = simplifyOutline(contour)

            if (poly == null || poly.size < 4) {
                return quad?.let { refineQuad(it, contour) }
            }

            val q = quad?.let { iou(bin.size(), it, poly) } ?: 0.0
            Log.d(TAG, "outline: poly=%d pts, iou(quad,poly)=%.3f, forcePolygon=%s".format(poly.size, q, FORCE_POLYGON))
            lastDebug = "mask poly=%d iou=%.2f solid=%.2f pile=%s split=%s topLogic=%s"
                .format(poly.size, q, lastSheetInfo.blobSolidity, pileMode, frameSplitApplied, ENABLE_TOP_SHEET_LOGIC)

            if (!FORCE_POLYGON && quad != null && q >= QUAD_IOU) {
                return refineQuad(quad, contour)
            }
            if (poly.size == 4) {
                return refineQuad(DocumentDetector.orderCorners(poly), contour)
            }
            return poly
        } catch (t: Throwable) {
            fitError = "fit ${t::class.java.simpleName}: ${t.message}"
            Log.w(TAG, "polygonFromMask failed: ${t::class.java.simpleName}: ${t.message}")
            return null
        } finally {
            bin.release()
        }
    }

    /** Intersection of the infinite lines a-b and c-d, or null if (nearly) parallel. */
    private fun cornerIntersection(a: Point, b: Point, c: Point, d: Point): Point? {
        val den = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)
        if (abs(den) < 1e-9) return null
        val t = ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / den
        return Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
    }

    /**
     * Restores sharp corners on a simplified outline.
     *
     * A rounded mask corner comes out of Douglas-Peucker as a short "chamfer" edge between two
     * vertices. For each such edge (short, flanked by two clearly longer edges, both ends
     * turning the same way) both vertices are replaced by the intersection of the two flanking
     * edges, i.e. the corner the page really has. Shortest chamfers are fixed first, repeated
     * until none are left.
     */
    private fun sharpenCorners(input: List<Point>): List<Point> {
        val pts = input.toMutableList()
        var guard = 0
        while (pts.size >= 5 && guard++ < 32) {
            val n = pts.size
            var peri = 0.0
            for (i in 0 until n) {
                val a = pts[i]
                val b = pts[(i + 1) % n]
                peri += hypot(b.x - a.x, b.y - a.y)
            }
            val maxChamfer = CHAMFER_MAX_FRACTION * peri

            var bestI = -1
            var bestLen = Double.MAX_VALUE
            var bestX: Point? = null
            for (i in 0 until n) {
                val p0 = pts[(i + n - 1) % n]
                val p1 = pts[i]
                val p2 = pts[(i + 1) % n]
                val p3 = pts[(i + 2) % n]

                val len = hypot(p2.x - p1.x, p2.y - p1.y)
                if (len < 1e-6 || len > maxChamfer) continue
                val lenPrev = hypot(p1.x - p0.x, p1.y - p0.y)
                val lenNext = hypot(p3.x - p2.x, p3.y - p2.y)
                if (lenPrev < 2.0 * len || lenNext < 2.0 * len) continue

                // Both ends of the chamfer must turn the same way (a cut corner, not an S-bend).
                val c1 = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x)
                val c2 = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x)
                if (c1 * c2 <= 0.0) continue

                // The two flanking edges must really form a corner (about 30..150 degrees of turn).
                val cross = (p1.x - p0.x) * (p3.y - p2.y) - (p1.y - p0.y) * (p3.x - p2.x)
                val dot = (p1.x - p0.x) * (p3.x - p2.x) + (p1.y - p0.y) * (p3.y - p2.y)
                val turn = Math.toDegrees(abs(atan2(cross, dot)))
                if (turn < 30.0 || turn > 150.0) continue

                val x = cornerIntersection(p0, p1, p2, p3) ?: continue
                val reach = 3.0 * len
                if (hypot(x.x - p1.x, x.y - p1.y) > reach || hypot(x.x - p2.x, x.y - p2.y) > reach) continue

                if (len < bestLen) {
                    bestLen = len
                    bestI = i
                    bestX = x
                }
            }

            val x = bestX
            if (bestI < 0 || x == null) break
            pts[bestI] = x
            pts.removeAt((bestI + 1) % n)
        }
        return pts
    }

    /** Moving average over +-[r] neighbours on a closed contour: removes pixel jitter before simplification. */
    private fun smoothClosed(pts: Array<Point>, r: Int): Array<Point> {
        val n = pts.size
        if (n < 2 * r + 1) return pts
        val w = (2 * r + 1).toDouble()
        return Array(n) { i ->
            var sx = 0.0
            var sy = 0.0
            for (d in -r..r) {
                val p = pts[(i + d + n) % n]
                sx += p.x
                sy += p.y
            }
            Point(sx / w, sy / w)
        }
    }

    /** Dense contour -> few, well-spaced vertices (no clusters), concavities preserved. */
    private fun simplifyOutline(contour: Array<Point>): Array<Point>? {
        val c2f = MatOfPoint2f(*smoothClosed(contour, 2))
        val approx = MatOfPoint2f()
        try {
            val peri = Imgproc.arcLength(c2f, true)
            var eps = 0.006 // CHANGED (A): ignores pixel jitter, still follows real bends
            do {
                Imgproc.approxPolyDP(c2f, approx, eps * peri, true)
                eps *= 1.35
            } while (approx.total() > MAX_POLY_POINTS && eps < 0.1)

            // Restore sharp corners BEFORE merging close vertices; merging a chamfer's two
            // vertices into their average is what pulled corners inwards.
            val sharpened = sharpenCorners(approx.toArray().toList())

            val minGap = 0.015 * peri // CHANGED (A): was 0.02
            val merged = ArrayList<Point>()
            for (p in sharpened) {
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