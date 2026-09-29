package com.docscannerrn

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

/**
 * Single source of truth for document (page) detection, shared by:
 *   - DocScannerModule.detectAllDocuments (runs on a captured full-res still,
 *     with the color/HSV pass enabled), and
 *   - DocumentScannerView's live camera analyzer (runs per frame on the
 *     grayscale luma plane only, for speed — HSV pass disabled).
 *
 * Given a grayscale Mat (and optionally an HSV Mat of the same size), it finds
 * the single most page-like quadrilateral and returns its 4 corners ordered
 * [top-left, top-right, bottom-right, bottom-left] in that Mat's pixel
 * coordinates, or null if nothing convincing is found.
 */
object DocumentDetector {

    /**
     * @param gray single-channel 8-bit image to detect in.
     * @param hsv  optional 3-channel HSV image (same size as [gray]); when
     *             provided, a "white paper" saturation/value mask pass is added,
     *             which separates a white page from a COLORED background far
     *             better than brightness alone. Pass null on the hot per-frame
     *             path to keep it fast.
     */
    fun findBestQuad(gray: Mat, hsv: Mat?): Array<Point>? {
        val w = gray.cols()
        val h = gray.rows()
        val imageArea = (w * h).toDouble()

        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        val allContours = mutableListOf<MatOfPoint>()

        // (a) Canny edges.
        val edges = Mat()
        Imgproc.Canny(blurred, edges, 75.0, 200.0)
        val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.dilate(edges, edges, k5)
        val h1 = Mat()
        val c1 = mutableListOf<MatOfPoint>()
        Imgproc.findContours(edges, c1, h1, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        allContours.addAll(c1)

        val k15 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))

        // (b) Otsu brightness threshold (bright sheet on darker surface).
        val otsu = Mat()
        Imgproc.threshold(blurred, otsu, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        Imgproc.morphologyEx(otsu, otsu, Imgproc.MORPH_CLOSE, k15)
        val h2 = Mat()
        val c2 = mutableListOf<MatOfPoint>()
        Imgproc.findContours(otsu, c2, h2, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        allContours.addAll(c2)

        // (c) HSV white-paper mask (optional): low saturation + high value.
        var whiteMask: Mat? = null
        var h3: Mat? = null
        if (hsv != null) {
            val wm = Mat()
            Core.inRange(hsv, Scalar(0.0, 0.0, 110.0), Scalar(180.0, 80.0, 255.0), wm)
            Imgproc.morphologyEx(wm, wm, Imgproc.MORPH_CLOSE, k15)
            val hh = Mat()
            val c3 = mutableListOf<MatOfPoint>()
            Imgproc.findContours(wm, c3, hh, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            allContours.addAll(c3)
            whiteMask = wm
            h3 = hh
        }

        val minArea = imageArea * 0.08
        val maxArea = imageArea * 0.95

        val candidates = allContours
            .map { it to Imgproc.contourArea(it) }
            .filter { it.second in minArea..maxArea }
            .sortedByDescending { it.second }
            .take(15)

        // Keep the most page-like quad: fills most of its minimum rotated
        // rectangle (rectangularity), sizeable, and not hugging the borders.
        var best: Array<Point>? = null
        var bestScore = 0.0

        for ((contour, _) in candidates) {
            val quad = contourToQuad(contour) ?: continue
            if (isNearFullFrame(quad, w, h)) continue

            val q2f = MatOfPoint2f(*quad)
            val quadArea = Imgproc.contourArea(q2f)
            if (quadArea < minArea || quadArea > maxArea) {
                q2f.release()
                continue
            }
            val rot = Imgproc.minAreaRect(q2f)
            q2f.release()
            val rectArea = rot.size.width * rot.size.height
            val rectangularity = if (rectArea > 0) quadArea / rectArea else 0.0
            if (rectangularity < 0.75) continue

            val borderPenalty = borderTouchFraction(quad, w, h)
            val areaWeight = quadArea / imageArea
            val score = rectangularity + 0.3 * areaWeight - 0.5 * borderPenalty
            if (score > bestScore) {
                bestScore = score
                best = quad
            }
        }

        // Expand slightly outward: approxPolyDP tends to cut corners a little
        // inside the true edge (especially on crumpled/curved paper), which
        // clips the page. A small outward margin errs toward including a sliver
        // of background instead — the user can still tighten it manually.
        val ordered = best?.let { expandQuad(orderCorners(it), w, h, 0.03) }

        blurred.release(); edges.release(); h1.release()
        otsu.release(); h2.release()
        whiteMask?.release(); h3?.release()

        return ordered
    }

    /**
     * Reduce a contour to a clean 4-corner quad. Convex hull first (bridges a
     * spiral binding / ragged or gappy edge), then sweep increasing epsilon on
     * the hull until it collapses to exactly 4 vertices. Null if not possible.
     */
    private fun contourToQuad(contour: MatOfPoint): Array<Point>? {
        val hullIndices = MatOfInt()
        Imgproc.convexHull(contour, hullIndices)
        val contourPts = contour.toArray()
        val hullPts = hullIndices.toArray().map { contourPts[it] }.toTypedArray()
        if (hullPts.size < 4) return null

        val hull2f = MatOfPoint2f(*hullPts)
        val peri = Imgproc.arcLength(hull2f, true)
        for (epsFactor in listOf(0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)) {
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(hull2f, approx, epsFactor * peri, true)
            if (approx.total() == 4L) {
                return approx.toArray()
            }
        }
        return null
    }

    private fun isNearFullFrame(quad: Array<Point>, w: Int, h: Int): Boolean {
        var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
        var maxX = -Double.MAX_VALUE; var maxY = -Double.MAX_VALUE
        for (p in quad) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }
        val mx = w * 0.03
        val my = h * 0.03
        return minX <= mx && minY <= my && maxX >= w - mx && maxY >= h - my
    }

    /** Fraction (0..1) of the quad's 4 corners within ~3% of an image border. */
    private fun borderTouchFraction(quad: Array<Point>, w: Int, h: Int): Double {
        val mx = w * 0.03
        val my = h * 0.03
        var touching = 0
        for (p in quad) {
            if (p.x <= mx || p.x >= w - mx || p.y <= my || p.y >= h - my) touching++
        }
        return touching / 4.0
    }

    /**
     * Pushes each corner away from the quad's centroid by [factor] (e.g. 0.03 =
     * 3%), clamped to the image bounds, to counteract the slight inward bias of
     * polygon simplification.
     */
    private fun expandQuad(quad: Array<Point>, w: Int, h: Int, factor: Double): Array<Point> {
        var cx = 0.0; var cy = 0.0
        for (p in quad) { cx += p.x; cy += p.y }
        cx /= quad.size; cy /= quad.size
        return Array(quad.size) { i ->
            val p = quad[i]
            val nx = (cx + (p.x - cx) * (1 + factor)).coerceIn(0.0, (w - 1).toDouble())
            val ny = (cy + (p.y - cy) * (1 + factor)).coerceIn(0.0, (h - 1).toDouble())
            Point(nx, ny)
        }
    }

    /** Orders 4 points as [top-left, top-right, bottom-right, bottom-left]. */
    fun orderCorners(points: Array<Point>): Array<Point> {
        var tl = points[0]; var br = points[0]; var tr = points[0]; var bl = points[0]
        var minSum = Double.MAX_VALUE; var maxSum = -Double.MAX_VALUE
        var minDiff = Double.MAX_VALUE; var maxDiff = -Double.MAX_VALUE
        for (p in points) {
            val sum = p.x + p.y
            val diff = p.x - p.y
            if (sum < minSum) { minSum = sum; tl = p }
            if (sum > maxSum) { maxSum = sum; br = p }
            if (diff > maxDiff) { maxDiff = diff; tr = p }
            if (diff < minDiff) { minDiff = diff; bl = p }
        }
        return arrayOf(tl, tr, br, bl)
    }
}
