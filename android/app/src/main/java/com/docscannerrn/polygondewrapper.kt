package com.docscannerrn

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Dewarps a page bounded by an arbitrary CLOSED POLYGON with any number of
 * points — including curved / distorted edges — into a clean rectangle.
 *
 * Why this exists: a 4-point perspective transform (homography) can only
 * rectify a FLAT page photographed at an angle. If the page edges are curved
 * (a bent sheet, a curled notebook page, a torn edge), a homography either
 * clips the bulge or leaves the distortion in place, and simply masking the
 * polygon (crop-to-contour) keeps the distortion entirely.
 *
 * Approach — a **Coons patch** (transfinite interpolation) built in
 * homography-rectified space:
 *
 *  1. Pick the 4 extreme points of the polygon as the page corners, and split
 *     the polygon ring into the 4 edges between them. Any extra points the
 *     user placed on an edge become that edge's curve control points.
 *  2. Compute the homography H mapping those 4 corners onto the output
 *     rectangle, and push every edge point through H ("rectified space").
 *  3. Fit an arc-length parameterised Catmull-Rom spline through each
 *     rectified edge, giving four boundary curves Ct(u), Cb(u), Cl(v), Cr(v).
 *  4. For each output pixel, blend the four curves with the bilinear Coons
 *     formula to get a point in rectified space, then map it back through
 *     H^-1 to find the source pixel. Finally [Imgproc.remap].
 *
 * The key property: when all edges are straight, each rectified boundary
 * curve IS the corresponding rectangle edge, the Coons patch collapses to the
 * identity, and the result is **exactly** the old perspective transform. So
 * this strictly generalises the previous behaviour — flat pages are unchanged,
 * curved pages get genuinely unbent.
 */
object PolygonDewarper {

    /** Output images are capped on the long edge to bound remap-map memory. */
    private const val MAX_OUTPUT_EDGE = 2000.0

    /** Samples per spline segment when building the arc-length table. */
    private const val SAMPLES_PER_SEGMENT = 24

    class DewarpException(message: String) : Exception(message)

    /**
     * @param srcRgba source image (RGBA, as produced by Utils.bitmapToMat)
     * @param polygon ordered closed polygon in [srcRgba] pixel coordinates,
     *                4 or more points, either winding direction
     * @return a new dewarped RGBA Mat (caller owns it and must release)
     */
    fun dewarp(srcRgba: Mat, polygon: List<Point>): Mat {
        if (polygon.size < 4) throw DewarpException("Need at least 4 points, got ${polygon.size}")

        val cornerIdx = findCornerIndices(polygon)
        val ring = orientRing(polygon, cornerIdx)
        val edges = splitEdges(ring)

        // Output size from the longest opposing edge curves, so a curved (and
        // therefore longer) edge isn't squashed: the arc length is the page's
        // true unrolled dimension.
        var outW = max(arcLength(edges.top), arcLength(edges.bottom))
        var outH = max(arcLength(edges.left), arcLength(edges.right))
        if (outW < 1.0 || outH < 1.0) throw DewarpException("Degenerate polygon")

        val longEdge = max(outW, outH)
        if (longEdge > MAX_OUTPUT_EDGE) {
            val s = MAX_OUTPUT_EDGE / longEdge
            outW *= s
            outH *= s
        }
        val w = max(2, outW.roundToInt())
        val h = max(2, outH.roundToInt())

        // Homography: polygon corners -> output rectangle.
        val srcCorners = MatOfPoint2f(edges.tl, edges.tr, edges.br, edges.bl)
        val dstCorners = MatOfPoint2f(
            Point(0.0, 0.0),
            Point((w - 1).toDouble(), 0.0),
            Point((w - 1).toDouble(), (h - 1).toDouble()),
            Point(0.0, (h - 1).toDouble())
        )
        val hMat = Imgproc.getPerspectiveTransform(srcCorners, dstCorners)
        val hInv = Mat()
        Core.invert(hMat, hInv)
        val hi = DoubleArray(9)
        for (r in 0 until 3) for (c in 0 until 3) hi[r * 3 + c] = hInv.get(r, c)[0]

        // Edge curves, expressed in rectified space.
        val top = Curve(transform(edges.top, hMat))
        val bottom = Curve(transform(edges.bottom, hMat))
        val left = Curve(transform(edges.left, hMat))
        val right = Curve(transform(edges.right, hMat))

        srcCorners.release(); dstCorners.release(); hMat.release(); hInv.release()

        // u depends only on the output column and v only on the row, so the
        // boundary curves can be evaluated once per column/row instead of once
        // per pixel (O(w + h) spline evaluations rather than O(w * h)).
        val topX = DoubleArray(w); val topY = DoubleArray(w)
        val botX = DoubleArray(w); val botY = DoubleArray(w)
        for (x in 0 until w) {
            val u = x.toDouble() / (w - 1)
            val pt = top.at(u); topX[x] = pt.x; topY[x] = pt.y
            val pb = bottom.at(u); botX[x] = pb.x; botY[x] = pb.y
        }
        val leftX = DoubleArray(h); val leftY = DoubleArray(h)
        val rightX = DoubleArray(h); val rightY = DoubleArray(h)
        for (y in 0 until h) {
            val v = y.toDouble() / (h - 1)
            val pl = left.at(v); leftX[y] = pl.x; leftY[y] = pl.y
            val pr = right.at(v); rightX[y] = pr.x; rightY[y] = pr.y
        }

        val mapX = FloatArray(w * h)
        val mapY = FloatArray(w * h)
        val wm1 = (w - 1).toDouble()
        val hm1 = (h - 1).toDouble()

        for (y in 0 until h) {
            val v = y.toDouble() / hm1
            val lx = leftX[y]; val ly = leftY[y]
            val rx = rightX[y]; val ry = rightY[y]
            val rowBase = y * w
            for (x in 0 until w) {
                val u = x.toDouble() / wm1

                // Bilinear Coons blend. The corner-correction term simplifies
                // because the rectified corners are (0,0), (w-1,0), (w-1,h-1),
                // (0,h-1): it reduces to (u*(w-1), v*(h-1)).
                val sx = (1 - v) * topX[x] + v * botX[x] + (1 - u) * lx + u * rx - u * wm1
                val sy = (1 - v) * topY[x] + v * botY[x] + (1 - u) * ly + u * ry - v * hm1

                // Back to source pixels through H^-1.
                val den = hi[6] * sx + hi[7] * sy + hi[8]
                val i = rowBase + x
                if (abs(den) < 1e-12) {
                    mapX[i] = -1f
                    mapY[i] = -1f
                } else {
                    mapX[i] = ((hi[0] * sx + hi[1] * sy + hi[2]) / den).toFloat()
                    mapY[i] = ((hi[3] * sx + hi[4] * sy + hi[5]) / den).toFloat()
                }
            }
        }

        val mx = Mat(h, w, CvType.CV_32FC1)
        val my = Mat(h, w, CvType.CV_32FC1)
        mx.put(0, 0, mapX)
        my.put(0, 0, mapY)

        val dst = Mat()
        Imgproc.remap(
            srcRgba, dst, mx, my,
            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT,
            Scalar(255.0, 255.0, 255.0, 255.0)
        )
        mx.release(); my.release()
        return dst
    }

    // ------------------------------------------------------------------
    // Corner identification and ring ordering
    // ------------------------------------------------------------------

    private class CornerIdx(val tl: Int, val tr: Int, val br: Int, val bl: Int)

    /**
     * Picks the 4 polygon points that act as page corners: extremes of x+y and
     * x-y. For a roughly rectangular page this reliably finds the true corners
     * regardless of how many extra points sit along the edges.
     */
    private fun findCornerIndices(pts: List<Point>): CornerIdx {
        var tl = 0; var br = 0; var tr = 0; var bl = 0
        var minSum = Double.MAX_VALUE; var maxSum = -Double.MAX_VALUE
        var minDiff = Double.MAX_VALUE; var maxDiff = -Double.MAX_VALUE
        for (i in pts.indices) {
            val sum = pts[i].x + pts[i].y
            val diff = pts[i].x - pts[i].y
            if (sum < minSum) { minSum = sum; tl = i }
            if (sum > maxSum) { maxSum = sum; br = i }
            if (diff > maxDiff) { maxDiff = diff; tr = i }
            if (diff < minDiff) { minDiff = diff; bl = i }
        }
        val chosen = listOf(tl, tr, br, bl)
        if (chosen.toSet().size == 4) return CornerIdx(tl, tr, br, bl)

        // Degenerate (e.g. a rotated 45-degree shape can collide extremes):
        // fall back to the min-area rectangle and snap each of its corners to
        // the nearest still-unused polygon point.
        val mop = MatOfPoint(*pts.toTypedArray())
        val rect = Imgproc.minAreaRect(MatOfPoint2f(*pts.toTypedArray()))
        val boxPts = arrayOfNulls<Point>(4)
        rect.points(boxPts)
        mop.release()

        val used = mutableSetOf<Int>()
        val resolved = IntArray(4)
        for (k in 0 until 4) {
            val target = boxPts[k] ?: Point(0.0, 0.0)
            var bestI = -1
            var bestD = Double.MAX_VALUE
            for (i in pts.indices) {
                if (i in used) continue
                val d = hypot(pts[i].x - target.x, pts[i].y - target.y)
                if (d < bestD) { bestD = d; bestI = i }
            }
            if (bestI < 0) bestI = pts.indices.first { it !in used }
            used.add(bestI)
            resolved[k] = bestI
        }
        // minAreaRect's points() are ordered consistently around the rect, so
        // re-derive which is which by the same sum/diff test on just these 4.
        val four = resolved.map { pts[it] }
        var t = 0; var b = 0; var r = 0; var l = 0
        var mnS = Double.MAX_VALUE; var mxS = -Double.MAX_VALUE
        var mnD = Double.MAX_VALUE; var mxD = -Double.MAX_VALUE
        for (i in 0 until 4) {
            val sum = four[i].x + four[i].y
            val diff = four[i].x - four[i].y
            if (sum < mnS) { mnS = sum; t = i }
            if (sum > mxS) { mxS = sum; b = i }
            if (diff > mxD) { mxD = diff; r = i }
            if (diff < mnD) { mnD = diff; l = i }
        }
        return CornerIdx(resolved[t], resolved[r], resolved[b], resolved[l])
    }

    private class Ring(val points: List<Point>, val pTr: Int, val pBr: Int, val pBl: Int)

    /**
     * Rotates the polygon so the top-left corner is first and walks it in
     * whichever direction yields the order TL -> TR -> BR -> BL, making the
     * result independent of the input winding.
     */
    private fun orientRing(pts: List<Point>, c: CornerIdx): Ring {
        val n = pts.size
        for (dir in intArrayOf(1, -1)) {
            val order = IntArray(n) { (((c.tl + dir * it) % n) + n) % n }
            val posOf = IntArray(n)
            for (p in 0 until n) posOf[order[p]] = p
            val pTr = posOf[c.tr]; val pBr = posOf[c.br]; val pBl = posOf[c.bl]
            if (pTr in 1 until pBr && pBr < pBl) {
                return Ring(order.map { pts[it] }, pTr, pBr, pBl)
            }
        }
        // Fall back to forward order; edges may be unevenly split but the
        // transform still produces a sane image.
        val order = IntArray(n) { (c.tl + it) % n }
        val posOf = IntArray(n)
        for (p in 0 until n) posOf[order[p]] = p
        return Ring(order.map { pts[it] }, posOf[c.tr], posOf[c.br], posOf[c.bl])
    }

    private class Edges(
        val top: List<Point>,    // TL -> TR
        val right: List<Point>,  // TR -> BR
        val bottom: List<Point>, // BL -> BR
        val left: List<Point>,   // TL -> BL
        val tl: Point, val tr: Point, val br: Point, val bl: Point
    )

    /** Splits the oriented ring into the four edge point lists. */
    private fun splitEdges(ring: Ring): Edges {
        val p = ring.points
        val top = p.subList(0, ring.pTr + 1).toList()
        val right = p.subList(ring.pTr, ring.pBr + 1).toList()
        val brToBl = p.subList(ring.pBr, ring.pBl + 1).toList()
        val blToTl = (p.subList(ring.pBl, p.size) + listOf(p[0])).toList()
        return Edges(
            top = top,
            right = right,
            bottom = brToBl.reversed(),
            left = blToTl.reversed(),
            tl = p[0], tr = p[ring.pTr], br = p[ring.pBr], bl = p[ring.pBl]
        )
    }

    private fun arcLength(pts: List<Point>): Double {
        var total = 0.0
        for (i in 0 until pts.size - 1) {
            total += hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y)
        }
        return total
    }

    private fun transform(pts: List<Point>, h: Mat): List<Point> {
        val src = MatOfPoint2f(*pts.toTypedArray())
        val dst = MatOfPoint2f()
        Core.perspectiveTransform(src, dst, h)
        val out = dst.toArray().toList()
        src.release(); dst.release()
        return out
    }

    // ------------------------------------------------------------------
    // Arc-length parameterised Catmull-Rom curve
    // ------------------------------------------------------------------

    /**
     * A curve through the given control points, evaluable by normalised
     * arc length so control-point spacing doesn't distort the sampling.
     * Two points give a straight line (which is what keeps the flat-page case
     * identical to a plain perspective transform).
     */
    private class Curve(control: List<Point>) {
        private val xs: DoubleArray
        private val ys: DoubleArray
        private val cum: DoubleArray
        private val total: Double

        init {
            val samples = ArrayList<Point>()
            if (control.size == 2) {
                samples.add(control[0])
                samples.add(control[1])
            } else {
                val n = control.size
                for (i in 0 until n - 1) {
                    val p0 = control[max(i - 1, 0)]
                    val p1 = control[i]
                    val p2 = control[i + 1]
                    val p3 = control[min(i + 2, n - 1)]
                    val steps = SAMPLES_PER_SEGMENT
                    for (s in 0 until steps) {
                        val t = s.toDouble() / steps
                        samples.add(catmullRom(p0, p1, p2, p3, t))
                    }
                }
                samples.add(control[n - 1])
            }

            xs = DoubleArray(samples.size)
            ys = DoubleArray(samples.size)
            cum = DoubleArray(samples.size)
            for (i in samples.indices) {
                xs[i] = samples[i].x
                ys[i] = samples[i].y
                cum[i] = if (i == 0) 0.0
                else cum[i - 1] + hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            }
            total = cum[cum.size - 1]
        }

        /** Point at normalised arc length [t] in 0..1. */
        fun at(t: Double): Point {
            if (total <= 0.0) return Point(xs[0], ys[0])
            val target = t.coerceIn(0.0, 1.0) * total

            // Binary search for the sample bracketing this arc length.
            var lo = 0
            var hi = cum.size - 1
            while (lo < hi) {
                val mid = (lo + hi) / 2
                if (cum[mid] < target) lo = mid + 1 else hi = mid
            }
            if (lo == 0) return Point(xs[0], ys[0])

            val segLen = cum[lo] - cum[lo - 1]
            val f = if (segLen > 0) (target - cum[lo - 1]) / segLen else 0.0
            return Point(
                xs[lo - 1] + (xs[lo] - xs[lo - 1]) * f,
                ys[lo - 1] + (ys[lo] - ys[lo - 1]) * f
            )
        }

        private fun catmullRom(p0: Point, p1: Point, p2: Point, p3: Point, t: Double): Point {
            val t2 = t * t
            val t3 = t2 * t
            val x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t +
                    (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
                    (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3)
            val y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t +
                    (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
                    (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)
            return Point(x, y)
        }
    }
}
