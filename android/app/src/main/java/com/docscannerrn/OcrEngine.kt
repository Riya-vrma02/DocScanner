package com.docscannerrn

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.ml.KNearest
import org.opencv.ml.Ml
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * OCR built only from OpenCV (classic image processing + OpenCV's own k-nearest-neighbour
 * classifier). No model files and no extra library.
 *
 *  1. Binarise the page (ink = dark locally AND globally).
 *  2. Connected components = letter pieces. Chain them left to right into text lines, fit each
 *     line's baseline, attach dots/commas/apostrophes, glue i/j dots onto their stems.
 *  3. Each letter -> feature vector: its shape scaled into a 20x20 grid + where it sits relative
 *     to the baseline and cap height (so 'o'/'O', 'p'/'P', ','/''' can be told apart).
 *  4. k-NN against letters the phone draws itself at start-up from its built-in fonts
 *     (sans, serif, mono; normal, bold, thin and blurred variants).
 *  5. Touching letters are cut at thin columns when the cut pieces match better.
 *  6. Words from the gaps between letters; then l/I/1, O/0, i/! are fixed from the word around them.
 *
 * Works on printed text (any common font, roughly 10 pt and up on an A4 scan). Not for handwriting.
 * recognize(bitmap) returns lines; each line's words carry a 4-corner box (tl, tr, br, bl) in
 * bitmap pixels for the searchable/selectable PDF text layer.
 */
class OcrEngine private constructor() {

    class Word(val text: String, /** x0,y0 (tl), x1,y1 (tr), x2,y2 (br), x3,y3 (bl) */ val quad: FloatArray)
    /**
     * One text line. box = l, t, r, b (px); capH = its capital-letter height (px);
     * fit = median distance of its letters to the closest known letter (lower = more certain).
     */
    class Line(val text: String, val score: Float, val words: List<Word>,
               val box: IntArray = IntArray(4), val capH: Float = 0f, val fit: Float = 0f)

    /** Everything read from one page: text lines + boxes of things that aren't text (pictures, equations, rules). */
    class Page(val lines: List<Line>, val nonText: List<IntArray>, val width: Int, val height: Int)

    companion object {
        private const val CHARSET =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?'\"()-/&%@#*+=[]\$"
        private const val GS = 20                 // letter shape grid (GS x GS)
        private const val POS_W = 6f              // weight of the position features vs the shape pixels
        private val TRAIN_CAPS = intArrayOf(14, 22, 34)   // cap heights (px) the letters are drawn at
        private const val K = 3
        private const val ADAPT_C = 30.0
        private const val MAX_FIT = 60f           // a line whose letters fit worse than this isn't text

        @Volatile private var instance: OcrEngine? = null

        /** Shared engine (its letter set is drawn once). Null if it couldn't be built. */
        fun get(@Suppress("UNUSED_PARAMETER") context: Context): OcrEngine? {
            instance?.let { return it }
            synchronized(this) {
                instance?.let { return it }
                return try { OcrEngine().also { instance = it } } catch (t: Throwable) { null }
            }
        }

        private val VOWELS = "aeiouy"
        private const val PUNCT = ".,;:!?\"'()[]"
    }

    private val knn: KNearest = KNearest.create()

    init { train() }

    // ================================================================== features

    /** bin: CV_8U, ink = 255, tight box. Aspect-preserving fit into GS x GS, centred, softened. */
    private fun shapeVec(bin: Mat, out: FloatArray) {
        val h = bin.rows(); val w = bin.cols()
        val s = (GS - 2).toDouble() / max(h, w)
        val nw = max(1, Math.round(w * s).toInt()); val nh = max(1, Math.round(h * s).toInt())
        val small = Mat()
        Imgproc.resize(bin, small, Size(nw.toDouble(), nh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val g = Mat.zeros(GS, GS, CvType.CV_32F)
        val x0 = (GS - nw) / 2; val y0 = (GS - nh) / 2
        val roi = g.submat(y0, y0 + nh, x0, x0 + nw)
        small.convertTo(roi, CvType.CV_32F, 1.0 / 255.0)
        Imgproc.GaussianBlur(g, g, Size(3.0, 3.0), 0.8)
        val v = FloatArray(GS * GS)
        g.get(0, 0, v)
        System.arraycopy(v, 0, out, 0, GS * GS)
        small.release(); roi.release(); g.release()
    }

    /** Geometry relative to the line: baseline y [base], cap height [capH] (px). */
    private fun posVec(top: Double, bottom: Double, left: Double, right: Double, base: Double, capH: Double, out: FloatArray) {
        val h = bottom - top; val w = right - left
        val o = GS * GS
        out[o] = ((base - top) / capH).toFloat() * POS_W          // how high it reaches
        out[o + 1] = ((base - bottom) / capH).toFloat() * POS_W   // below the baseline? (p, g, y)
        out[o + 2] = (h / capH).toFloat() * POS_W
        out[o + 3] = (w / capH).toFloat() * POS_W
        out[o + 4] = ln((w + 1) / (h + 1)).toFloat() * POS_W
    }

    // ================================================================== training (letters drawn by the phone)

    private fun train() {
        val fonts = listOf(
            Typeface.create(Typeface.SANS_SERIF, Typeface.NORMAL),
            Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD),
            Typeface.create(Typeface.SERIF, Typeface.NORMAL),
            Typeface.create(Typeface.SERIF, Typeface.BOLD),
            Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL),
        )
        val rows = ArrayList<FloatArray>()
        val labels = ArrayList<Int>()
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK }
        val bounds = Rect()
        val kernel = Mat.ones(2, 2, CvType.CV_8U)
        for (tf in fonts) {
            paint.typeface = tf
            for (cap in TRAIN_CAPS) {
                paint.textSize = 100f
                paint.getTextBounds("H", 0, 1, bounds)
                paint.textSize = 100f * cap / bounds.height()
                val S = cap * 4
                val base = (cap * 2.5).toInt()
                val bmp = Bitmap.createBitmap(S, S, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(bmp)
                val rgba = Mat(); val gray = Mat(); val bin = Mat(); val tmp = Mat()
                for ((ci, ch) in CHARSET.withIndex()) {
                    canvas.drawColor(Color.WHITE)
                    canvas.drawText(ch.toString(), cap.toFloat(), base.toFloat(), paint)
                    Utils.bitmapToMat(bmp, rgba)
                    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
                    for (variant in 0 until 4) {
                        when (variant) {
                            0 -> Imgproc.threshold(gray, bin, 127.0, 255.0, Imgproc.THRESH_BINARY_INV)
                            1 -> {   // heavier print
                                Imgproc.threshold(gray, bin, 127.0, 255.0, Imgproc.THRESH_BINARY_INV)
                                Imgproc.dilate(bin, bin, kernel)
                            }
                            2 -> {   // lighter print (small letters can't get thinner: kept as drawn)
                                Imgproc.threshold(gray, bin, 127.0, 255.0, Imgproc.THRESH_BINARY_INV)
                                if (cap >= 22) Imgproc.erode(bin, bin, kernel)
                            }
                            else -> { // blurred print: blur the drawing and threshold it again
                                Core.bitwise_not(gray, tmp)
                                Imgproc.GaussianBlur(tmp, tmp, Size(0.0, 0.0), 0.06 * cap + 0.4)
                                Imgproc.threshold(tmp, bin, 100.0, 255.0, Imgproc.THRESH_BINARY)
                            }
                        }
                        val box = Imgproc.boundingRect(bin)
                        if (box.width <= 0 || box.height <= 0) continue
                        val crop = bin.submat(box)
                        val f = FloatArray(GS * GS + 5)
                        shapeVec(crop, f)
                        crop.release()
                        posVec(box.y.toDouble(), (box.y + box.height).toDouble(), box.x.toDouble(),
                            (box.x + box.width).toDouble(), base.toDouble(), cap.toDouble(), f)
                        rows.add(f); labels.add(ci)
                    }
                }
                rgba.release(); gray.release(); bin.release(); tmp.release()
                bmp.recycle()
            }
        }
        kernel.release()
        val D = GS * GS + 5
        val samples = Mat(rows.size, D, CvType.CV_32F)
        val flat = FloatArray(rows.size * D)
        for ((i, r) in rows.withIndex()) System.arraycopy(r, 0, flat, i * D, D)
        samples.put(0, 0, flat)
        val resp = Mat(rows.size, 1, CvType.CV_32F)
        resp.put(0, 0, FloatArray(labels.size) { labels[it].toFloat() })
        knn.defaultK = K
        knn.train(samples, Ml.ROW_SAMPLE, resp)
        samples.release(); resp.release()
    }

    private val qMat = Mat(1, GS * GS + 5, CvType.CV_32F)
    private val qRes = Mat(); private val qNb = Mat(); private val qDist = Mat()

    /** (char, distance to the closest drawn letter; lower = better fit) */
    private fun classify(f: FloatArray): Pair<Char, Float> {
        qMat.put(0, 0, f)
        knn.findNearest(qMat, K, qRes, qNb, qDist)
        val idx = qRes.get(0, 0)[0].roundToInt().coerceIn(0, CHARSET.length - 1)
        return CHARSET[idx] to qDist.get(0, 0)[0].toFloat()
    }

    // ================================================================== page

    /** A group of connected components (one letter, possibly several pieces like 'i'). Box is [l, r) x [t, b). */
    private class Comp(var l: Int, var t: Int, var r: Int, var b: Int, var ids: IntArray) {
        val h get() = b - t
        val w get() = r - l
        val cy get() = (t + b) / 2.0
        val cx get() = (l + r) / 2.0
        fun absorb(o: Comp) {
            l = min(l, o.l); t = min(t, o.t); r = max(r, o.r); b = max(b, o.b); ids = ids + o.ids
        }
    }

    private class Ch(val c: Char, val l: Int, val t: Int, val r: Int, val b: Int, val d: Float)

    /** Text lines of a page (for the PDF text layer). */
    fun recognize(bitmap: Bitmap): List<Line> = analyze(bitmap).lines

    /** Text lines plus the non-text regions of a page (for page layout, e.g. LaTeX export). */
    @Synchronized
    fun analyze(bitmap: Bitmap): Page {
        val rgba = Mat(); val gray = Mat(); val bin = Mat(); val otsu = Mat()
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        try {
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            rgba.release()
            val blk = max(15, (min(gray.rows(), gray.cols()) / 40) or 1)
            Imgproc.adaptiveThreshold(gray, bin, 255.0, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, blk, ADAPT_C)
            Imgproc.threshold(gray, otsu, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
            Core.bitwise_and(bin, otsu, bin)     // ink must be dark locally AND globally
            val n = Imgproc.connectedComponentsWithStats(bin, labels, stats, cents, 8, CvType.CV_32S)
            val st = IntArray(n * 5)
            stats.get(0, 0, st)
            val nonText = ArrayList<IntArray>()
            val lines = readPage(n, st, labels, nonText)
            return Page(lines, nonText, bitmap.width, bitmap.height)
        } finally {
            rgba.release(); gray.release(); bin.release(); otsu.release()
            labels.release(); stats.release(); cents.release()
        }
    }

    private fun readPage(n: Int, st: IntArray, labels: Mat, nonText: MutableList<IntArray>): List<Line> {
        var comps = ArrayList<Comp>()
        for (i in 1 until n) {
            val x = st[i * 5]; val y = st[i * 5 + 1]; val w = st[i * 5 + 2]; val h = st[i * 5 + 3]; val a = st[i * 5 + 4]
            if (a >= 3) comps.add(Comp(x, y, x + w, y + h, intArrayOf(i)))
        }
        if (comps.isEmpty()) return emptyList()
        val tallEnough = comps.filter { it.h >= 6 }.map { it.h.toDouble() }
        val medH = if (tallEnough.isNotEmpty()) median(tallEnough) else 10.0
        // pictures and ruled lines aren't letters
        for (c in comps) if (c.h > 4 * medH || c.w > 15 * medH) nonText.add(intArrayOf(c.l, c.t, c.r, c.b))
        comps = ArrayList(comps.filter { it.h <= 4 * medH && it.w <= 15 * medH })

        val core = comps.filter { it.h >= 0.6 * medH }.sortedBy { it.l }
        val small = comps.filter { it.h < 0.6 * medH }

        // ---- text lines: chain the letters left to right, following the line's drift
        var lines = ArrayList<ArrayList<Comp>>()
        for (c in core) {
            var best: ArrayList<Comp>? = null; var bd = 1e9
            for (L in lines) {
                if (c.l - L.last().r > 6 * medH) continue
                for (last in L.takeLast(3)) {   // the last few, so an i-dot or comma can't break the chain
                    val d = abs(c.cy - last.cy)
                    val hh = maxOf(c.h.toDouble(), last.h.toDouble(), medH)
                    val vov = min(c.b, last.b) - max(c.t, last.t)
                    if (d < 0.45 * hh && vov > 0.3 * min(c.h, last.h) && d < bd) { bd = d; best = L }
                }
            }
            if (best == null) lines.add(arrayListOf(c)) else best.add(c)
        }
        // join pieces of the same line that were split by a wide gap
        lines.sortBy { L -> median(L.map { it.cy }) }
        val joined = ArrayList<ArrayList<Comp>>()
        for (L in lines) {
            val ly = median(L.map { it.cy })
            var done = false
            for (M in joined) {
                val my = median(M.map { it.cy })
                if (abs(ly - my) < 0.4 * medH &&
                    (L.first().l >= M.maxOf { it.r } - 2 || L.last().r <= M.minOf { it.l } + 2)) {
                    M.addAll(L); M.sortBy { it.l }; done = true; break
                }
            }
            if (!done) joined.add(L)
        }
        lines = joined

        // ---- baseline per line: straight-line fit through letter bottoms, ignoring descenders
        val slope = DoubleArray(lines.size); val icpt = DoubleArray(lines.size)
        for ((li, L) in lines.withIndex()) {
            val xs = L.map { it.cx }; val bs = L.map { it.b.toDouble() }
            val medLH = median(L.map { it.h.toDouble() })
            var keep = BooleanArray(L.size) { true }
            var m = 0.0; var b0 = median(bs)
            repeat(3) {
                val idx = L.indices.filter { keep[it] }
                if (idx.size >= 3) {
                    val mx = idx.sumOf { xs[it] } / idx.size; val my = idx.sumOf { bs[it] } / idx.size
                    var sxx = 0.0; var sxy = 0.0
                    for (i in idx) { sxx += (xs[i] - mx) * (xs[i] - mx); sxy += (xs[i] - mx) * (bs[i] - my) }
                    m = if (sxx > 1e-9) sxy / sxx else 0.0
                    b0 = my - m * mx
                } else { m = 0.0; b0 = median(bs) }
                keep = BooleanArray(L.size) { bs[it] - (m * xs[it] + b0) < 0.12 * medLH + 1 }
            }
            slope[li] = m; icpt[li] = b0
        }

        // ---- dots, commas, apostrophes, hyphens: to the line whose band they sit in
        val ext = lines.map { L -> L.minOf { it.l } to L.maxOf { it.r } }
        for (c in small) {
            var best = -1; var bd = 1e9
            for (li in lines.indices) {
                if (c.cx < ext[li].first - 2 * medH || c.cx > ext[li].second + 2 * medH) continue
                val base = slope[li] * c.cx + icpt[li]
                if (c.cy >= base - 2.0 * medH && c.cy <= base + 0.6 * medH) {
                    val d = abs(c.cy - (base - 0.5 * medH))
                    if (d < bd) { bd = d; best = li }
                }
            }
            if (best >= 0) lines[best].add(c)
        }

        val out = ArrayList<Pair<Double, Line>>()
        for ((li, L0) in lines.withIndex()) {
            val m = slope[li]; val b0 = icpt[li]
            val G = groupPieces(L0)
            if (G.isEmpty()) continue
            val hts = G.map { it.h.toDouble() }
            val capH = max(4.0, if (hts.size >= 3) percentile(hts, 80.0) else hts.max())

            val chars = ArrayList<Ch>()
            for (g in G) {
                val fb = feature(g, g.l, g.r, labels, m, b0, capH) ?: continue
                var (ch, d) = classify(fb.first)
                val box = fb.second
                val w = box[2] - box[0]; val h = box[3] - box[1]
                // a narrow stem with a dot: which side the dot is on decides (i/j above, ! below)
                if (g.ids.size == 2 && w < 0.5 * capH + 2) {
                    val rowsA = inkRows(g, g.ids[0], labels); val rowsB = inkRows(g, g.ids[1], labels)
                    if (rowsA != null && rowsB != null) {
                        val hA = rowsA.second - rowsA.first; val hB = rowsB.second - rowsB.first
                        val (stem, dot) = if (hA >= hB) rowsA to rowsB else rowsB to rowsA
                        if (min(hA, hB) < 0.5 * max(hA, hB)) {
                            val base = m * (box[0] + box[2]) / 2.0 + b0
                            if (dot.second < stem.first) {
                                ch = if (g.t + stem.second > base + 0.12 * capH) 'j' else 'i'
                            } else if (ch != '?') ch = '!'
                        }
                    }
                }
                // touching letters: a wide blob -> try cutting it at thin columns
                if (w > 0.75 * capH && w > 0.9 * h) {
                    val sp = split(g, labels, m, b0, capH)
                    if (sp != null && sp.first < 0.75 * d) { chars.addAll(sp.second); continue }
                }
                chars.add(Ch(ch, box[0], box[1], box[2], box[3], d))
            }
            if (chars.isEmpty()) continue
            val lbox = intArrayOf(chars.minOf { it.l }, chars.minOf { it.t }, chars.maxOf { it.r }, chars.maxOf { it.b })
            val fit = median(chars.map { it.d.toDouble() }).toFloat()
            if (fit > MAX_FIT) { nonText.add(lbox); continue }   // not text (picture, noise)

            // ---- words from the gaps between letters
            val gaps = (0 until chars.size - 1).map { (chars[it + 1].l - chars[it].r).toDouble() }
            val thr = if (gaps.size >= 3) max(0.2 * capH, min(0.33 * capH, median(gaps) + 0.15 * capH)) else 0.33 * capH
            val groups = ArrayList<ArrayList<Ch>>()
            var cur = arrayListOf(chars[0])
            for (i in gaps.indices) {
                if (gaps[i] > thr) { groups.add(cur); cur = ArrayList() }
                cur.add(chars[i + 1])
            }
            groups.add(cur)
            val words = groups.map { wd ->
                val l = wd.minOf { it.l }.toDouble(); val r = wd.maxOf { it.r }.toDouble()
                val bl = m * l + b0; val br = m * r + b0
                Word(fixWord(String(wd.map { it.c }.toCharArray())), floatArrayOf(
                    l.toFloat(), (bl - capH * 1.05).toFloat(), r.toFloat(), (br - capH * 1.05).toFloat(),
                    r.toFloat(), (br + capH * 0.3).toFloat(), l.toFloat(), (bl + capH * 0.3).toFloat()))
            }
            out.add(words[0].quad[1].toDouble() to Line(words.joinToString(" ") { it.text }, 1f, words, lbox, capH.toFloat(), fit))
        }
        out.sortBy { it.first }
        return out.map { it.second }
    }

    /** Glue the pieces of one letter: ':' ';' '=' dots together, then i/j/!/? dots onto their stem. */
    private fun groupPieces(L: List<Comp>): List<Comp> {
        val items = L.map { Comp(it.l, it.t, it.r, it.b, it.ids.copyOf()) }
        val lh = median(items.map { it.h.toDouble() })
        val big = items.filter { it.h >= 0.6 * lh }
        val sml = items.filter { it.h < 0.6 * lh }.sortedBy { it.l }
        fun vdisjoint(a: Comp, b: Comp) = min(a.b, b.b) - max(a.t, b.t) <= 1
        fun xov(a: Comp, b: Comp) = min(a.r, b.r) - max(a.l, b.l)
        val ms = ArrayList<Comp>()
        for (c in sml) {
            val g = ms.firstOrNull { xov(it, c) > 0.4 * min(it.w, c.w) && vdisjoint(it, c) && abs(it.t - c.t) < 1.2 * lh }
            if (g != null) g.absorb(c) else ms.add(c)
        }
        val G = ArrayList(big)
        for (c in ms) {
            var best: Comp? = null; var bd = 1e9
            for (g in big) {
                if (c.cx >= g.l - 1 && c.cx <= g.r + 1 && vdisjoint(g, c) && min(abs(c.b - g.t), abs(c.t - g.b)) < 0.6 * lh) {
                    val d = abs(c.cx - g.cx)
                    if (d < bd) { bd = d; best = g }
                }
            }
            if (best != null && best.w < 0.7 * lh + 2) best.absorb(c) else G.add(c)
        }
        return G.sortedBy { it.l }
    }

    /** Pixels of [g] (only its own pieces) between columns [x0, x1). Returns null if empty. */
    private fun mask(g: Comp, x0: Int, x1: Int, labels: Mat): Pair<IntArray, Int>? {
        val w = x1 - x0; val h = g.h
        if (w <= 0 || h <= 0) return null
        val sub = labels.submat(g.t, g.b, x0, x1)
        val lab = IntArray(w * h)
        sub.get(0, 0, lab)
        sub.release()
        for (i in lab.indices) lab[i] = if (lab[i] in g.ids) 1 else 0
        return lab to w
    }

    /** First/last ink row of one piece of [g] (relative to g.t). */
    private fun inkRows(g: Comp, id: Int, labels: Mat): Pair<Int, Int>? {
        val sub = labels.submat(g.t, g.b, g.l, g.r)
        val lab = IntArray(g.w * g.h)
        sub.get(0, 0, lab); sub.release()
        var first = -1; var last = -1
        for (y in 0 until g.h) for (x in 0 until g.w) if (lab[y * g.w + x] == id) { if (first < 0) first = y; last = y; break }
        return if (first < 0) null else first to last
    }

    /** Feature vector + tight box (l, t, r, b in page px) of [g] cut to columns [x0, x1). */
    private fun feature(g: Comp, x0: Int, x1: Int, labels: Mat, m: Double, b0: Double, capH: Double): Pair<FloatArray, IntArray>? {
        val (mk, w) = mask(g, x0, x1, labels) ?: return null
        val h = g.h
        var minX = w; var maxX = -1; var minY = h; var maxY = -1
        for (y in 0 until h) for (x in 0 until w) if (mk[y * w + x] != 0) {
            if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y
        }
        if (maxX < 0) return null
        val cw = maxX - minX + 1; val chh = maxY - minY + 1
        val bytes = ByteArray(cw * chh)
        for (y in 0 until chh) for (x in 0 until cw) if (mk[(y + minY) * w + x + minX] != 0) bytes[y * cw + x] = 255.toByte()
        val img = Mat(chh, cw, CvType.CV_8U)
        img.put(0, 0, bytes)
        val f = FloatArray(GS * GS + 5)
        shapeVec(img, f)
        img.release()
        val l = x0 + minX; val t = g.t + minY; val r = x0 + maxX + 1; val b = g.t + maxY + 1
        val base = m * (l + r) / 2.0 + b0
        posVec(t.toDouble(), b.toDouble(), l.toDouble(), r.toDouble(), base, capH, f)
        return f to intArrayOf(l, t, r, b)
    }

    /** Cut a blob of touching letters at thin columns; keep the cut set whose pieces fit best. */
    private fun split(g: Comp, labels: Mat, m: Double, b0: Double, capH: Double): Pair<Float, List<Ch>>? {
        fun go(x0: Int, x1: Int, depth: Int): Pair<Float, List<Ch>>? {
            val fb = feature(g, x0, x1, labels, m, b0, capH) ?: return null
            val (ch, d) = classify(fb.first)
            var cost = d
            var res: List<Ch> = listOf(Ch(ch, fb.second[0], fb.second[1], fb.second[2], fb.second[3], d))
            if (depth < 3 && (x1 - x0) > 0.75 * capH) {
                val (mk, w) = mask(g, x0, x1, labels) ?: return cost to res
                val col = IntArray(w)
                for (y in 0 until g.h) for (x in 0 until w) col[x] += mk[y * w + x]
                val lo = (0.25 * capH).toInt(); val hi = w - (0.25 * capH).toInt()
                if (hi > lo) {
                    val cands = (lo until hi).sortedBy { col[it] }.take(3)
                    for (cx in cands) {
                        val a = go(x0, x0 + cx, depth + 1) ?: continue
                        val bb = go(x0 + cx, x1, depth + 1) ?: continue
                        val c2 = (a.first * a.second.size + bb.first * bb.second.size) / (a.second.size + bb.second.size)
                        if (c2 < cost * 0.75f) { cost = c2; res = a.second + bb.second }
                    }
                }
            }
            return cost to res
        }
        val o = go(g.l, g.r, 0) ?: return null
        return if (o.second.size <= 1) null else o
    }

    /** Shapes that look the same in many fonts (l/I/1/|, O/0, i/!) resolved from the word around them. */
    private fun fixWord(w: String): String {
        val core = w.trim { it in PUNCT }
        if (core.isEmpty()) return w
        val pre = w.substring(0, w.length - w.trimStart { it in PUNCT }.length)
        val post = w.substring(pre.length + core.length)
        val letters = core.count { it.isLetter() }; val digits = core.count { it.isDigit() }
        val lower = core.count { it.isLowerCase() }; val upper = core.count { it.isUpperCase() }
        val c = core.toCharArray()
        if (digits > letters) {                                     // a number: 4471, 12/03/2026
            for (i in c.indices) c[i] = when (c[i]) { 'O', 'o' -> '0'; 'l', 'I', '|' -> '1'; else -> c[i] }
        } else if (letters >= 2 && upper > lower && upper >= 2) {    // ALL CAPS word
            for (i in c.indices) c[i] = when (c[i]) { 'l', '|' -> 'I'; else -> c[i] }
            if (digits == 1) for (i in c.indices) c[i] = when (c[i]) { '1' -> 'I'; '0' -> 'O'; else -> c[i] }
        } else if (letters >= 2) {                                   // ordinary word
            for (i in c.indices) {
                if (c[i] in "I1|" && i > 0) c[i] = 'l'
                else if (c[i] == '!' && i < c.size - 1) c[i] = 'i'
                else if (c[i] == '0' && lower > 0) c[i] = 'o'
            }
            if (c[0] in "Il1|" && c.size > 1 && c[1].isLowerCase()) c[0] = if (c[1] in VOWELS) 'l' else 'I'
        } else if ((core == "l" || core == "|") && pre.isEmpty() && post.isEmpty()) {
            return "I"
        }
        return pre + String(c) + post
    }

    private fun median(v: List<Double>): Double {
        if (v.isEmpty()) return 0.0
        val s = v.sorted(); val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
    }

    /** Linear-interpolated percentile (numpy's default). */
    private fun percentile(v: List<Double>, p: Double): Double {
        val s = v.sorted()
        val pos = (s.size - 1) * p / 100.0
        val lo = pos.toInt(); val hi = min(lo + 1, s.size - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    }
}