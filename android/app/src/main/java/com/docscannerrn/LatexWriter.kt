package com.docscannerrn

import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Turns pages read by [OcrEngine] into a LaTeX document.
 *
 * Layout rules (single-column pages):
 *  - A line clearly bigger than the body text is a heading: the biggest size -> \section*, smaller -> \subsection*.
 *  - Lines starting with a bullet (•, -, *, ...) -> itemize; with "1." "a)" "(iv)" -> enumerate (labels kept).
 *  - Paragraphs break at a bigger gap, a first-line indent, or after a short line ending in . : ! ?
 *  - Pictures, charts, equations and anything the OCR can't read are cut out of the page and included
 *    as images (\includegraphics), so nothing is lost.
 */
class LatexWriter(private val figDir: File) {

    private sealed class Block {
        class Heading(val size: Double, val text: String) : Block()
        class Para(val lines: MutableList<String>) : Block()
        class Item(val marker: String, val lines: MutableList<String>, val textX: Double) : Block()   // marker "bullet" or label
        class Figure(val file: String, val frac: Double) : Block()
    }

    private val blocks = ArrayList<Block>()
    /** File names of the figures written so far (inside figDir). */
    val figures = ArrayList<String>()

    companion object {
        private const val BULLETS = ".·•*-–o»e"
        private val NUM_RE = Regex("""^(\(?\d{1,2}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])$""")
        private const val HEADING = 1.25        // line cap height / body cap height
    }

    /** Adds one page. [bitmap] is the page image the OCR ran on (figures are cut from it). */
    fun addPage(page: OcrEngine.Page, bitmap: Bitmap) {
        val lines = page.lines
        if (lines.isEmpty() && page.nonText.isEmpty()) return
        val caps = lines.filter { it.text.length >= 8 }.map { it.capH.toDouble() }
            .ifEmpty { lines.map { it.capH.toDouble() } }.ifEmpty { listOf(20.0) }
        val body = median(caps)
        val pad = 0.8 * body

        // ---- zones: items whose vertical ranges overlap (non-text boxes padded) chain into one zone
        class It(val line: OcrEngine.Line?, val box: IntArray, val t: Double, val b: Double)
        val items = ArrayList<It>()
        for (x in lines) items.add(It(x, x.box, x.box[1].toDouble(), x.box[3].toDouble()))
        for (n in page.nonText) items.add(It(null, n, n[1] - pad, n[3] + pad))
        items.sortBy { it.t }
        class Zone(var b: Double, val items: MutableList<It>)
        val zones = ArrayList<Zone>()
        for (it in items) {
            val z = zones.lastOrNull()
            if (z != null && it.t < z.b - 1) { z.items.add(it); z.b = max(z.b, it.b) } else zones.add(Zone(it.b, arrayListOf(it)))
        }

        val bodyLines = lines.filter { it.capH < HEADING * body }
        val left = bodyLines.minOfOrNull { it.box[0] }?.toDouble() ?: 0.0
        val right = bodyLines.maxOfOrNull { it.box[2] }?.toDouble() ?: page.width.toDouble()
        val tw = max(1.0, right - left)

        // ---- reading order: text lines and figures
        val stream = ArrayList<Any>()   // OcrEngine.Line or Block.Figure
        for (z in zones) {
            val tl = z.items.mapNotNull { it.line }
            val hasNonText = z.items.any { it.line == null }
            val poor = tl.any { x -> x.fit > 8 || x.text.count { it.isLetter() } < 4 }
            if (hasNonText || (z.items.size >= 2 && poor)) {
                val m = (0.4 * body).toInt()
                val x0 = max(0, z.items.minOf { it.box[0] } - m)
                val y0 = max(0, z.items.minOf { it.box[1] } - m)
                val x1 = min(bitmap.width, z.items.maxOf { it.box[2] } + m)
                val y1 = min(bitmap.height, z.items.maxOf { it.box[3] } + m)
                if (x1 - x0 < 4 || y1 - y0 < 4) continue
                val name = "fig${figures.size + 1}.png"
                val crop = Bitmap.createBitmap(bitmap, x0, y0, x1 - x0, y1 - y0)
                figDir.mkdirs()
                FileOutputStream(File(figDir, name)).use { crop.compress(Bitmap.CompressFormat.PNG, 100, it) }
                if (crop != bitmap) crop.recycle()
                figures.add(name)
                stream.add(Block.Figure(name, min(1.0, (x1 - x0) / tw)))
            } else {
                stream.addAll(tl.sortedWith(compareBy({ it.box[1] }, { it.box[0] })))
            }
        }

        // typical gap between consecutive body lines
        val gaps = ArrayList<Double>()
        var p: OcrEngine.Line? = null
        for (s in stream) {
            if (s is OcrEngine.Line) {
                val q = p
                if (q != null && s.capH < HEADING * body && q.capH < HEADING * body) gaps.add((s.box[1] - q.box[3]).toDouble())
                p = s
            } else p = null
        }
        val lead = if (gaps.isNotEmpty()) median(gaps) else body

        // ---- structure
        var cur: Block? = null
        var prev: OcrEngine.Line? = null
        val pageBlocks = ArrayList<Block>()
        fun close() { cur?.let { pageBlocks.add(it) }; cur = null }
        for (s in stream) {
            if (s is Block.Figure) { close(); pageBlocks.add(s); prev = null; continue }
            val x = s as OcrEngine.Line
            val words = x.text.split(' ')
            if (x.capH >= HEADING * body && x.text.length < 90) {
                close(); pageBlocks.add(Block.Heading(x.capH / body, x.text)); prev = x; continue
            }
            var marker: String? = null
            if (words.size >= 2 && x.box[0] <= left + 4 * body) {
                if (words[0].length == 1 && words[0][0] in BULLETS) marker = "bullet"
                else if (NUM_RE.matches(words[0])) {
                    marker = words[0]
                    // "l." / "I." at the start of a numbered list is "1."
                    if (marker.length == 2 && marker[0] in "lI|" && marker[1] in ".)") marker = "1" + marker[1]
                }
            }
            val gap = prev?.let { (x.box[1] - it.box[3]).toDouble() } ?: 1e9
            if (marker != null) {
                close()
                cur = Block.Item(marker, arrayListOf(words.drop(1).joinToString(" ")), x.box[0] + body)
                prev = x; continue
            }
            val c = cur
            if (c is Block.Item) {
                if (x.box[0] > c.textX - 0.5 * body && gap < 1.6 * lead) { c.lines.add(x.text); prev = x; continue }
                close()
            }
            val pv = prev
            val newPar = cur == null || gap > 1.6 * lead || x.box[0] > left + 1.2 * body ||
                (pv != null && pv.box[2] < left + 0.8 * tw && pv.text.trimEnd().lastOrNull()?.let { it in ".:!?" } == true)
            if (newPar) { close(); cur = Block.Para(arrayListOf(x.text)) } else (cur as Block.Para).lines.add(x.text)
            prev = x
        }
        close()

        // A paragraph cut by the page break continues on the next page.
        val last = blocks.lastOrNull(); val first = pageBlocks.firstOrNull()
        if (last is Block.Para && first is Block.Para &&
            last.lines.last().trimEnd().lastOrNull()?.let { it in ".:!?" } != true) {
            last.lines.addAll(first.lines); pageBlocks.removeAt(0)
        }
        blocks.addAll(pageBlocks)
    }

    /** The whole document as LaTeX source. Figures are referenced as "<figPrefix><name>". */
    fun toLatex(figPrefix: String = ""): String {
        val sizes = blocks.filterIsInstance<Block.Heading>().map { it.size }
        val biggest = sizes.maxOrNull() ?: 0.0
        fun level(sz: Double) = if (sz >= biggest - 0.12 || sz >= 1.6) "section" else "subsection"
        val out = StringBuilder()
        out.append("\\documentclass[11pt]{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n")
        out.append("\\usepackage[margin=2.5cm]{geometry}\n")
        if (figures.isNotEmpty()) out.append("\\usepackage{graphicx}\n")
        out.append("\\usepackage{parskip}\n\n\\begin{document}\n\n")
        var i = 0
        while (i < blocks.size) {
            when (val b = blocks[i]) {
                is Block.Heading -> out.append("\\${level(b.size)}*{${esc(b.text)}}\n\n")
                is Block.Para -> out.append(esc(joinLines(b.lines))).append("\n\n")
                is Block.Figure -> out.append("\\begin{center}\n  \\includegraphics[width=")
                    .append(String.format(java.util.Locale.US, "%.2f", b.frac))
                    .append("\\linewidth]{").append(figPrefix).append(b.file).append("}\n\\end{center}\n\n")
                is Block.Item -> {
                    val bullet = b.marker == "bullet"
                    val env = if (bullet) "itemize" else "enumerate"
                    out.append("\\begin{$env}\n")
                    while (i < blocks.size) {
                        val it = blocks[i]
                        if (it !is Block.Item || (it.marker == "bullet") != bullet) break
                        out.append("  \\item")
                        if (!bullet) out.append("[").append(esc(it.marker)).append("]")
                        out.append(" ").append(esc(joinLines(it.lines))).append("\n")
                        i++
                    }
                    out.append("\\end{$env}\n\n")
                    continue
                }
            }
            i++
        }
        out.append("\\end{document}\n")
        return out.toString()
    }

    /** Joins a paragraph's lines, undoing end-of-line hyphenation ("aggre-" + "gation"). */
    private fun joinLines(ls: List<String>): String {
        val sb = StringBuilder()
        for (raw in ls) {
            val s = raw.trim()
            if (sb.isEmpty()) { sb.append(s); continue }
            if (sb.endsWith("-") && s.firstOrNull()?.isLowerCase() == true && sb.length > 1 && sb[sb.length - 2].isLetter()) {
                sb.setLength(sb.length - 1); sb.append(s)
            } else sb.append(' ').append(s)
        }
        return sb.toString()
    }

    private fun esc(s: String): String {
        val sb = StringBuilder()
        for (ch in s) sb.append(when (ch) {
            '\\' -> "\\textbackslash{}"; '{' -> "\\{"; '}' -> "\\}"; '$' -> "\\$"; '&' -> "\\&"; '#' -> "\\#"
            '^' -> "\\^{}"; '_' -> "\\_"; '%' -> "\\%"; '~' -> "\\~{}"; '"' -> "''"
            '<' -> "\\textless{}"; '>' -> "\\textgreater{}"
            else -> ch.toString()
        })
        return sb.toString()
    }

    private fun median(v: List<Double>): Double {
        if (v.isEmpty()) return 0.0
        val s = v.sorted(); val n = s.size
        return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}