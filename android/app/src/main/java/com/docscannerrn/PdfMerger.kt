package com.docscannerrn

import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.zip.Inflater

/**
 * Lossless PDF merger written from scratch (standard Java/Kotlin only, no PDF library).
 *
 * Every page is copied together with everything it uses (fonts, images, content streams,
 * annotations/links) byte for byte, so text stays sharp, selectable and searchable and the file
 * stays small. Pages are renumbered into one new page tree. Handles classic xref tables, xref
 * streams + compressed object streams (PDF 1.5+), incremental updates, and rebuilds the object
 * table by scanning when it is damaged.
 *
 * Not carried over: bookmarks, fillable-form logic (fields still show as drawn), document metadata.
 * Encrypted (password / permission-protected) PDFs are refused with [ProtectedPdfException].
 */
class PdfMerger {

    class ProtectedPdfException(val fileName: String) : Exception("$fileName is password-protected")
    class BadPdfException(val fileName: String, msg: String) : Exception("$fileName: $msg")

    // ======================================================================= object model

    /** Any token kept exactly as written: numbers, strings, true/false/null. */
    class PRaw(val b: ByteArray) {
        fun text() = String(b, Charsets.ISO_8859_1)
    }
    class PName(val n: String)
    class PArr(val items: MutableList<Any>)
    class PDict(val map: LinkedHashMap<String, Any>) {
        operator fun get(k: String) = map[k]
    }
    class PRef(val num: Int, val gen: Int)
    class PStream(val dict: PDict, val data: ByteArray)

    private val NULL = PRaw("null".toByteArray())

    // ======================================================================= lexer / parser

    private class Parser(val b: ByteArray, var p: Int, val lengthOf: (Any?) -> Int?) {
        fun isWs(c: Int) = c == 0 || c == 9 || c == 10 || c == 12 || c == 13 || c == 32
        fun isDelim(c: Int) = c == '('.code || c == ')'.code || c == '<'.code || c == '>'.code ||
            c == '['.code || c == ']'.code || c == '{'.code || c == '}'.code || c == '/'.code || c == '%'.code

        fun skipWs() {
            while (p < b.size) {
                val c = b[p].toInt() and 0xFF
                if (isWs(c)) p++
                else if (c == '%'.code) { while (p < b.size && b[p].toInt() != 10 && b[p].toInt() != 13) p++ }
                else break
            }
        }

        fun peek(): Int = if (p < b.size) b[p].toInt() and 0xFF else -1

        fun startsWith(s: String, at: Int = p): Boolean {
            if (at + s.length > b.size) return false
            for (i in s.indices) if (b[at + i].toInt() != s[i].code) return false
            return true
        }

        /** Next plain token (keyword or number) without consuming it. */
        fun readToken(): String {
            skipWs()
            val s = p
            while (p < b.size) {
                val c = b[p].toInt() and 0xFF
                if (isWs(c) || isDelim(c)) break
                p++
            }
            return String(b, s, p - s, Charsets.ISO_8859_1)
        }

        fun parse(depth: Int = 0): Any {
            if (depth > 200) throw IllegalStateException("nesting too deep")
            skipWs()
            if (p >= b.size) throw IllegalStateException("unexpected end of file")
            val c = peek()
            when {
                c == '/'.code -> {
                    p++
                    val s = p
                    while (p < b.size) {
                        val d = b[p].toInt() and 0xFF
                        if (isWs(d) || isDelim(d)) break
                        p++
                    }
                    return PName(String(b, s, p - s, Charsets.ISO_8859_1))
                }
                c == '<'.code && p + 1 < b.size && b[p + 1].toInt() == '<'.code -> {
                    p += 2
                    val m = LinkedHashMap<String, Any>()
                    while (true) {
                        skipWs()
                        if (p >= b.size) throw IllegalStateException("unterminated dictionary")
                        if (peek() == '>'.code && p + 1 < b.size && b[p + 1].toInt() == '>'.code) { p += 2; break }
                        val k = parse(depth + 1)
                        if (k !is PName) { // junk inside a dictionary: skip it
                            continue
                        }
                        skipWs()
                        if (peek() == '>'.code && p + 1 < b.size && b[p + 1].toInt() == '>'.code) { m[k.n] = PRaw("null".toByteArray()); p += 2; break }
                        m[k.n] = parse(depth + 1)
                    }
                    val dict = PDict(m)
                    // stream?
                    val save = p
                    skipWs()
                    if (startsWith("stream")) {
                        p += 6
                        if (peek() == 13) p++
                        if (peek() == 10) p++
                        val start = p
                        var len = lengthOf(m["Length"]) ?: -1
                        var end = start + len
                        // trust /Length only if "endstream" follows it
                        val ok = len >= 0 && end <= b.size && run {
                            var q = end
                            while (q < b.size && isWs(b[q].toInt() and 0xFF)) q++
                            startsWith("endstream", q)
                        }
                        if (!ok) {
                            val e = indexOf("endstream", start)
                            if (e < 0) throw IllegalStateException("stream without endstream")
                            end = e
                            if (end > start && b[end - 1].toInt() == 10) end--
                            if (end > start && b[end - 1].toInt() == 13) end--
                            len = end - start
                        }
                        p = end
                        skipWs()
                        if (startsWith("endstream")) p += 9
                        return PStream(dict, b.copyOfRange(start, start + len))
                    }
                    p = save
                    return dict
                }
                c == '<'.code -> {
                    val s = p
                    while (p < b.size && b[p].toInt() != '>'.code) p++
                    p++
                    return PRaw(b.copyOfRange(s, minOf(p, b.size)))
                }
                c == '['.code -> {
                    p++
                    val items = ArrayList<Any>()
                    while (true) {
                        skipWs()
                        if (p >= b.size) throw IllegalStateException("unterminated array")
                        if (peek() == ']'.code) { p++; break }
                        items.add(parse(depth + 1))
                    }
                    return PArr(items)
                }
                c == '('.code -> {
                    val s = p
                    var level = 0
                    while (p < b.size) {
                        val d = b[p].toInt()
                        if (d == '\\'.code) { p += 2; continue }
                        if (d == '('.code) level++
                        if (d == ')'.code) { level--; if (level == 0) { p++; break } }
                        p++
                    }
                    return PRaw(b.copyOfRange(s, minOf(p, b.size)))
                }
                c == ')'.code || c == '>'.code || c == ']'.code || c == '}'.code || c == '{'.code -> {
                    p++  // stray delimiter
                    return PRaw("null".toByteArray())
                }
                else -> {
                    val tok = readToken()
                    if (tok.isEmpty()) { p++; return PRaw("null".toByteArray()) }
                    // "num gen R" ?
                    if (tok.all { it.isDigit() }) {
                        val save = p
                        val t2 = readToken()
                        if (t2.isNotEmpty() && t2.all { it.isDigit() }) {
                            val t3 = readToken()
                            if (t3 == "R") return PRef(tok.toInt(), t2.toInt())
                        }
                        p = save
                    }
                    return PRaw(tok.toByteArray(Charsets.ISO_8859_1))
                }
            }
        }

        fun indexOf(s: String, from: Int): Int {
            var i = from
            val first = s[0].code
            while (i <= b.size - s.length) {
                if (b[i].toInt() == first && startsWith(s, i)) return i
                i++
            }
            return -1
        }
    }

    // ======================================================================= document loader

    private class Entry(val type: Int, val a: Int, val bIdx: Int)   // type 1: offset a; type 2: in objstm a at index bIdx

    private inner class Doc(val name: String, val b: ByteArray) {
        val xref = HashMap<Int, Entry>()
        var trailer: PDict? = null
        val cache = HashMap<Int, Any>()
        val objStms = HashMap<Int, Pair<IntArray, Pair<ByteArray, Int>>>()   // num -> (object numbers, (data, first))
        private val loading = HashSet<Int>()

        init {
            var ok = false
            try {
                readXref()
                ok = trailer?.get("Root") != null && xref.isNotEmpty()
            } catch (e: Exception) { ok = false }
            if (!ok) reconstruct()
            if (trailer?.get("Encrypt") != null) throw ProtectedPdfException(name)
        }

        fun num(o: Any?): Long? {
            val r = resolve(o)
            return if (r is PRaw) r.text().toDoubleOrNull()?.toLong() else null
        }

        private fun parser(at: Int) = Parser(b, at) { len -> num(len)?.toInt() }

        private fun readXref() {
            val tail = String(b, maxOf(0, b.size - 2048), minOf(2048, b.size), Charsets.ISO_8859_1)
            val i = tail.lastIndexOf("startxref")
            if (i < 0) throw IllegalStateException("no startxref")
            var off = tail.substring(i + 9).trim().takeWhile { it.isDigit() }.toIntOrNull() ?: throw IllegalStateException("bad startxref")
            val seen = HashSet<Int>()
            while (off in 0 until b.size && seen.add(off)) {
                val ps = parser(off)
                ps.skipWs()
                val tr: PDict
                if (ps.startsWith("xref")) {
                    ps.p += 4
                    while (true) {
                        ps.skipWs()
                        if (ps.startsWith("trailer")) { ps.p += 7; break }
                        val start = ps.readToken().toIntOrNull() ?: throw IllegalStateException("bad xref")
                        val count = ps.readToken().toIntOrNull() ?: throw IllegalStateException("bad xref")
                        for (k in 0 until count) {
                            val o = ps.readToken().toLongOrNull() ?: throw IllegalStateException("bad xref entry")
                            val g = ps.readToken().toIntOrNull() ?: 0
                            val t = ps.readToken()
                            val n = start + k
                            if (!xref.containsKey(n)) xref[n] = if (t == "n" && o > 0) Entry(1, o.toInt(), g) else Entry(0, 0, 0)
                        }
                    }
                    tr = ps.parse() as? PDict ?: throw IllegalStateException("bad trailer")
                    // hybrid file: extra xref stream
                    val xs = tr["XRefStm"]
                    if (xs is PRaw) xs.text().toIntOrNull()?.let { readXrefStream(it) }
                } else {
                    tr = readXrefStream(off)
                }
                if (trailer == null) trailer = tr
                off = (tr["Prev"] as? PRaw)?.text()?.toDoubleOrNull()?.toInt() ?: break
            }
        }

        private fun readXrefStream(off: Int): PDict {
            val ps = parser(off)
            ps.readToken(); ps.readToken()
            if (ps.readToken() != "obj") throw IllegalStateException("bad xref stream")
            val st = ps.parse() as? PStream ?: throw IllegalStateException("bad xref stream")
            val d = st.dict
            val data = decode(st)
            val w = (d["W"] as PArr).items.map { (it as PRaw).text().toInt() }
            val size = num(d["Size"])?.toInt() ?: 0
            val index = (d["Index"] as? PArr)?.items?.map { (it as PRaw).text().toInt() } ?: listOf(0, size)
            val rec = w.sum()
            var pos = 0
            var k = 0
            while (k + 1 < index.size) {
                val start = index[k]; val count = index[k + 1]
                for (j in 0 until count) {
                    if (pos + rec > data.size) break
                    fun field(i: Int, def: Long): Long {
                        if (w[i] == 0) return def
                        var v = 0L
                        for (q in 0 until w[i]) v = (v shl 8) or (data[pos + q].toLong() and 0xFF)
                        pos += w[i]
                        return v
                    }
                    val t = field(0, 1).toInt(); val f2 = field(1, 0); val f3 = field(2, 0)
                    val n = start + j
                    if (!xref.containsKey(n)) xref[n] = when (t) {
                        1 -> Entry(1, f2.toInt(), f3.toInt())
                        2 -> Entry(2, f2.toInt(), f3.toInt())
                        else -> Entry(0, 0, 0)
                    }
                }
                k += 2
            }
            return d
        }

        /** Damaged table: find every "n g obj" in the file (later definitions win). */
        private fun reconstruct() {
            xref.clear(); cache.clear()
            val re = Regex("""(?<![0-9])(\d{1,10})\s+(\d{1,5})\s+obj\b""")
            val text = String(b, Charsets.ISO_8859_1)
            for (m in re.findAll(text)) {
                xref[m.groupValues[1].toInt()] = Entry(1, m.range.first, m.groupValues[2].toInt())
            }
            // objects inside object streams
            for (n in xref.keys.toList()) {
                val o = try { getObj(n) } catch (e: Exception) { null }
                if (o is PStream && (o.dict["Type"] as? PName)?.n == "ObjStm") {
                    val nums = loadObjStm(n)?.first ?: continue
                    for ((i, on) in nums.withIndex()) if (!xref.containsKey(on)) xref[on] = Entry(2, n, i)
                }
            }
            // trailer: last "trailer" dictionary, else find the catalog
            var tr: PDict? = null
            var at = text.lastIndexOf("trailer")
            while (at >= 0 && tr == null) {
                tr = try { parser(at + 7).parse() as? PDict } catch (e: Exception) { null }
                if (tr?.get("Root") == null) { tr = null; at = text.lastIndexOf("trailer", at - 1) }
            }
            if (tr == null) {
                for (n in xref.keys.sorted()) {
                    val o = try { getObj(n) } catch (e: Exception) { null }
                    val d = (o as? PDict) ?: (o as? PStream)?.dict
                    if (d != null && (d["Encrypt"] != null || d["Root"] != null) && d["Type"].let { it is PName && it.n == "XRef" }) { tr = d; break }
                    if (o is PDict && (o["Type"] as? PName)?.n == "Catalog") {
                        tr = PDict(linkedMapOf("Root" to PRef(n, 0))); break
                    }
                }
            }
            trailer = tr ?: throw BadPdfException(name, "not a readable PDF")
        }

        private fun loadObjStm(n: Int): Pair<IntArray, Pair<ByteArray, Int>>? {
            objStms[n]?.let { return it }
            val st = getObj(n) as? PStream ?: return null
            val data = decode(st)
            val cnt = num(st.dict["N"])?.toInt() ?: return null
            val first = num(st.dict["First"])?.toInt() ?: return null
            val ps = Parser(data, 0) { null }
            val nums = IntArray(cnt); val offs = IntArray(cnt)
            for (i in 0 until cnt) {
                nums[i] = ps.readToken().toIntOrNull() ?: 0
                offs[i] = ps.readToken().toIntOrNull() ?: 0
            }
            val res = nums to (data to first)
            objStms[n] = res
            objStmOffsets[n] = offs
            return res
        }
        private val objStmOffsets = HashMap<Int, IntArray>()

        fun getObj(n: Int): Any {
            cache[n]?.let { return it }
            val e = xref[n] ?: return NULL
            if (!loading.add(n)) return NULL     // reference loop while resolving /Length
            try {
                val o: Any = when (e.type) {
                    1 -> {
                        val ps = parser(e.a)
                        ps.readToken(); ps.readToken()
                        if (ps.readToken() != "obj") NULL else ps.parse()
                    }
                    2 -> {
                        val stm = loadObjStm(e.a) ?: return NULL
                        val offs = objStmOffsets[e.a]!!
                        val idx = if (e.bIdx in offs.indices && stm.first[e.bIdx] == n) e.bIdx else stm.first.indexOf(n)
                        if (idx < 0) NULL else Parser(stm.second.first, stm.second.second + offs[idx]) { null }.parse()
                    }
                    else -> NULL
                }
                cache[n] = o
                return o
            } catch (ex: ProtectedPdfException) {
                throw ex
            } catch (ex: Exception) {
                return NULL
            } finally {
                loading.remove(n)
            }
        }

        fun resolve(o: Any?): Any? {
            var x = o; var guard = 0
            while (x is PRef && guard++ < 32) x = getObj(x.num)
            return x
        }

        /** Decoded data of an xref / object stream (FlateDecode with PNG/TIFF predictors). */
        fun decode(st: PStream): ByteArray {
            val f = resolve(st.dict["Filter"])
            val filters = when (f) { is PName -> listOf(f.n); is PArr -> f.items.mapNotNull { (resolve(it) as? PName)?.n }; else -> emptyList() }
            val parmsAny = resolve(st.dict["DecodeParms"])
            val parms = when (parmsAny) { is PDict -> listOf(parmsAny); is PArr -> parmsAny.items.map { resolve(it) as? PDict }; else -> emptyList() }
            var data = st.data
            for ((i, name) in filters.withIndex()) {
                data = when (name) {
                    "FlateDecode", "Fl" -> inflate(data)
                    else -> throw BadPdfException(this.name, "unsupported filter $name")
                }
                val pr = parms.getOrNull(i)
                if (pr != null) data = unpredict(data, pr)
            }
            return data
        }

        private fun unpredict(data: ByteArray, pr: PDict): ByteArray {
            val predictor = num(pr["Predictor"])?.toInt() ?: 1
            if (predictor < 2) return data
            val colors = num(pr["Colors"])?.toInt() ?: 1
            val bpc = num(pr["BitsPerComponent"])?.toInt() ?: 8
            val columns = num(pr["Columns"])?.toInt() ?: 1
            val bpp = maxOf(1, (colors * bpc + 7) / 8)
            val rowLen = (colors * bpc * columns + 7) / 8
            val out = ByteArrayOutputStream(data.size)
            if (predictor == 2) {   // TIFF, 8-bit only
                var p = 0
                while (p + rowLen <= data.size) {
                    val row = data.copyOfRange(p, p + rowLen)
                    for (i in bpp until rowLen) row[i] = (row[i] + row[i - bpp]).toByte()
                    out.write(row); p += rowLen
                }
                return out.toByteArray()
            }
            var prev = ByteArray(rowLen)
            var p = 0
            while (p < data.size) {
                val type = data[p].toInt() and 0xFF; p++
                val n = minOf(rowLen, data.size - p)
                val row = ByteArray(rowLen)
                System.arraycopy(data, p, row, 0, n); p += n
                for (i in 0 until rowLen) {
                    val a = if (i >= bpp) row[i - bpp].toInt() and 0xFF else 0
                    val up = prev[i].toInt() and 0xFF
                    val c = if (i >= bpp) prev[i - bpp].toInt() and 0xFF else 0
                    val x = row[i].toInt() and 0xFF
                    row[i] = when (type) {
                        1 -> x + a
                        2 -> x + up
                        3 -> x + (a + up) / 2
                        4 -> { val pp = a + up - c; val pa = Math.abs(pp - a); val pb = Math.abs(pp - up); val pc = Math.abs(pp - c)
                               x + if (pa <= pb && pa <= pc) a else if (pb <= pc) up else c }
                        else -> x
                    }.toByte()
                }
                out.write(row); prev = row
            }
            return out.toByteArray()
        }

        private fun inflate(data: ByteArray): ByteArray {
            val inf = Inflater()
            inf.setInput(data)
            val out = ByteArrayOutputStream(maxOf(64, data.size * 3))
            val buf = ByteArray(65536)
            try {
                while (!inf.finished()) {
                    val n = inf.inflate(buf)
                    if (n == 0) { if (inf.needsInput() || inf.needsDictionary()) break }
                    out.write(buf, 0, n)
                }
            } catch (e: java.util.zip.DataFormatException) {
                // keep what was decoded (damaged streams are common)
            } finally {
                inf.end()
            }
            return out.toByteArray()
        }
    }

    // ======================================================================= writer

    private class Out(file: File) {
        val os = BufferedOutputStream(FileOutputStream(file), 1 shl 16)
        var pos = 0L
        val offsets = ArrayList<Long>()  // index = object number - 1
        fun w(s: String) { val bb = s.toByteArray(Charsets.ISO_8859_1); os.write(bb); pos += bb.size }
        fun w(bb: ByteArray) { os.write(bb); pos += bb.size }
        fun setOffset(n: Int) {
            while (offsets.size < n) offsets.add(-1L)
            offsets[n - 1] = pos
        }
    }

    private fun write(o: Any, out: Out) {
        when (o) {
            is PRaw -> { out.w(" "); out.w(o.b) }
            is PName -> out.w("/" + o.n)
            is PRef -> out.w(" ${o.num} ${o.gen} R")
            is PArr -> { out.w("["); for (x in o.items) write(x, out); out.w("]") }
            is PDict -> { out.w("<<"); for ((k, v) in o.map) { out.w("/$k"); write(v, out) }; out.w(">>") }
            is PStream -> {
                o.dict.map["Length"] = PRaw(o.data.size.toString().toByteArray())
                write(o.dict, out)
                out.w("\nstream\n"); out.w(o.data); out.w("\nendstream")
            }
        }
    }

    // ======================================================================= merge

    private val INHERITED = listOf("Resources", "MediaBox", "CropBox", "Rotate")

    /**
     * Merges [inputs] (file, display name) in order into [output]. Returns the page count.
     * [onProgress] gets (file index, file count).
     */
    fun merge(inputs: List<Pair<File, String>>, output: File, onProgress: ((Int, Int) -> Unit)? = null): Int {
        output.parentFile?.mkdirs()
        val out = Out(output)
        var nextNum = 3                       // 1 = catalog, 2 = page tree root
        val kids = ArrayList<Int>()
        try {
            out.w("%PDF-1.7\n%âãÏÓ\n")
            for ((fi, input) in inputs.withIndex()) {
                onProgress?.invoke(fi, inputs.size)
                val doc = try { Doc(input.second, input.first.readBytes()) }
                          catch (e: ProtectedPdfException) { throw e }
                          catch (e: BadPdfException) { throw e }
                          catch (e: Exception) { throw BadPdfException(input.second, "can't be read (${e.message})") }

                // ---- the document's pages, in order, with inherited attributes
                val root = doc.resolve(doc.trailer?.get("Root")) as? PDict ?: throw BadPdfException(input.second, "has no page catalog")
                val leaves = ArrayList<Pair<Int, PDict>>()   // (old object number or -1, page dict)
                val seen = HashSet<Int>()
                fun walk(ref: Any?, inh: Map<String, Any>, depth: Int) {
                    if (depth > 64) return
                    val num = (ref as? PRef)?.num ?: -1
                    if (num >= 0 && !seen.add(num)) return
                    val node = doc.resolve(ref) as? PDict ?: return
                    val here = HashMap(inh)
                    for (k in INHERITED) node[k]?.let { here[k] = it }
                    val kidsArr = doc.resolve(node["Kids"]) as? PArr
                    val type = (node["Type"] as? PName)?.n
                    if (kidsArr != null && type != "Page") {
                        for (k in kidsArr.items) walk(k, here, depth + 1)
                    } else {
                        val page = PDict(LinkedHashMap(node.map))
                        for (k in INHERITED) if (page[k] == null) here[k]?.let { page.map[k] = it }
                        if (page["MediaBox"] == null) page.map["MediaBox"] = PArr(arrayListOf(PRaw("0".toByteArray()), PRaw("0".toByteArray()), PRaw("612".toByteArray()), PRaw("792".toByteArray())))
                        leaves.add(num to page)
                    }
                }
                walk(root["Pages"], emptyMap(), 0)
                if (leaves.isEmpty()) throw BadPdfException(input.second, "has no pages")

                // ---- copy each page and everything it references, renumbered
                val map = HashMap<Int, Int>()
                val queue = ArrayDeque<Int>()
                fun newNum(old: Int): Int = map.getOrPut(old) { queue.addLast(old); nextNum++ }
                fun isPageTreeNode(n: Int): Boolean {
                    val o = doc.getObj(n)
                    return o is PDict && (o["Type"] as? PName)?.n == "Pages"
                }
                fun xf(o: Any): Any = when (o) {
                    is PRef -> when {
                        doc.xref[o.num] == null -> NULL
                        isPageTreeNode(o.num) -> PRef(2, 0)      // never drag in the old page tree
                        else -> PRef(newNum(o.num), 0)
                    }
                    is PArr -> PArr(o.items.mapTo(ArrayList()) { xf(it) })
                    is PDict -> PDict(LinkedHashMap<String, Any>().also { m -> for ((k, v) in o.map) m[k] = xf(v) })
                    is PStream -> PStream(xf(o.dict) as PDict, o.data)
                    else -> o
                }
                fun writeObj(n: Int, o: Any) {
                    out.setOffset(n)
                    out.w("$n 0 obj\n")
                    write(o, out)
                    out.w("\nendobj\n")
                }
                fun drain() {
                    while (queue.isNotEmpty()) {
                        val old = queue.removeFirst()
                        writeObj(map[old]!!, xf(doc.getObj(old)))
                    }
                }
                for ((oldNum, page) in leaves) {
                    val n = if (oldNum >= 0) map.getOrPut(oldNum) { nextNum++ } else nextNum++
                    page.map["Parent"] = PRef(2, 0)
                    page.map["Type"] = PName("Page")
                    page.map.remove("B")
                    page.map.remove("Kids")
                    val copy = xf(PDict(LinkedHashMap(page.map.filterKeys { it != "Parent" }))) as PDict
                    copy.map["Parent"] = PRef(2, 0)
                    writeObj(n, copy)
                    kids.add(n)
                    drain()
                }
            }

            // ---- page tree, catalog, xref, trailer
            out.setOffset(2)
            out.w("2 0 obj\n<</Type/Pages/Count ${kids.size}/Kids[")
            for (k in kids) out.w("$k 0 R ")
            out.w("]>>\nendobj\n")
            out.setOffset(1)
            out.w("1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n")
            val total = nextNum - 1
            while (out.offsets.size < total) out.offsets.add(-1L)
            val xrefAt = out.pos
            val sb = StringBuilder()
            sb.append("xref\n0 ${total + 1}\n0000000000 65535 f\r\n")
            for (i in 0 until total) {
                val off = out.offsets[i]
                if (off < 0) sb.append("0000000000 00000 f\r\n") else sb.append(String.format(java.util.Locale.US, "%010d 00000 n\r\n", off))
            }
            sb.append("trailer\n<</Size ${total + 1}/Root 1 0 R>>\nstartxref\n$xrefAt\n%%EOF\n")
            out.w(sb.toString())
            out.os.flush()
        } finally {
            out.os.close()
        }
        return kids.size
    }
}