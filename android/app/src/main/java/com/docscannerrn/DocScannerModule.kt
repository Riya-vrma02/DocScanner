package com.docscannerrn

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.provider.MediaStore
import android.util.Base64
import androidx.core.content.FileProvider
import com.facebook.react.bridge.*
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Replaces three third-party npm packages with one native module built entirely
 * on Android's own stable APIs, plus OpenCV (official Maven artifact) and an
 * optional bundled TFLite model for curl correction:
 *   - react-native-document-scanner-plugin -> system camera Intent + OpenCV edge/perspective
 *   - @shopify/react-native-skia            -> android.graphics Bitmap/Canvas/ColorMatrix
 *   - react-native-fs                       -> plain java.io.File
 */
class DocScannerModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), ActivityEventListener {

    companion object {
        private const val CAPTURE_REQUEST_CODE = 9001
    }

    private var capturePromise: Promise? = null
    private var pendingCapturePath: String? = null


    init {
        reactContext.addActivityEventListener(this)
    }

    override fun getName(): String = "DocScannerModule"

    // ---------------------------------------------------------------------
    // 1. CAPTURE — launches the phone's own camera app, no camera library
    // ---------------------------------------------------------------------
    @ReactMethod
    fun capturePhoto(promise: Promise) {
        val activity: Activity? = currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "No current activity")
            return
        }

        val photoFile = File(reactContext.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(
            reactContext, "${reactContext.packageName}.fileprovider", photoFile
        )

        capturePromise = promise
        pendingCapturePath = photoFile.absolutePath

        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
            putExtra(MediaStore.EXTRA_OUTPUT, uri)
            addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        }

        if (intent.resolveActivity(reactContext.packageManager) == null) {
            promise.reject("NO_CAMERA_APP", "No camera app found on this device")
            capturePromise = null
            return
        }

        activity.startActivityForResult(intent, CAPTURE_REQUEST_CODE)
    }

    override fun onActivityResult(activity: Activity?, requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode != CAPTURE_REQUEST_CODE) return
        val promise = capturePromise ?: return
        capturePromise = null

        if (resultCode == Activity.RESULT_OK && pendingCapturePath != null) {
            promise.resolve(pendingCapturePath)
        } else {
            promise.reject("CAPTURE_CANCELLED", "Photo capture was cancelled or failed")
        }
        pendingCapturePath = null
    }

    override fun onNewIntent(intent: Intent?) {}

    // ---------------------------------------------------------------------
    // 2. DOCUMENT DETECTION — now finds MULTIPLE documents, and falls back
    //    to an irregular polygon (not forced to 4 points) for torn edges.
    // ---------------------------------------------------------------------
    @ReactMethod
    fun detectDocuments(imagePath: String, promise: Promise) {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image at $imagePath")

            val candidates = findDocumentCandidates(bitmap)

            val result = Arguments.createArray()
            if (candidates.isEmpty()) {
                // Nothing detected at all — fall back to full image bounds.
                val fallback = Arguments.createMap()
                val pts = Arguments.createArray()
                listOf(
                    floatArrayOf(0f, 0f), floatArrayOf(bitmap.width.toFloat(), 0f),
                    floatArrayOf(bitmap.width.toFloat(), bitmap.height.toFloat()), floatArrayOf(0f, bitmap.height.toFloat())
                ).forEach { p ->
                    val m = Arguments.createMap(); m.putDouble("x", p[0].toDouble()); m.putDouble("y", p[1].toDouble())
                    pts.pushMap(m)
                }
                fallback.putArray("points", pts)
                fallback.putBoolean("isQuad", true)
                fallback.putDouble("area", (bitmap.width * bitmap.height).toDouble())
                result.pushMap(fallback)
            } else {
                for (candidate in candidates) {
                    val m = Arguments.createMap()
                    val pts = Arguments.createArray()
                    candidate.points.forEach { p ->
                        val pm = Arguments.createMap(); pm.putDouble("x", p[0].toDouble()); pm.putDouble("y", p[1].toDouble())
                        pts.pushMap(pm)
                    }
                    m.putArray("points", pts)
                    m.putBoolean("isQuad", candidate.isQuad)
                    m.putDouble("area", candidate.area)
                    result.pushMap(m)
                }
            }

            val out = Arguments.createMap()
            out.putArray("documents", result)
            out.putInt("imageWidth", bitmap.width)
            out.putInt("imageHeight", bitmap.height)
            promise.resolve(out)
        } catch (e: Exception) {
            promise.reject("DETECT_FAILED", e.message, e)
        }
    }

    private data class DocCandidate(val points: List<FloatArray>, val isQuad: Boolean, val area: Double)

    /**
     * Finds every plausible document-shaped region in the frame (handles
     * multiple stacked/overlapping pages). For each large contour: tries a
     * clean 4-point approximation first (fast path for a flat, intact page);
     * if the shape doesn't reduce to 4 points cleanly (torn/ragged edge),
     * keeps the full contour polygon instead of forcing a bad quad fit.
     */
    private fun findDocumentCandidates(bitmap: Bitmap): List<DocCandidate> {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)
        // Close small gaps in torn/ragged edges so the contour stays connected.
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.dilate(edges, edges, Mat())

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imageArea = src.rows() * src.cols()
        val minArea = imageArea * 0.05 // smaller floor than before, so stacked/partial docs aren't skipped

        val candidates = mutableListOf<DocCandidate>()
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < minArea) continue

            val c2f = MatOfPoint2f(*c.toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true)

            if (approx.total() == 4L) {
                val pts = orderCorners(approx.toArray().map { floatArrayOf(it.x.toFloat(), it.y.toFloat()) })
                candidates.add(DocCandidate(pts, isQuad = true, area = area))
            } else {
                // Not a clean rectangle — likely a torn/ragged edge, or a curved
                // page whose outline isn't 4 straight sides. Keep the actual
                // contour shape (simplified a little for a manageable point count)
                // rather than forcing a bad 4-point crop that would cut into the page.
                val looser = MatOfPoint2f()
                Imgproc.approxPolyDP(c2f, looser, 0.005 * peri, true)
                val pts = looser.toArray().map { floatArrayOf(it.x.toFloat(), it.y.toFloat()) }
                if (pts.size >= 3) candidates.add(DocCandidate(pts, isQuad = false, area = area))
            }
        }

        gray.release(); edges.release(); hierarchy.release(); src.release()

        // Largest first — the UI can let the user pick if more than one is found.
        return candidates.sortedByDescending { it.area }
    }

    private fun orderCorners(points: List<FloatArray>): List<FloatArray> {
        val sums = points.map { it[0] + it[1] }
        val diffs = points.map { it[0] - it[1] }
        val tl = points[sums.indexOf(sums.min())]
        val br = points[sums.indexOf(sums.max())]
        val tr = points[diffs.indexOf(diffs.max())]
        val bl = points[diffs.indexOf(diffs.min())]
        return listOf(tl, tr, br, bl)
    }

    /** Straight 4-point perspective warp — for clean, flat, intact pages. */
    @ReactMethod
    fun perspectiveCorrect(imagePath: String, cornersJs: ReadableArray, outputPath: String, promise: Promise) {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image")

            val corners = (0 until cornersJs.size()).map {
                val m = cornersJs.getMap(it)!!
                Point(m.getDouble("x"), m.getDouble("y"))
            }
            require(corners.size == 4) { "Expected exactly 4 corners" }
            val (tl, tr, br, bl) = corners

            val widthTop = dist(tl, tr); val widthBottom = dist(bl, br)
            val maxWidth = max(widthTop, widthBottom).toInt().coerceAtLeast(1)
            val heightLeft = dist(tl, bl); val heightRight = dist(tr, br)
            val maxHeight = max(heightLeft, heightRight).toInt().coerceAtLeast(1)

            val src = Mat()
            Utils.bitmapToMat(bitmap, src)

            val srcPoints = MatOfPoint2f(tl, tr, br, bl)
            val dstPoints = MatOfPoint2f(
                Point(0.0, 0.0), Point((maxWidth - 1).toDouble(), 0.0),
                Point((maxWidth - 1).toDouble(), (maxHeight - 1).toDouble()), Point(0.0, (maxHeight - 1).toDouble())
            )

            val transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
            val dst = Mat()
            Imgproc.warpPerspective(src, dst, transform, Size(maxWidth.toDouble(), maxHeight.toDouble()))

            val result = Bitmap.createBitmap(maxWidth, maxHeight, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(dst, result)
            src.release(); dst.release(); transform.release()

            writeJpeg(result, outputPath)
            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("WARP_FAILED", e.message, e)
        }
    }

    /**
     * Crops to an IRREGULAR polygon (torn/ragged edge) instead of forcing a
     * rectangle. Masks out everything outside the shape (transparent), then
     * crops to its bounding box — preserves the actual torn boundary rather
     * than fabricating straight edges that would cut into or beyond the page.
     */
    @ReactMethod
    fun cropToPolygon(imagePath: String, pointsJs: ReadableArray, outputPath: String, promise: Promise) {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image")

            val points = (0 until pointsJs.size()).map {
                val m = pointsJs.getMap(it)!!
                Point(m.getDouble("x"), m.getDouble("y"))
            }

            val src = Mat()
            Utils.bitmapToMat(bitmap, src)

            val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
            val contourMat = MatOfPoint(*points.toTypedArray())
            Imgproc.fillPoly(mask, listOf(contourMat), Scalar(255.0))

            val masked = Mat()
            src.copyTo(masked, mask)

            val rect = Imgproc.boundingRect(contourMat)
            val cropped = Mat(masked, rect)

            val result = Bitmap.createBitmap(rect.width, rect.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(cropped, result)

            src.release(); mask.release(); masked.release(); cropped.release()

            writeJpeg(result, outputPath)
            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("CROP_FAILED", e.message, e)
        }
    }

    private fun dist(a: Point, b: Point): Double {
        val dx = a.x - b.x; val dy = a.y - b.y
        return Math.sqrt(dx * dx + dy * dy)
    }

    private operator fun <T> List<T>.component4(): T = this[3]

    // ---------------------------------------------------------------------
    // 3. CURL / WARP CORRECTION — ML-based (DewarpNet TFLite model)
    // ---------------------------------------------------------------------
    

    // ---------------------------------------------------------------------
    // 4. FILTERS + ROTATION — plain android.graphics, no Skia
    // ---------------------------------------------------------------------
    @ReactMethod
    fun applyFilter(
        imagePath: String, filterType: String, rotationDegrees: Int,
        brightness: Double, contrast: Double, outputPath: String, promise: Promise
    ) {
        try {
            var bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image")

            if (rotationDegrees % 360 != 0) {
                val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }

            val matrix = ColorMatrix()
            when (filterType) {
                "GRAYSCALE" -> matrix.setSaturation(0f)
                "BLACK_AND_WHITE" -> {
                    matrix.setSaturation(0f)
                    val highContrast = ColorMatrix(floatArrayOf(
                        3f, 0f, 0f, 0f, -400f,
                        0f, 3f, 0f, 0f, -400f,
                        0f, 0f, 3f, 0f, -400f,
                        0f, 0f, 0f, 1f, 0f
                    ))
                    matrix.postConcat(highContrast)
                }
                "AUTO_ENHANCE" -> {
                    val enhance = ColorMatrix(floatArrayOf(
                        1.2f, 0f, 0f, 0f, 10f,
                        0f, 1.2f, 0f, 0f, 10f,
                        0f, 0f, 1.2f, 0f, 10f,
                        0f, 0f, 0f, 1f, 0f
                    ))
                    matrix.postConcat(enhance)
                }
                else -> {}
            }

            if (brightness != 0.0 || contrast != 0.0) {
                val c = (1 + contrast * 0.5).toFloat()
                val b = (brightness * 100).toFloat()
                val bc = ColorMatrix(floatArrayOf(
                    c, 0f, 0f, 0f, b,
                    0f, c, 0f, 0f, b,
                    0f, 0f, c, 0f, b,
                    0f, 0f, 0f, 1f, 0f
                ))
                matrix.postConcat(bc)
            }

            val output = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(output)
            val paint = Paint().apply { colorFilter = ColorMatrixColorFilter(matrix) }
            canvas.drawBitmap(bitmap, 0f, 0f, paint)

            writeJpeg(output, outputPath)
            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("FILTER_FAILED", e.message, e)
        }
    }

    private fun writeJpeg(bitmap: Bitmap, path: String) {
        val file = File(path)
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }
    }

    // ---------------------------------------------------------------------
    // 5. FILE HELPERS — plain java.io.File, no react-native-fs
    // ---------------------------------------------------------------------
    @ReactMethod
    fun mkdir(path: String, promise: Promise) {
        try { File(path).mkdirs(); promise.resolve(true) }
        catch (e: Exception) { promise.reject("MKDIR_FAILED", e.message, e) }
    }

    @ReactMethod
    fun readFileBase64(path: String, promise: Promise) {
        try { promise.resolve(Base64.encodeToString(File(path).readBytes(), Base64.NO_WRAP)) }
        catch (e: Exception) { promise.reject("READ_FAILED", e.message, e) }
    }

    @ReactMethod
    fun deleteFile(path: String, promise: Promise) {
        try { File(path).delete(); promise.resolve(true) }
        catch (e: Exception) { promise.reject("DELETE_FAILED", e.message, e) }
    }

    @ReactMethod
    fun getDocumentDirectory(promise: Promise) { promise.resolve(reactContext.filesDir.absolutePath) }

    @ReactMethod
    fun getCacheDirectory(promise: Promise) { promise.resolve(reactContext.cacheDir.absolutePath) }

    // ---------------------------------------------------------------------
    // 6. PDF EXPORT — android.graphics.pdf.PdfDocument
    // ---------------------------------------------------------------------
    @ReactMethod
    fun exportToPdf(imagePathsJs: ReadableArray, pageSizeName: String, outputPath: String, promise: Promise) {
        try {
            val document = PdfDocument()
            val (pageWidthPt, pageHeightPt) = pointDimensions(pageSizeName)

            for (i in 0 until imagePathsJs.size()) {
                val path = imagePathsJs.getString(i)!!
                val bitmap = BitmapFactory.decodeFile(path)

                val isLandscape = bitmap.width > bitmap.height
                val pageW = if (isLandscape) max(pageWidthPt, pageHeightPt) else min(pageWidthPt, pageHeightPt)
                val pageH = if (isLandscape) min(pageWidthPt, pageHeightPt) else max(pageWidthPt, pageHeightPt)

                val pageInfo = PdfDocument.PageInfo.Builder(pageW, pageH, i + 1).create()
                val page = document.startPage(pageInfo)

                val scale = min(pageW.toFloat() / bitmap.width, pageH.toFloat() / bitmap.height)
                val drawW = bitmap.width * scale
                val drawH = bitmap.height * scale
                val left = (pageW - drawW) / 2f
                val top = (pageH - drawH) / 2f

                page.canvas.drawColor(Color.WHITE)
                val destRect = RectF(left, top, left + drawW, top + drawH)
                page.canvas.drawBitmap(bitmap, null, destRect, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))

                document.finishPage(page)
                bitmap.recycle()
            }

            val outFile = File(outputPath)
            outFile.parentFile?.mkdirs()
            FileOutputStream(outFile).use { document.writeTo(it) }
            document.close()

            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("PDF_EXPORT_FAILED", e.message, e)
        }
    }

    private fun pointDimensions(pageSizeName: String): Pair<Int, Int> = when (pageSizeName) {
        "A4" -> 595 to 842
        "LEGAL" -> 612 to 1008
        "LETTER" -> 612 to 792
        else -> 595 to 842
    }

    // ---------------------------------------------------------------------
    // 6. MULTI-DOCUMENT DETECTION + TORN/IRREGULAR-EDGE CROPPING
    // ---------------------------------------------------------------------
    private data class DetectedDoc(val points: List<FloatArray>, val isQuadrilateral: Boolean, val area: Double)

    /**
     * Finds EVERY significant document-shaped region in the photo, not just
     * the single largest one — handles multiple pages photographed stacked
     * or side by side in one shot. Each result also reports whether it's a
     * clean 4-point quadrilateral (use perspectiveCorrect on it as normal)
     * or an irregular/torn shape (use cropToContour instead, since there's
     * no well-defined "4 corners" to perspective-warp for a torn edge).
     */
    @ReactMethod
    fun detectAllDocuments(imagePath: String, promise: Promise) {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image at $imagePath")

            val docs = findAllDocuments(bitmap)

            val docsArray = Arguments.createArray()
            for (doc in docs) {
                val docMap = Arguments.createMap()
                docMap.putBoolean("isQuadrilateral", doc.isQuadrilateral)
                docMap.putDouble("area", doc.area)
                val pointsArray = Arguments.createArray()
                for (p in doc.points) {
                    val pm = Arguments.createMap()
                    pm.putDouble("x", p[0].toDouble())
                    pm.putDouble("y", p[1].toDouble())
                    pointsArray.pushMap(pm)
                }
                docMap.putArray("points", pointsArray)
                docsArray.pushMap(docMap)
            }

            val out = Arguments.createMap()
            out.putArray("documents", docsArray)
            out.putInt("imageWidth", bitmap.width)
            out.putInt("imageHeight", bitmap.height)
            promise.resolve(out)
        } catch (e: Exception) {
            promise.reject("DETECT_FAILED", e.message, e)
        }
    }

    private fun findAllDocuments(bitmap: Bitmap): List<DetectedDoc> {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)
        // Morphological closing bridges small gaps along a torn/ragged edge
        // that a plain Canny output would otherwise leave as broken lines.
        val closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, closeKernel)
        Imgproc.dilate(edges, edges, Mat())

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imageArea = src.rows() * src.cols()
        // Lower threshold than the single-document case (0.2 -> 0.05): a stacked
        // or side-by-side page can legitimately occupy a smaller fraction of frame.
        val minAreaThreshold = imageArea * 0.05

        val candidates = contours
            .map { it to Imgproc.contourArea(it) }
            .filter { it.second > minAreaThreshold }
            .sortedByDescending { it.second }

        val results = mutableListOf<DetectedDoc>()
        val usedRects = mutableListOf<org.opencv.core.Rect>()

        for ((contour, area) in candidates) {
            val rect = Imgproc.boundingRect(contour)

            // Skip contours that substantially overlap one already accepted —
            // avoids double-counting the same page's inner and outer edge.
            val overlapsExisting = usedRects.any { existing ->
                val interW = (min(existing.x + existing.width, rect.x + rect.width) - max(existing.x, rect.x))
                val interH = (min(existing.y + existing.height, rect.y + rect.height) - max(existing.y, rect.y))
                val interArea = interW.coerceAtLeast(0) * interH.coerceAtLeast(0)
                interArea > 0.5 * min(existing.width * existing.height, rect.width * rect.height)
            }
            if (overlapsExisting) continue

            val c2f = MatOfPoint2f(*contour.toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx4 = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx4, 0.02 * peri, true)

            if (approx4.total() == 4L) {
                val quadArea = Imgproc.contourArea(approx4)
                // A torn/irregular shape can sometimes still approx down to 4
                // points, but loses real area doing so — only trust it as a
                // clean quadrilateral if the 4-point simplification keeps at
                // least 85% of the actual contour's area.
                if (quadArea > area * 0.85) {
                    val pts = approx4.toArray().map { floatArrayOf(it.x.toFloat(), it.y.toFloat()) }
                    results.add(DetectedDoc(orderCorners(pts), true, area))
                    usedRects.add(rect)
                    continue
                }
            }

            // Irregular/torn edge: keep a more detailed polygon (smaller
            // epsilon than the quad attempt) instead of forcing 4 points.
            val approxDetailed = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approxDetailed, 0.005 * peri, true)
            val pts = approxDetailed.toArray().map { floatArrayOf(it.x.toFloat(), it.y.toFloat()) }
            results.add(DetectedDoc(pts, false, area))
            usedRects.add(rect)
        }

        gray.release(); edges.release(); hierarchy.release(); src.release()
        return results
    }

    /**
     * Crops to an arbitrary (possibly non-convex, torn-edge) polygon instead
     * of perspective-warping a quadrilateral: masks out everything outside
     * the polygon to white, then crops to its bounding box. Use this for any
     * document detectAllDocuments() reported as isQuadrilateral: false.
     */
    @ReactMethod
    fun cropToContour(imagePath: String, pointsJs: ReadableArray, outputPath: String, promise: Promise) {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image")

            val points = (0 until pointsJs.size()).map {
                val m = pointsJs.getMap(it)!!
                Point(m.getDouble("x"), m.getDouble("y"))
            }

            val src = Mat()
            Utils.bitmapToMat(bitmap, src)

            val mask = Mat.zeros(src.size(), CvType.CV_8UC1)
            val matOfPoint = MatOfPoint(*points.toTypedArray())
            // fillPoly (not fillConvexPoly) since a torn edge is often non-convex.
            Imgproc.fillPoly(mask, listOf(matOfPoint), org.opencv.core.Scalar(255.0))

            val whiteBg = Mat(src.size(), src.type(), org.opencv.core.Scalar(255.0, 255.0, 255.0, 255.0))
            src.copyTo(whiteBg, mask)

            val rect = Imgproc.boundingRect(matOfPoint)
            val cropped = Mat(whiteBg, rect)

            val result = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(cropped, result)

            writeJpeg(result, outputPath)
            mask.release(); whiteBg.release(); src.release(); cropped.release()
            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("CROP_FAILED", e.message, e)
        }
    }
}
