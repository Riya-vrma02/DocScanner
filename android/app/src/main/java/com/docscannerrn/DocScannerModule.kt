package com.docscannerrn

import android.app.Activity
import android.content.ContentValues
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
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import androidx.core.content.FileProvider
import com.facebook.react.bridge.*
import org.opencv.android.Utils
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
import org.opencv.photo.Photo
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

// ---- shadow-removal tuning (see removeShadowsFrom) ----
/** Ink is darkened only below this fraction of its local background (x255). */
private const val INK_KNEE = 230.0
/** Ink-darkening strength below the knee (1.0 = off, higher = darker text). */
private const val INK_GAMMA = 1.4
/** Lab colour distance from the paper colour above which a region counts as coloured content. */
private const val CHROMA_T = 10.0
/** Max chromaticity spread for a region to count as a FLAT colour (box, highlight, band). */
private const val FLAT_CHROM = 0.03
/**
 * A colour region whose colour changes less than this (max channel, 0..255) across its own edge is a
 * soft-edged SHADOW (its edge lies inside a gradual penumbra), not printed colour, which has a crisp step.
 */
private const val SOFT_STEP = 8.0
/**
 * Ink-removal kernel = long side / this. Must be wider than pen strokes but narrower than thin
 * shadows (a cable's shadow is only a few % of the page wide). Lower = removes thicker strokes.
 */
private const val INK_DIV = 120
/**
 * Contrast of the default "Auto" filter, stretched around white (1.0 = none). Paper stays white,
 * ink gets darker as this rises. Lower = softer, more natural-looking scan.
 */
private const val AUTO_CONTRAST = 1.1f
/** Saturation applied after the Auto stretch to cancel the colour boost it causes (1.0 = none). */
private const val AUTO_SATURATION = 0.87f

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

    /** ML page segmentation (paper_seg.onnx); falls back to CV when unavailable. */
    private val paperSegmenter by lazy { PaperSegmenter(reactContext.applicationContext) }


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
            val path = pendingCapturePath!!
            try {
                // Bake the camera's EXIF orientation into the actual pixels.
                // BitmapFactory.decodeFile (used everywhere downstream for OpenCV
                // detection + perspective warp) ignores EXIF, while RN's <Image>
                // honors it — so without this, detection runs in a rotated
                // coordinate space that doesn't match what the user sees in the
                // crop UI, and the document never gets cropped correctly.
                normalizeOrientation(path)
            } catch (e: Exception) {
                // Non-fatal: if normalization fails, fall back to the raw capture.
            }
            promise.resolve(path)
        } else {
            promise.reject("CAPTURE_CANCELLED", "Photo capture was cancelled or failed")
        }
        pendingCapturePath = null
    }

    /**
     * Reads the JPEG's EXIF orientation flag, physically rotates/flips the
     * pixels so the image is upright, rewrites the file, and resets the flag
     * to "normal". This guarantees a single, consistent orientation for OpenCV
     * detection, the crop UI, and the perspective warp.
     */
    private fun normalizeOrientation(path: String) {
        val exif = ExifInterface(path)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
        )
        if (orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED
        ) {
            return
        }

        val bitmap = BitmapFactory.decodeFile(path) ?: return
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.postRotate(90f); matrix.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.postRotate(270f); matrix.postScale(-1f, 1f) }
            else -> return
        }

        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        writeJpeg(rotated, path)
        if (rotated != bitmap) bitmap.recycle()

        // Clear the flag so nothing rotates the already-upright pixels a second time.
        ExifInterface(path).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL.toString())
            saveAttributes()
        }
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

            val incoming = (0 until cornersJs.size()).map {
                val m = cornersJs.getMap(it)!!
                Point(m.getDouble("x"), m.getDouble("y"))
            }
            require(incoming.size == 4) { "Expected exactly 4 corners" }

            // Normalize the corner order (clockwise, starting top-left) before
            // warping. The caller's order can't be trusted: dragging a handle
            // past another in the crop UI reverses the winding, and a reversed
            // winding against the fixed destination rectangle makes
            // getPerspectiveTransform emit a MIRRORED page.
            val corners = DocumentDetector.orderCorners(incoming.toTypedArray()).toList()
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

            // Every cropped page is cleaned once here, before it reaches the filter screen.
            writeJpeg(removeShadowsFrom(result), outputPath)
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

            // Every cropped page is cleaned once here, before it reaches the filter screen.
            writeJpeg(removeShadowsFrom(result), outputPath)
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
                    // The cropped page has already had its shadows removed, so its paper is pure
                    // white; brightening would only wash out light text. Instead stretch contrast
                    // AROUND WHITE:  out = 255 - (255 - in) * c  ->  paper stays 255, ink darker.
                    val c = AUTO_CONTRAST
                    val o = 255f * (1f - c)
                    val enhance = ColorMatrix(floatArrayOf(
                        c, 0f, 0f, 0f, o,
                        0f, c, 0f, 0f, o,
                        0f, 0f, c, 0f, o,
                        0f, 0f, 0f, 1f, 0f
                    ))
                    matrix.postConcat(enhance)
                    // The contrast stretch above also boosts saturation; pull it back so colours stay true.
                    matrix.postConcat(ColorMatrix().apply { setSaturation(AUTO_SATURATION) })
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

    // ---------------------------------------------------------------------
    // SHADOW REMOVAL (colour-preserving) — runs once on every cropped page
    // ---------------------------------------------------------------------
    /**
     * Removes shadows and uneven lighting from a page photo while keeping its COLOURS.
     *
     * Model: image = lighting x page, where the lighting has a brightness AND a colour (shadows are
     * often tinted: light bounced off a hand, a table or warm room light). Paper is white, so on bare
     * paper the lighting is simply the paper's own colour; under coloured content the lighting colour
     * is taken from the paper around it. Dividing by that removes shadows and their colour casts while
     * keeping printed colours.
     *
     *  1. Text-free background: on a copy downscaled to <= 1024 px, a morphological CLOSE with a disc
     *     wider than a text stroke erases dark ink; a median filter removes specks.
     *  2. Content mask: background pixels whose Lab colour differs from the paper colour (tinted boxes,
     *     highlights, photos, colour bands) or that are very dark. Soft-edged regions are dropped: a
     *     tinted shadow differs in colour too, but its edge is a gradual penumbra with no step across
     *     it. Regions that nearly fill their bounding rectangle are filled as rectangles.
     *  3. Lighting map:
     *     - bare paper: its own brightness;
     *     - under colour: the BRIGHTEST surrounding paper (grey dilation = max filter), so coloured
     *       areas are never brightened more than the lit paper beside them (no blow-outs);
     *     - FLAT colour regions (near-constant chromaticity, measured after an OPENING that removes
     *       white text): their own brightness variation is lighting, so it is divided out as well.
     *  4. out = in * gain, with a gain PER CHANNEL = 255 / lighting colour (paper -> white, casts gone).
     *  5. Ink is darkened relative to its LOCAL background (luminance only), so faint text stays
     *     readable while colour blocks — their own background — keep their tone.
     *  6. Hue-preserving clip: a pixel that would exceed 255 is scaled down as a whole.
     */
    private fun removeShadowsFrom(src: Bitmap): Bitmap {
        val rgba = Mat()
        Utils.bitmapToMat(src.copy(Bitmap.Config.ARGB_8888, false), rgba)
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        val w = rgb.cols()
        val h = rgb.rows()
        val scale = min(1.0, 1024.0 / max(w, h))
        val sw = max(1, (w * scale).toInt())
        val sh = max(1, (h * scale).toInt())
        val ns = sw * sh

        val small = Mat()
        Imgproc.resize(rgb, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        val k = max(7, (max(sw, sh) / 60) or 1)
        val ker = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))

        // 1) text-free colour background. The ink kernel is small: just wider than pen strokes, so thin
        //    shadows (e.g. a cable lying across the page) stay in the background and get removed.
        val ki = max(5, (max(sw, sh) / INK_DIV) or 1)
        val keri = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(ki.toDouble(), ki.toDouble()))
        val bgc = Mat()
        Imgproc.morphologyEx(small, bgc, Imgproc.MORPH_CLOSE, keri)
        Imgproc.medianBlur(bgc, bgc, ki)
        keri.release()
        small.release()

        val lab = Mat(); Imgproc.cvtColor(bgc, lab, Imgproc.COLOR_RGB2Lab)
        val grayM = Mat(); Imgproc.cvtColor(bgc, grayM, Imgproc.COLOR_RGB2GRAY)
        val labB = ByteArray(ns * 3); lab.get(0, 0, labB); lab.release()
        val gB = ByteArray(ns); grayM.get(0, 0, gB); grayM.release()
        val bgB = ByteArray(ns * 3); bgc.get(0, 0, bgB)
        val L = FloatArray(ns) { (gB[it].toInt() and 0xFF).toFloat() }

        // 2) paper colour = brightest 10% of the background
        val p90 = histPercentile(gB, 0.90)
        val bright = BooleanArray(ns) { L[it] >= p90 }
        val pa = maskedMedian(labB, 3, 1, bright)
        val pb = maskedMedian(labB, 3, 2, bright)
        val paperL = histPercentile(gB, 0.95)

        val content = Mat(sh, sw, CvType.CV_8UC1)
        val cB = ByteArray(ns)
        for (i in 0 until ns) {
            val da = (labB[i * 3 + 1].toInt() and 0xFF) - pa
            val db = (labB[i * 3 + 2].toInt() and 0xFF) - pb
            if (Math.hypot(da, db) > CHROMA_T || L[i] < 0.30 * paperL) cB[i] = 255.toByte()
        }
        content.put(0, 0, cB)
        dropSoftRegions(content, bgB, sw, sh, k, ker)
        Imgproc.dilate(content, content, ker)
        run {
            val contours = mutableListOf<MatOfPoint>()
            val hier = Mat()
            Imgproc.findContours(content.clone(), contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            hier.release()
            for (cnt in contours) {
                val area = Imgproc.contourArea(cnt)
                if (area >= k * k) {
                    val c2f = MatOfPoint2f(*cnt.toArray())
                    val rr = Imgproc.minAreaRect(c2f)
                    c2f.release()
                    val ra = rr.size.width * rr.size.height
                    if (ra > 0 && area / ra >= 0.6) {
                        val pts = arrayOfNulls<Point>(4)
                        rr.points(pts)
                        Imgproc.fillConvexPoly(content, MatOfPoint(*pts.requireNoNulls()), Scalar(255.0))
                    }
                }
                cnt.release()
            }
        }
        content.get(0, 0, cB)
        val isPaper = BooleanArray(ns) { cB[it].toInt() == 0 }
        val paperCount = isPaper.count { it }

        // 3) lighting COLOUR map (per channel)
        val bgf = FloatArray(ns * 3) { (bgB[it].toInt() and 0xFF).toFloat() }
        val illumC = bgf.copyOf()                 // bare paper: its own colour = the light's colour
        if (paperCount > 0.05 * ns) {
            // Under coloured areas: brightest surrounding paper, per channel (grey dilation = max filter)
            val fillM = Mat(sh, sw, CvType.CV_32FC3)
            fillM.put(0, 0, FloatArray(ns * 3) { if (isPaper[it / 3]) bgf[it] else 0f })
            val known = Mat(sh, sw, CvType.CV_8UC1)
            known.put(0, 0, ByteArray(ns) { if (isPaper[it]) 1 else 0 })
            val hole = Mat(); val grown = Mat(); val knownD = Mat(); val newly = Mat()
            var r = k
            while (r < max(sw, sh)) {
                Core.compare(known, Scalar(0.0), hole, Core.CMP_EQ)
                if (Core.countNonZero(hole) == 0) break
                val ker2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(2.0 * r + 1, 2.0 * r + 1))
                Imgproc.dilate(fillM, grown, ker2)
                Imgproc.dilate(known, knownD, ker2)
                Core.bitwise_and(hole, knownD, newly)
                grown.copyTo(fillM, newly)
                known.setTo(Scalar(1.0), newly)
                ker2.release()
                r *= 2
            }
            hole.release(); grown.release(); knownD.release(); newly.release(); known.release()
            Imgproc.GaussianBlur(fillM, fillM, Size(0.0, 0.0), max(sw, sh) / 30.0)
            val fill = FloatArray(ns * 3); fillM.get(0, 0, fill); fillM.release()
            val fillL = FloatArray(ns) { (0.299 * fill[it * 3] + 0.587 * fill[it * 3 + 1] + 0.114 * fill[it * 3 + 2]).toFloat() }
            for (i in 0 until ns) if (!isPaper[i]) for (c in 0..2) illumC[i * 3 + c] = max(fill[i * 3 + c], bgf[i * 3 + c])

            // flat colour regions: their own brightness variation is lighting too
            val bgo = Mat(); Imgproc.morphologyEx(bgc, bgo, Imgproc.MORPH_OPEN, ker)
            val loM = Mat(); Imgproc.cvtColor(bgo, loM, Imgproc.COLOR_RGB2GRAY)
            val oB = ByteArray(ns * 3); bgo.get(0, 0, oB); bgo.release()
            val loB = ByteArray(ns); loM.get(0, 0, loB); loM.release()
            val core0 = Mat(); Imgproc.erode(content, core0, ker)
            val c0B = ByteArray(ns); core0.get(0, 0, c0B); core0.release()
            val labels = Mat()
            val nl = Imgproc.connectedComponents(content, labels, 8, CvType.CV_32S)
            val lbl = IntArray(ns); labels.get(0, 0, lbl); labels.release()
            if (nl > 1) {
                val regCount = IntArray(nl); val coreCount = IntArray(nl)
                val s1 = DoubleArray(nl); val q1 = DoubleArray(nl); val s2 = DoubleArray(nl); val q2 = DoubleArray(nl)
                val hist = Array(nl) { IntArray(256) }
                val isCore = BooleanArray(ns)
                for (i in 0 until ns) {
                    val l = lbl[i]
                    if (l == 0) continue
                    regCount[l]++
                    val lo = loB[i].toInt() and 0xFF
                    if (c0B[i].toInt() != 0 && lo > 0.2 * paperL) {
                        isCore[i] = true
                        coreCount[l]++
                        val rr = (oB[i * 3].toInt() and 0xFF) + 1.0
                        val gg = (oB[i * 3 + 1].toInt() and 0xFF) + 1.0
                        val bb = (oB[i * 3 + 2].toInt() and 0xFF) + 1.0
                        val t = rr + gg + bb
                        val cb = bb / t; val cg = gg / t
                        s1[l] += cb; q1[l] += cb * cb; s2[l] += cg; q2[l] += cg * cg
                        hist[l][lo]++
                    }
                }
                val flat = BooleanArray(nl)
                val lmax = DoubleArray(nl)
                for (l in 1 until nl) {
                    val n = coreCount[l]
                    if (regCount[l] < k * k || n < k * k) continue
                    val sd1 = Math.sqrt(max(0.0, q1[l] / n - (s1[l] / n) * (s1[l] / n)))
                    val sd2 = Math.sqrt(max(0.0, q2[l] / n - (s2[l] / n) * (s2[l] / n)))
                    if (max(sd1, sd2) > FLAT_CHROM) continue
                    flat[l] = true
                    var acc = 0; var v = 255
                    for (b in 0..255) { acc += hist[l][b]; if (acc >= 0.95 * n) { v = b; break } }
                    lmax[l] = v.toDouble()
                }
                val topVals = Array(nl) { ArrayList<Float>() }
                for (i in 0 until ns) {
                    val l = lbl[i]
                    if (l != 0 && flat[l] && isCore[i] && (loB[i].toInt() and 0xFF) >= lmax[l] * 0.97) topVals[l].add(fillL[i])
                }
                val pLevel = DoubleArray(nl) { l ->
                    val vs = topVals[l]
                    if (vs.isEmpty()) paperL else { vs.sort(); vs[vs.size / 2].toDouble() }
                }
                for (i in 0 until ns) {
                    val l = lbl[i]
                    if (l == 0 || !flat[l]) continue
                    if (isCore[i]) {
                        // lighting relative to the envelope, applied to the envelope's colour
                        val sfac = ((loB[i].toInt() and 0xFF) * (pLevel[l] / max(lmax[l], 1.0))) / max(fillL[i].toDouble(), 1.0)
                        for (c in 0..2) illumC[i * 3 + c] = max(1.0, fill[i * 3 + c] * sfac).toFloat()
                    } else {
                        for (c in 0..2) illumC[i * 3 + c] = bgf[i * 3 + c]   // margin is (mostly) paper
                    }
                }
            }
        } else {
            // nothing to go on: one global lighting colour from the paper
            val pc = FloatArray(3) { c -> maskedMedian(bgB, 3, c, bright).toFloat() }
            for (i in 0 until ns) for (c in 0..2) illumC[i * 3 + c] = pc[c]
        }
        // Smooth the lighting map ONLY around coloured content, to hide seams there. On bare paper the
        // background is already smooth, and blurring it would smear sharp shadow edges (a cable's
        // shadow, say) and leave a grey outline along them.
        val nearM = Mat()
        Imgproc.dilate(content, nearM, ker)
        val nearB = ByteArray(ns); nearM.get(0, 0, nearB); nearM.release()
        content.release(); ker.release(); bgc.release()

        val illumM = Mat(sh, sw, CvType.CV_32FC3)
        illumM.put(0, 0, illumC)
        Imgproc.GaussianBlur(illumM, illumM, Size(0.0, 0.0), k / 4.0)
        val blurred = FloatArray(ns * 3); illumM.get(0, 0, blurred); illumM.release()
        for (i in 0 until ns) if (nearB[i].toInt() != 0) for (c in 0..2) illumC[i * 3 + c] = blurred[i * 3 + c]

        // 4) per-channel gain; corrected background brightness for the ink step
        val gainS = FloatArray(ns * 3) { (255.0 / max(illumC[it].toDouble(), 1.0)).coerceIn(1.0, 4.0).toFloat() }
        val bgYS = FloatArray(ns) { i ->
            val r = min(255.0, bgf[i * 3] * gainS[i * 3].toDouble())
            val g = min(255.0, bgf[i * 3 + 1] * gainS[i * 3 + 1].toDouble())
            val b = min(255.0, bgf[i * 3 + 2] * gainS[i * 3 + 2].toDouble())
            (0.299 * r + 0.587 * g + 0.114 * b).coerceIn(1.0, 255.0).toFloat()
        }

        // 5+6) full-resolution pass, row by row (the small maps are sampled bilinearly)
        val kr = INK_KNEE / 255.0
        val row = ByteArray(w * 3)
        val fx = sw.toDouble() / w
        val fy = sh.toDouble() / h
        for (y in 0 until h) {
            rgb.get(y, 0, row)
            val syf = ((y + 0.5) * fy - 0.5).coerceIn(0.0, (sh - 1).toDouble())
            val y0 = syf.toInt(); val y1 = min(y0 + 1, sh - 1); val ty = syf - y0
            for (x in 0 until w) {
                val sxf = ((x + 0.5) * fx - 0.5).coerceIn(0.0, (sw - 1).toDouble())
                val x0 = sxf.toInt(); val x1 = min(x0 + 1, sw - 1); val tx = sxf - x0
                val i00 = y0 * sw + x0; val i01 = y0 * sw + x1; val i10 = y1 * sw + x0; val i11 = y1 * sw + x1
                val w00 = (1 - tx) * (1 - ty); val w01 = tx * (1 - ty); val w10 = (1 - tx) * ty; val w11 = tx * ty
                fun gainAt(c: Int): Double =
                    gainS[i00 * 3 + c] * w00 + gainS[i01 * 3 + c] * w01 + gainS[i10 * 3 + c] * w10 + gainS[i11 * 3 + c] * w11
                val bgY = bgYS[i00] * w00 + bgYS[i01] * w01 + bgYS[i10] * w10 + bgYS[i11] * w11

                var r = (row[x * 3].toInt() and 0xFF) * gainAt(0)
                var g = (row[x * 3 + 1].toInt() and 0xFF) * gainAt(1)
                var b = (row[x * 3 + 2].toInt() and 0xFF) * gainAt(2)

                val yv = max(1.0, 0.299 * r + 0.587 * g + 0.114 * b)
                val rel = yv / bgY
                if (rel < kr) {
                    val f = (kr * Math.pow(rel / kr, INK_GAMMA)) / max(rel, 1e-6)
                    r *= f; g *= f; b *= f
                }
                val mx = max(r, max(g, b))
                if (mx > 255.0) { val f = 255.0 / mx; r *= f; g *= f; b *= f }
                row[x * 3] = r.toInt().coerceIn(0, 255).toByte()
                row[x * 3 + 1] = g.toInt().coerceIn(0, 255).toByte()
                row[x * 3 + 2] = b.toInt().coerceIn(0, 255).toByte()
            }
            rgb.put(y, 0, row)
        }

        val outRgba = Mat()
        Imgproc.cvtColor(rgb, outRgba, Imgproc.COLOR_RGB2RGBA)
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(outRgba, result)
        rgb.release(); outRgba.release()
        return result
    }

    /**
     * Removes soft-edged regions (tinted shadows) from [content] (8-bit mask, 255 = content). For each
     * region, the background colour just inside its edge is compared with the colour just outside: printed
     * colour shows a clear step, a shadow's penumbra barely any. Regions too thin to have an inside are
     * dropped as well. [bg] is the interleaved RGB text-free background of the same size.
     */
    private fun dropSoftRegions(content: Mat, bg: ByteArray, sw: Int, sh: Int, k: Int, ker: Mat) {
        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val n = Imgproc.connectedComponentsWithStats(content, labels, stats, cents, 8, CvType.CV_32S)
        cents.release()
        if (n <= 1) { labels.release(); stats.release(); return }
        val lbl = IntArray(sw * sh); labels.get(0, 0, lbl); labels.release()
        val cB = ByteArray(sw * sh); content.get(0, 0, cB)
        val reg = Mat(); val inner = Mat(); val outer = Mat()
        for (l in 1 until n) {
            val bx = stats.get(l, Imgproc.CC_STAT_LEFT)[0].toInt()
            val by = stats.get(l, Imgproc.CC_STAT_TOP)[0].toInt()
            val bw = stats.get(l, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val bh = stats.get(l, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            // work in the region's bounding box, padded by k so the outer ring fits
            val x0 = max(0, bx - k); val y0 = max(0, by - k)
            val x1 = min(sw, bx + bw + k); val y1 = min(sh, by + bh + k)
            val rw = x1 - x0; val rh = y1 - y0
            val rb = ByteArray(rw * rh)
            for (yy in 0 until rh) for (xx in 0 until rw) if (lbl[(y0 + yy) * sw + x0 + xx] == l) rb[yy * rw + xx] = 1
            reg.create(rh, rw, CvType.CV_8UC1); reg.put(0, 0, rb)
            Imgproc.erode(reg, inner, ker); Imgproc.dilate(reg, outer, ker)
            val ib = ByteArray(rw * rh); inner.get(0, 0, ib)
            val ob = ByteArray(rw * rh); outer.get(0, 0, ob)
            val hIn = Array(3) { IntArray(256) }; val hOut = Array(3) { IntArray(256) }
            var nIn = 0; var nOut = 0
            for (j in 0 until rw * rh) {
                val gi = ((y0 + j / rw) * sw + x0 + j % rw) * 3
                if (ib[j].toInt() != 0) { nIn++; for (c in 0..2) hIn[c][bg[gi + c].toInt() and 0xFF]++ }
                else if (ob[j].toInt() != 0 && rb[j].toInt() == 0) { nOut++; for (c in 0..2) hOut[c][bg[gi + c].toInt() and 0xFF]++ }
            }
            var step = 0.0
            if (nIn > 0 && nOut > 0) {
                for (c in 0..2) step = max(step, Math.abs(histMedian(hIn[c], nIn) - histMedian(hOut[c], nOut)))
            }
            if (step < SOFT_STEP) {
                for (yy in 0 until rh) for (xx in 0 until rw) {
                    val gi = (y0 + yy) * sw + x0 + xx
                    if (lbl[gi] == l) cB[gi] = 0
                }
            }
        }
        reg.release(); inner.release(); outer.release(); stats.release()
        content.put(0, 0, cB)
    }

    private fun histMedian(hist: IntArray, n: Int): Double {
        var acc = 0
        for (v in 0..255) { acc += hist[v]; if (acc * 2 >= n) return v.toDouble() }
        return 255.0
    }

    /** Value at fraction q (0..1) of an 8-bit single-channel buffer. */
    private fun histPercentile(bytes: ByteArray, q: Double): Double {
        val hist = IntArray(256)
        for (b in bytes) hist[b.toInt() and 0xFF]++
        val target = (bytes.size * q).toLong()
        var acc = 0L
        for (i in 0 until 256) { acc += hist[i]; if (acc >= target) return i.toDouble() }
        return 255.0
    }

    /** Median of channel [ch] of an interleaved 8-bit buffer, over the pixels where [mask] is true. */
    private fun maskedMedian(bytes: ByteArray, channels: Int, ch: Int, mask: BooleanArray): Double {
        val hist = IntArray(256)
        var n = 0
        for (i in mask.indices) if (mask[i]) { hist[bytes[i * channels + ch].toInt() and 0xFF]++; n++ }
        if (n == 0) return 128.0
        var acc = 0
        for (v in 0..255) { acc += hist[v]; if (acc * 2 >= n) return v.toDouble() }
        return 255.0
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

    /**
     * Copies a generated PDF into the device's public Downloads folder so it
     * persists outside the app's private storage and is visible in Files apps.
     *
     * On API 29+ this uses MediaStore (scoped storage — no permission needed).
     * On API 23-28 it writes directly to the public Downloads directory, which
     * requires WRITE_EXTERNAL_STORAGE (declared with maxSdkVersion=28 and
     * requested at runtime by the caller).
     *
     * Resolves with a user-facing location string.
     */
    @ReactMethod
    fun savePdfToDownloads(sourcePath: String, displayName: String, promise: Promise) {
        try {
            val src = File(sourcePath)
            if (!src.exists()) {
                promise.reject("NOT_FOUND", "No file at $sourcePath")
                return
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val resolver = reactContext.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "application/pdf")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri == null) {
                    promise.reject("INSERT_FAILED", "Could not create an entry in Downloads")
                    return
                }
                val opened = resolver.openOutputStream(uri)
                if (opened == null) {
                    resolver.delete(uri, null, null)
                    promise.reject("OPEN_FAILED", "Could not open Downloads entry for writing")
                    return
                }
                opened.use { out -> src.inputStream().use { input -> input.copyTo(out) } }

                // Clear IS_PENDING so the file becomes visible to other apps.
                val done = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                resolver.update(uri, done, null, null)

                promise.resolve("Downloads/$displayName")
            } else {
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if (!dir.exists()) dir.mkdirs()
                val dest = File(dir, displayName)
                src.inputStream().use { input ->
                    FileOutputStream(dest).use { out -> input.copyTo(out) }
                }
                promise.resolve(dest.absolutePath)
            }
        } catch (e: Exception) {
            promise.reject("SAVE_FAILED", e.message, e)
        }
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
        val srcFull = Mat()
        Utils.bitmapToMat(bitmap, srcFull)

        // Downscale for detection (real photos are huge, e.g. 2080x4608):
        // faster and less texture noise. Corners get scaled back to full res.
        val longEdge = max(srcFull.rows(), srcFull.cols()).toDouble()
        val detectScale = if (longEdge > 800.0) 800.0 / longEdge else 1.0
        val src = Mat()
        if (detectScale < 1.0) {
            Imgproc.resize(srcFull, src, Size(srcFull.cols() * detectScale, srcFull.rows() * detectScale))
        } else {
            srcFull.copyTo(src)
        }

        // Build grayscale + HSV and delegate to the shared detector (the HSV
        // pass isolates a white page from a colored background). See
        // DocumentDetector — the same code powers the live camera analyzer.
        val rgb = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        val gray = Mat()
        Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
        val hsv = Mat()
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

        // Primary: ML page segmentation -> the paper's real OUTLINE (irregular polygon,
        // or exactly 4 corners when the page is a clean rectangle). Fallback: classical
        // contour detection, which can only produce a quad.
        val polygon = paperSegmenter.findPagePolygon(src)
        val shape: Array<Point>? = when {
            polygon == null -> DocumentDetector.findBestQuad(gray, hsv)
            // perspectiveCorrect expects [TL, TR, BR, BL], so order a 4-point result.
            polygon.size == 4 -> DocumentDetector.orderCorners(polygon)
            else -> polygon
        }

        val results = mutableListOf<DetectedDoc>()
        if (shape != null) {
            val shapeArea = Imgproc.contourArea(MatOfPoint2f(*shape))
            val pts = shape.map {
                floatArrayOf((it.x / detectScale).toFloat(), (it.y / detectScale).toFloat())
            }
            results.add(DetectedDoc(pts, shape.size == 4, shapeArea / (detectScale * detectScale)))
        }

        rgb.release(); gray.release(); hsv.release(); src.release(); srcFull.release()
        return results
    }

    /**
     * Crops to an arbitrary (possibly non-convex, torn-edge) polygon instead
     * of perspective-warping a quadrilateral: masks out everything outside the
     * polygon to white, then DESKEWS so the page comes out upright.
     *
     * The deskew matters: a plain axis-aligned boundingRect() crop keeps
     * whatever tilt the page had in the photo, so the exported PDF shows a
     * slanted page. Instead we take the polygon's minimum-area (rotated)
     * rectangle, rotate the image by that angle so the page edges become
     * horizontal/vertical, and crop to the now-upright rectangle.
     *
     * Use this for any document detectAllDocuments() reported as
     * isQuadrilateral: false (perspectiveCorrect handles the clean-quad case,
     * which is already straightened by the homography).
     */
    @ReactMethod
    fun cropToContour(imagePath: String, pointsJs: ReadableArray, outputPath: String, promise: Promise) {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image")

            val points = (0 until pointsJs.size()).map {
                val m = pointsJs.getMap(it)
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

            // Rotated bounding rect gives the page's actual tilt.
            val poly2f = MatOfPoint2f(*points.toTypedArray())
            val rr = Imgproc.minAreaRect(poly2f)
            poly2f.release()

            // Normalize to the smallest rotation that makes the page upright.
            // OpenCV reports the angle in (0, 90]; past 45 degrees it's shorter
            // to rotate the other way and swap width/height.
            var angle = rr.angle
            var rw = rr.size.width
            var rh = rr.size.height
            if (angle > 45.0) {
                angle -= 90.0
                val t = rw; rw = rh; rh = t
            }

            val deskewed: Mat
            val center = rr.center
            if (kotlin.math.abs(angle) < 0.3) {
                // Already straight — skip the resample.
                deskewed = whiteBg
            } else {
                deskewed = Mat()
                val rot = Imgproc.getRotationMatrix2D(center, angle, 1.0)
                Imgproc.warpAffine(
                    whiteBg, deskewed, rot, whiteBg.size(), Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT, org.opencv.core.Scalar(255.0, 255.0, 255.0, 255.0)
                )
                rot.release()
            }

            // Crop the upright rectangle around the (unmoved) center.
            val x0 = (center.x - rw / 2.0).toInt().coerceIn(0, deskewed.cols() - 1)
            val y0 = (center.y - rh / 2.0).toInt().coerceIn(0, deskewed.rows() - 1)
            val x1 = (center.x + rw / 2.0).toInt().coerceIn(x0 + 1, deskewed.cols())
            val y1 = (center.y + rh / 2.0).toInt().coerceIn(y0 + 1, deskewed.rows())
            val roi = org.opencv.core.Rect(x0, y0, x1 - x0, y1 - y0)
            val cropped = Mat(deskewed, roi)

            val result = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(cropped, result)

            // Every cropped page is cleaned once here, before it reaches the filter screen.
            writeJpeg(removeShadowsFrom(result), outputPath)
            mask.release(); src.release(); cropped.release()
            if (deskewed !== whiteBg) deskewed.release()
            whiteBg.release()
            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("CROP_FAILED", e.message, e)
        }
    }
}