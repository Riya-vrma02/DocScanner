package com.docscannerrn

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.media.ExifInterface
import android.util.Log
import android.view.View
import android.widget.FrameLayout
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactContext
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.uimanager.ThemedReactContext
import com.facebook.react.uimanager.events.RCTEventEmitter
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.hypot
import kotlin.math.max

/**
 * In-app live document scanner. Hosts a CameraX preview, runs [PaperSegmenter]
 * (the paper_seg ONNX model) on analysis frames to draw a live outline overlay,
 * and on capture takes a full-resolution still, detects the page on it, and
 * emits `onDocumentCaptured` with the image path + outline points so JS can
 * hand it to the existing CropScreen for manual fine-tuning.
 *
 * Changes vs. the previous version:
 *  1. The live outline is smoothed (EMA, fixed vertex count for polygons) so a
 *     many-point outline that wraps curved/torn edges does not jitter.
 *  2. Capture honours what the user was looking at: if the still's own
 *     detection disagrees with the last live outline (typical when the page lies
 *     on a pile of other pages), the live outline is used instead of falling
 *     back to the whole frame / the whole pile.
 *  3. The classical CV detector is only used live when the model is unavailable
 *     (see [CV_FALLBACK_ONLY_IF_MODEL_UNAVAILABLE]); on a pile it tends to
 *     outline the entire stack and look like a successful detection.
 *  4. Lifecycle/executor fix: a LifecycleRegistry cannot leave DESTROYED and a
 *     shut-down executor rejects work, so after navigating away and back the
 *     camera analysis and captures silently died. Both are recreated on attach.
 *
 * [DocumentDetector] (classical CV) remains the fallback for the still.
 */
@SuppressLint("ViewConstructor")
class DocumentScannerView(private val reactCtx: ThemedReactContext) :
    FrameLayout(reactCtx), LifecycleOwner {

    companion object {
        /**
         * true : live preview uses DocumentDetector only if the ONNX model failed to load.
         * false: also use it whenever the model returns nothing (old behaviour).
         */
        private const val CV_FALLBACK_ONLY_IF_MODEL_UNAVAILABLE = true

        private const val TAG = "DocScannerView"

        /** Draws a small text readout on the preview (outline source, point count, pile state). */
        private const val DEBUG_HUD = false

        /** A live outline older than this is not trusted as the capture fallback. */
        private const val LIVE_FRESH_MS = 1500L

        /** Still-detection and live outline must overlap at least this much (IoU) to be "in agreement". */
        private const val LIVE_AGREE_IOU = 0.80

        /** Vertex count polygons are resampled to for stable live smoothing. */
        private const val OUTLINE_POINTS = 32

        /** Weight of the newest frame in the outline EMA (lower = steadier, laggier). */
        private const val SMOOTH_ALPHA = 0.5

        /** If any vertex jumps more than this fraction of the frame diagonal, snap instead of smoothing. */
        private const val SNAP_JUMP_FRACTION = 0.15
    }

    private var lifecycleRegistry = LifecycleRegistry(this)
    override val lifecycle: Lifecycle get() = lifecycleRegistry

    private val previewView = PreviewView(reactCtx)
    private val overlay = OverlayView(reactCtx)
    private var executor = Executors.newSingleThreadExecutor()

    private var imageCapture: ImageCapture? = null
    private var lastCaptureTrigger = 0
    private val capturing = AtomicBoolean(false)
    private var missCount = 0
    private var frameCount = 0

    /** Last detection state pushed to JS, so identical states aren't re-sent. */
    private var lastDetectionKey = ""

    /**
     * ML page segmentation (paper_seg.onnx). Used for both the live overlay and
     * the captured still; [DocumentDetector] is the fallback when the model is
     * unavailable or can't fit a single quad.
     */
    private val segmenter = PaperSegmenter(reactCtx.applicationContext)

    private val smoother = OutlineSmoother()

    /** Last good live outline, normalised to 0..1 of the upright analysis frame. */
    @Volatile private var lastLive: Array<Point>? = null
    @Volatile private var lastLiveInfo = PaperSegmenter.SheetInfo(false, 0, 1.0, 1.0, false)
    @Volatile private var lastLiveAt = 0L

    init {
        addView(previewView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        addView(overlay, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        lifecycleRegistry.currentState = Lifecycle.State.INITIALIZED
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        // DESTROYED is terminal for a LifecycleRegistry, and a shut-down executor rejects
        // all work, so both must be recreated when the view is re-attached.
        if (lifecycleRegistry.currentState == Lifecycle.State.DESTROYED) {
            lifecycleRegistry = LifecycleRegistry(this)
        }
        if (executor.isShutdown) {
            executor = Executors.newSingleThreadExecutor()
        }
        capturing.set(false)
        missCount = 0
        lastDetectionKey = ""
        lastLive = null
        smoother.reset()
        segmenter.resetSmoothing()

        lifecycleRegistry.currentState = Lifecycle.State.RESUMED
        startCamera()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        lifecycleRegistry.currentState = Lifecycle.State.DESTROYED
        executor.shutdown()
    }

    // React Native does not lay out children of custom native views on its own;
    // this forces a measure/layout pass so the preview + overlay fill the view.
    private val measureAndLayout = Runnable {
        measure(
            MeasureSpec.makeMeasureSpec(width, MeasureSpec.EXACTLY),
            MeasureSpec.makeMeasureSpec(height, MeasureSpec.EXACTLY)
        )
        layout(left, top, right, bottom)
    }

    override fun requestLayout() {
        super.requestLayout()
        post(measureAndLayout)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(reactCtx)
        future.addListener({
            try {
                val provider = future.get()

                val preview = Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                preview.setSurfaceProvider(previewView.surfaceProvider)

                val analysis = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(executor, ImageAnalysis.Analyzer { proxy -> analyzeFrame(proxy) })

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .build()

                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, imageCapture
                )
            } catch (e: Exception) {
                emitError(e.message ?: "camera start failed")
            }
        }, ContextCompat.getMainExecutor(reactCtx))
    }

    // ----- Live per-frame detection -----------------------------------------

    private fun analyzeFrame(image: ImageProxy) {
        var upright: Mat? = null
        try {
            val rgba = rgbaToMat(image)
            val rotated = rotateUpright(rgba, image.imageInfo.rotationDegrees)
            if (rotated !== rgba) rgba.release()
            upright = rotated

            val fw = rotated.cols()
            val fh = rotated.rows()

            // Prefer the ML segmentation model. Frames are dropped while this runs
            // (STRATEGY_KEEP_ONLY_LATEST), which throttles the overlay to inference speed.
            var poly: Array<Point>? = segmenter.findPagePolygon(rotated, live = true)
            var info = segmenter.lastSheetInfo
            var usedCv = false

            // Classical CV is a weak detector on piles (it outlines the whole stack and
            // looks like a success), so by default it is only used when the model itself
            // is unavailable.
            val allowCv = !CV_FALLBACK_ONLY_IF_MODEL_UNAVAILABLE || segmenter.status == "unavailable"
            if (poly == null && allowCv) {
                val rgb = Mat(); Imgproc.cvtColor(rotated, rgb, Imgproc.COLOR_RGBA2RGB)
                val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
                val hsv = Mat(); Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
                poly = DocumentDetector.findBestQuad(gray, hsv)
                usedCv = poly != null
                rgb.release(); gray.release(); hsv.release()
                info = PaperSegmenter.SheetInfo(false, 0, 1.0, 1.0, false)
            }

            val vw = width
            val vh = height

            if (poly != null && vw > 0 && vh > 0) {
                missCount = 0

                val smooth = smoother.smooth(poly, hypot(fw.toDouble(), fh.toDouble()))

                // Remember what the user is looking at, for the capture step.
                if (!usedCv) {
                    lastLive = Array(smooth.size) { Point(smooth[it].x / fw, smooth[it].y / fh) }
                    lastLiveInfo = info
                    lastLiveAt = System.currentTimeMillis()
                }

                if (++frameCount % 30 == 1) {
                    Log.d(
                        TAG,
                        "outline: ${poly.size} pts raw, ${smooth.size} smoothed, " +
                            "source=${if (usedCv) "CV (always 4 pts)" else "model"}"
                    )
                }

                val mapped = smooth.map { mapToView(it, fw, fh, vw, vh) }.toTypedArray()
                post { overlay.setOutline(mapped) }

                if (DEBUG_HUD) {
                    val dbg = "src=${if (usedCv) "CV (4 pts only)" else "model"}  raw=${poly.size}  " +
                        "smooth=${smooth.size}\n${segmenter.lastDebug}" +
                        (if (usedCv) "\nmodel err=${segmenter.lastError}" else "")
                    post { overlay.setDebug(dbg) }
                }

                emitDetectionState(
                    detected = true,
                    overlapping = info.overlapping,
                    separateSheets = info.separateSheets,
                    blobRectangularity = info.blobRectangularity,
                    blobSolidity = info.blobSolidity,
                    splitApplied = info.splitApplied
                )
            } else {
                // Keep the last outline briefly to avoid flicker on a dropped detection.
                missCount++
                if (DEBUG_HUD) {
                    val dbg = "NO OUTLINE  seg=${segmenter.status}  err=${segmenter.lastError}\n${segmenter.lastDebug}"
                    post { overlay.setDebug(dbg) }
                }
                if (missCount % 30 == 1) {
                    Log.w(
                        TAG,
                        "no outline: poly=${poly != null} view=${vw}x$vh " +
                            "seg=${segmenter.status} err=${segmenter.lastError}"
                    )
                }
                if (missCount >= 8) {
                    segmenter.resetSmoothing()
                    smoother.reset()
                    lastLive = null
                    post { overlay.setOutline(null) }
                    emitDetectionState(
                        detected = false,
                        overlapping = false,
                        separateSheets = 0,
                        blobRectangularity = 1.0,
                        blobSolidity = 1.0,
                        splitApplied = false
                    )
                }
            }
        } catch (e: Exception) {
            // A single bad frame is fine, but never fail silently: if this fires on
            // every frame the overlay will never appear.
            Log.w(TAG, "analyzeFrame failed: ${e::class.java.simpleName}: ${e.message}")
        } finally {
            upright?.release()
            image.close()
        }
    }

    private fun mapToView(p: Point, aw: Int, ah: Int, vw: Int, vh: Int): PointF {
        // PreviewView default scale type is FILL_CENTER: scale to fill, crop overflow.
        val scale = max(vw.toFloat() / aw, vh.toFloat() / ah)
        val dx = (vw - aw * scale) / 2f
        val dy = (vh - ah * scale) / 2f
        return PointF((p.x * scale + dx).toFloat(), (p.y * scale + dy).toFloat())
    }

    /** Builds an RGBA Mat from an ImageAnalysis frame configured for RGBA_8888 output. */
    private fun rgbaToMat(image: ImageProxy): Mat {
        val width = image.width
        val height = image.height
        val plane = image.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val mat = Mat(height, width, CvType.CV_8UC4)
        val rowBuf = ByteArray(rowStride)
        val rowPix = ByteArray(width * 4)
        for (r in 0 until height) {
            val start = r * rowStride
            if (start >= buffer.capacity()) break
            buffer.position(start)
            val toRead = minOf(rowStride, buffer.capacity() - start)
            buffer.get(rowBuf, 0, toRead)
            System.arraycopy(rowBuf, 0, rowPix, 0, minOf(width * 4, toRead))
            mat.put(r, 0, rowPix)
        }
        return mat
    }

    private fun rotateUpright(src: Mat, degrees: Int): Mat {
        return when (degrees) {
            90 -> Mat().also { Core.rotate(src, it, Core.ROTATE_90_CLOCKWISE) }
            180 -> Mat().also { Core.rotate(src, it, Core.ROTATE_180) }
            270 -> Mat().also { Core.rotate(src, it, Core.ROTATE_90_COUNTERCLOCKWISE) }
            else -> src
        }
    }

    // ----- Full-resolution capture -----------------------------------------

    fun onCaptureTriggerChanged(value: Int) {
        if (value <= 0 || value == lastCaptureTrigger) return
        lastCaptureTrigger = value
        capture()
    }

    private fun capture() {
        val ic = imageCapture ?: return
        if (!capturing.compareAndSet(false, true)) return

        val file = File(reactCtx.cacheDir, "scan_${System.currentTimeMillis()}.jpg")
        val options = ImageCapture.OutputFileOptions.Builder(file).build()
        ic.takePicture(
            options,
            ContextCompat.getMainExecutor(reactCtx),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                    try {
                        executor.execute {
                            try {
                                processCaptured(file.absolutePath)
                            } catch (e: Exception) {
                                emitError(e.message ?: "capture processing failed")
                            } finally {
                                capturing.set(false)
                            }
                        }
                    } catch (e: Exception) {
                        // Executor already shut down (view detached mid-capture).
                        capturing.set(false)
                        emitError(e.message ?: "scanner is not running")
                    }
                }

                override fun onError(exc: ImageCaptureException) {
                    capturing.set(false)
                    emitError(exc.message ?: "capture failed")
                }
            }
        )
    }

    private fun processCaptured(path: String) {
        normalizeOrientation(path)
        val bitmap = BitmapFactory.decodeFile(path) ?: run {
            emitError("could not decode captured image")
            return
        }
        val fullW = bitmap.width
        val fullH = bitmap.height

        val srcFull = Mat()
        val src = Mat()
        val rgb = Mat()
        val gray = Mat()
        val hsv = Mat()
        try {
            Utils.bitmapToMat(bitmap, srcFull)
            val longEdge = max(srcFull.rows(), srcFull.cols()).toDouble()
            val detectScale = if (longEdge > 800.0) 800.0 / longEdge else 1.0
            if (detectScale < 1.0) {
                Imgproc.resize(srcFull, src, Size(srcFull.cols() * detectScale, srcFull.rows() * detectScale))
            } else {
                srcFull.copyTo(src)
            }
            Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

            // 1) ML page segmentation on the still.
            val mlPoly = segmenter.findPagePolygon(src)
            val mlInfo = segmenter.lastSheetInfo

            // 2) The outline the user was looking at just before pressing capture,
            //    mapped into this still's coordinate space (both are upright, so
            //    normalised coordinates carry over).
            val liveSnap = lastLive
            val live: Array<Point>? =
                if (liveSnap != null && System.currentTimeMillis() - lastLiveAt < LIVE_FRESH_MS) {
                    Array(liveSnap.size) { Point(liveSnap[it].x * src.cols(), liveSnap[it].y * src.rows()) }
                } else null
            val liveInfo = lastLiveInfo

            // 3) Choose. The still's own result is more precise (higher res) but on a
            //    pile it can flip to the whole stack or fail; when it disagrees with
            //    what the user saw, trust the live outline.
            var poly: Array<Point>?
            var detector: String
            var info = mlInfo

            val mlAgrees = mlPoly != null &&
                (live == null || polyIou(src.size(), mlPoly, live) >= LIVE_AGREE_IOU)

            if (mlAgrees) {
                poly = mlPoly
                detector = "ml"
            } else if (live != null) {
                poly = live
                detector = "live"
                info = liveInfo
            } else {
                poly = DocumentDetector.findBestQuad(gray, hsv)
                detector = if (poly != null) "cv" else "fullframe"
            }

            val corners: Array<DoubleArray> = if (poly != null) {
                poly.map { doubleArrayOf(it.x / detectScale, it.y / detectScale) }.toTypedArray()
            } else {
                // Full-frame fallback so the user can still adjust manually.
                arrayOf(
                    doubleArrayOf(0.0, 0.0),
                    doubleArrayOf(fullW.toDouble(), 0.0),
                    doubleArrayOf(fullW.toDouble(), fullH.toDouble()),
                    doubleArrayOf(0.0, fullH.toDouble())
                )
            }

            emitCaptured(path, fullW, fullH, corners, detector, segmenter.status, segmenter.lastError, info)
        } finally {
            srcFull.release(); src.release(); rgb.release(); gray.release(); hsv.release()
            bitmap.recycle()
        }
    }

    /** Intersection-over-union of two filled polygons, rasterised at [size]. */
    private fun polyIou(size: Size, a: Array<Point>, b: Array<Point>): Double {
        val ma = Mat.zeros(size, CvType.CV_8UC1)
        val mb = Mat.zeros(size, CvType.CV_8UC1)
        val inter = Mat()
        val union = Mat()
        val pa = MatOfPoint(*a)
        val pb = MatOfPoint(*b)
        try {
            Imgproc.fillPoly(ma, listOf(pa), Scalar(255.0))
            Imgproc.fillPoly(mb, listOf(pb), Scalar(255.0))
            Core.bitwise_and(ma, mb, inter)
            Core.bitwise_or(ma, mb, union)
            val i = Core.countNonZero(inter).toDouble()
            val u = Core.countNonZero(union).toDouble()
            return if (u > 0) i / u else 0.0
        } finally {
            ma.release(); mb.release(); inter.release(); union.release()
            pa.release(); pb.release()
        }
    }

    /** Bakes EXIF orientation into the pixels so downstream coords are consistent. */
    private fun normalizeOrientation(path: String) {
        val orientation = ExifInterface(path).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
        )
        if (orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED
        ) return

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
        FileOutputStream(File(path)).use { rotated.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        if (rotated != bitmap) bitmap.recycle()
        ExifInterface(path).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL.toString())
            saveAttributes()
        }
    }

    // ----- Events to JS ------------------------------------------------------

    private fun emitCaptured(
        path: String,
        w: Int,
        h: Int,
        corners: Array<DoubleArray>,
        detector: String,
        segStatus: String,
        segError: String?,
        info: PaperSegmenter.SheetInfo
    ) {
        val cornerArray: WritableArray = Arguments.createArray()
        for (c in corners) {
            val m = Arguments.createMap()
            m.putDouble("x", c[0])
            m.putDouble("y", c[1])
            cornerArray.pushMap(m)
        }
        val event: WritableMap = Arguments.createMap()
        event.putString("path", path)
        event.putInt("imageWidth", w)
        event.putInt("imageHeight", h)
        // 4 points for a clean page, up to OUTLINE_POINTS when the outline is a real polygon.
        event.putArray("corners", cornerArray)
        event.putString("detector", detector)
        event.putString("segStatus", segStatus)
        event.putString("segError", segError)
        // Overlap verdict for THIS capture, so the user can be told that only the
        // top sheet made it into the scan.
        event.putBoolean("overlapping", info.overlapping)
        event.putInt("separateSheets", info.separateSheets)
        event.putDouble("blobRectangularity", info.blobRectangularity)
        event.putDouble("blobSolidity", info.blobSolidity)
        event.putBoolean("splitApplied", info.splitApplied)
        dispatch("onDocumentCaptured", event)
    }

    /**
     * Live sheet state for the preview UI. Only dispatched when it actually
     * changes — the analyzer runs continuously and re-sending an identical
     * state every frame would flood the bridge.
     */
    private fun emitDetectionState(
        detected: Boolean,
        overlapping: Boolean,
        separateSheets: Int,
        blobRectangularity: Double,
        blobSolidity: Double,
        splitApplied: Boolean
    ) {
        // Rectangularity is continuous and would differ every single frame, which
        // would defeat the throttle entirely. Bucket it for the comparison key and
        // send the precise value only when something else also changed.
        val bucket = (blobRectangularity * 20).toInt()
        val sBucket = (blobSolidity * 20).toInt()
        val key = "$detected|$overlapping|$separateSheets|$bucket|$sBucket|$splitApplied"
        if (key == lastDetectionKey) return
        lastDetectionKey = key

        val event: WritableMap = Arguments.createMap()
        event.putBoolean("detected", detected)
        event.putBoolean("overlapping", overlapping)
        event.putInt("separateSheets", separateSheets)
        event.putDouble("blobRectangularity", blobRectangularity)
        event.putDouble("blobSolidity", blobSolidity)
        event.putBoolean("splitApplied", splitApplied)
        dispatch("onDetectionState", event)
    }

    private fun emitError(message: String) {
        val event: WritableMap = Arguments.createMap()
        event.putString("message", message)
        dispatch("onScannerError", event)
    }

    private fun dispatch(name: String, event: WritableMap) {
        (reactCtx as ReactContext)
            .getJSModule(RCTEventEmitter::class.java)
            .receiveEvent(id, name, event)
    }

    // ----- Outline smoothing -------------------------------------------------

    /**
     * Temporal smoothing for the live outline.
     *
     * - 4-point outlines: corners are aligned to the previous frame's order and averaged.
     * - Polygons (5+ points): resampled to a fixed [OUTLINE_POINTS] vertices at even arc
     *   length, wound clockwise, rotated to best match the previous frame, then averaged.
     *   A varying vertex count would otherwise make the outline shimmer.
     * - A sudden large move (different page, camera swing) snaps instead of smoothing.
     *
     * Only touched from the analysis executor thread (and reset before the camera starts).
     */
    private class OutlineSmoother {
        private var prev: Array<Point>? = null

        fun reset() {
            prev = null
        }

        fun smooth(raw: Array<Point>, frameDiag: Double): Array<Point> {
            // Keep the model's own vertices (no resampling): resampling to evenly spaced
            // points chamfers every sharp corner and made the outline look distorted.
            val cur: Array<Point> = if (raw.size <= 4) raw else clockwise(raw)

            val p = prev
            if (p == null || p.size != cur.size) {
                prev = cur
                return cur
            }

            val aligned = rotateToMatch(cur, p)

            var maxJump = 0.0
            for (i in aligned.indices) {
                val d = hypot(aligned[i].x - p[i].x, aligned[i].y - p[i].y)
                if (d > maxJump) maxJump = d
            }
            if (maxJump > SNAP_JUMP_FRACTION * frameDiag) {
                prev = aligned
                return aligned
            }

            val out = Array(aligned.size) { i ->
                Point(
                    p[i].x + (aligned[i].x - p[i].x) * SMOOTH_ALPHA,
                    p[i].y + (aligned[i].y - p[i].y) * SMOOTH_ALPHA
                )
            }
            prev = out
            return out
        }

        /** Same winding every frame (image coords, y down: positive area = clockwise on screen). */
        private fun clockwise(poly: Array<Point>): Array<Point> {
            var area = 0.0
            for (i in poly.indices) {
                val a = poly[i]
                val b = poly[(i + 1) % poly.size]
                area += a.x * b.y - b.x * a.y
            }
            return if (area < 0) poly.reversedArray() else poly
        }

        /** [n] points spaced evenly along the closed outline's perimeter. */
        private fun resampleClosed(poly: Array<Point>, n: Int): Array<Point> {
            val m = poly.size
            val segLen = DoubleArray(m) { i ->
                val a = poly[i]
                val b = poly[(i + 1) % m]
                hypot(b.x - a.x, b.y - a.y)
            }
            val total = segLen.sum()
            if (total < 1e-6) return poly

            val out = ArrayList<Point>(n)
            var i = 0
            var acc = 0.0
            for (k in 0 until n) {
                val target = total * k / n
                while (i < m - 1 && acc + segLen[i] < target) {
                    acc += segLen[i]
                    i++
                }
                val a = poly[i]
                val b = poly[(i + 1) % m]
                val t = if (segLen[i] > 1e-9) (target - acc) / segLen[i] else 0.0
                out.add(Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
            }
            return out.toTypedArray()
        }

        /** Cyclic shift of [cur] that lines up best with [ref], so vertex i tracks vertex i. */
        private fun rotateToMatch(cur: Array<Point>, ref: Array<Point>): Array<Point> {
            val n = cur.size
            var bestShift = 0
            var bestCost = Double.MAX_VALUE
            for (s in 0 until n) {
                var cost = 0.0
                for (i in 0 until n) {
                    val a = cur[(i + s) % n]
                    val b = ref[i]
                    cost += hypot(a.x - b.x, a.y - b.y)
                }
                if (cost < bestCost) {
                    bestCost = cost
                    bestShift = s
                }
            }
            return Array(n) { cur[(it + bestShift) % n] }
        }
    }

    // ----- Overlay view ------------------------------------------------------

    private class OverlayView(context: android.content.Context) : View(context) {
        private var outline: Array<PointF>? = null
        private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.STROKE
            strokeWidth = 5f
            strokeJoin = Paint.Join.MITER
        }
        private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#334CAF50")
            style = Paint.Style.FILL
        }
        private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.FILL
        }

        private var debugText: String = ""
        private val debugPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 30f
            setShadowLayer(4f, 0f, 0f, Color.BLACK)
        }

        fun setOutline(q: Array<PointF>?) {
            outline = q
            invalidate()
        }

        fun setDebug(text: String) {
            debugText = text
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)

            if (debugText.isNotEmpty()) {
                var y = 60f
                for (line in debugText.split("\n")) {
                    canvas.drawText(line, 16f, y, debugPaint)
                    y += 38f
                }
            }

            val q = outline ?: return
            if (q.size < 3) return

            // Straight segments between the real vertices: rounding corners through edge
            // midpoints made the outline drift away from the page edge.
            val path = Path()
            path.moveTo(q[0].x, q[0].y)
            for (i in 1 until q.size) path.lineTo(q[i].x, q[i].y)
            path.close()

            canvas.drawPath(path, fillPaint)
            canvas.drawPath(path, linePaint)
            if (q.size == 4) for (p in q) canvas.drawCircle(p.x, p.y, 10f, dotPaint)
        }
    }
}