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
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * In-app live document scanner. Hosts a CameraX preview, runs [PaperSegmenter]
 * (the paper_seg ONNX model) on analysis frames to draw a live edge overlay,
 * and on capture takes a full-resolution still, detects the page on it, and
 * emits `onDocumentCaptured` with the image path + corners so JS can hand it
 * to the existing CropScreen for manual fine-tuning.
 *
 * [DocumentDetector] (classical CV) is the fallback whenever the model is
 * unavailable or can't fit a single quad.
 *
 * Implemented as its own LifecycleOwner so the camera starts/stops with the
 * view's attach/detach rather than the host Activity's lifecycle.
 */
@SuppressLint("ViewConstructor")
class DocumentScannerView(private val reactCtx: ThemedReactContext) :
    FrameLayout(reactCtx), LifecycleOwner {

    private val lifecycleRegistry = LifecycleRegistry(this)
    override val lifecycle: Lifecycle get() = lifecycleRegistry

    private val previewView = PreviewView(reactCtx)
    private val overlay = OverlayView(reactCtx)
    private val executor = Executors.newSingleThreadExecutor()

    private var imageCapture: ImageCapture? = null
    private var lastCaptureTrigger = 0
    private val capturing = AtomicBoolean(false)
    private var missCount = 0

    /**
     * ML page segmentation (paper_seg.onnx). Used for both the live overlay and
     * the captured still; [DocumentDetector] is the fallback when the model is
     * unavailable or can't fit a single quad.
     */
    private val segmenter = PaperSegmenter(reactCtx.applicationContext)

    init {
        addView(previewView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        addView(overlay, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        lifecycleRegistry.currentState = Lifecycle.State.INITIALIZED
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
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
        try {
            val rgba = rgbaToMat(image)
            val upright = rotateUpright(rgba, image.imageInfo.rotationDegrees)
            if (upright !== rgba) rgba.release()

            // Prefer the ML segmentation model — classical CV can't reliably
            // separate a page from a patterned/cluttered background (it kept
            // locking onto a near-full-frame contour). Only fall back to the CV
            // detector when the model is unavailable or finds no single quad.
            // Frames are dropped while this runs (STRATEGY_KEEP_ONLY_LATEST),
            // which naturally throttles the overlay to inference speed.
            var quad = segmenter.findPagePolygon(upright)
            if (quad == null) {
                val rgb = Mat(); Imgproc.cvtColor(upright, rgb, Imgproc.COLOR_RGBA2RGB)
                val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
                val hsv = Mat(); Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
                quad = DocumentDetector.findBestQuad(gray, hsv)
                rgb.release(); gray.release(); hsv.release()
            }

            val vw = width
            val vh = height

            if (quad != null && vw > 0 && vh > 0) {
                missCount = 0
                val mapped = quad.map { mapToView(it, upright.cols(), upright.rows(), vw, vh) }.toTypedArray()
                post { overlay.setQuad(mapped) }
            } else {
                // Keep the last box briefly to avoid flicker on a dropped detection.
                missCount++
                if (missCount >= 5) post { overlay.setQuad(null) }
            }

            upright.release()
        } catch (e: Exception) {
            // Ignore a single bad frame.
        } finally {
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
                    executor.execute {
                        try {
                            processCaptured(file.absolutePath)
                        } catch (e: Exception) {
                            emitError(e.message ?: "capture processing failed")
                        } finally {
                            capturing.set(false)
                        }
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
        Utils.bitmapToMat(bitmap, srcFull)
        val longEdge = max(srcFull.rows(), srcFull.cols()).toDouble()
        val detectScale = if (longEdge > 800.0) 800.0 / longEdge else 1.0
        val src = Mat()
        if (detectScale < 1.0) {
            Imgproc.resize(srcFull, src, Size(srcFull.cols() * detectScale, srcFull.rows() * detectScale))
        } else {
            srcFull.copyTo(src)
        }
        val rgb = Mat(); Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
        val hsv = Mat(); Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)

        // Primary: ML page segmentation (robust on cluttered/bright backgrounds).
        // Fallback: classical contour detection if the model is unavailable or
        // returns nothing usable. Both return coords in `src`'s space, which is
        // scaled back to full resolution below.
        val mlQuad = segmenter.findPagePolygon(src)
        val quad = mlQuad ?: DocumentDetector.findBestQuad(gray, hsv)

        // Which detector actually produced these corners — reported to JS so a
        // silent ML failure can't masquerade as poor detection.
        val detector = when {
            mlQuad != null -> "ml"
            quad != null -> "cv"
            else -> "fullframe"
        }

        val corners: Array<DoubleArray> = if (quad != null) {
            quad.map { doubleArrayOf(it.x / detectScale, it.y / detectScale) }.toTypedArray()
        } else {
            // Full-frame fallback so the user can still adjust manually.
            arrayOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(fullW.toDouble(), 0.0),
                doubleArrayOf(fullW.toDouble(), fullH.toDouble()),
                doubleArrayOf(0.0, fullH.toDouble())
            )
        }

        srcFull.release(); src.release(); rgb.release(); gray.release(); hsv.release()
        bitmap.recycle()

        emitCaptured(path, fullW, fullH, corners, detector, segmenter.status, segmenter.lastError)
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
        segError: String?
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
        event.putArray("corners", cornerArray)
        event.putString("detector", detector)
        event.putString("segStatus", segStatus)
        event.putString("segError", segError)
        dispatch("onDocumentCaptured", event)
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

    // ----- Overlay view ------------------------------------------------------

    private class OverlayView(context: android.content.Context) : View(context) {
        private var quad: Array<PointF>? = null
        private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#334CAF50")
            style = Paint.Style.FILL
        }
        private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#4CAF50")
            style = Paint.Style.FILL
        }

        fun setQuad(q: Array<PointF>?) {
            quad = q
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val q = quad ?: return
            if (q.size < 4) return
            val path = Path()
            path.moveTo(q[0].x, q[0].y)
            for (i in 1 until q.size) path.lineTo(q[i].x, q[i].y)
            path.close()
            canvas.drawPath(path, fillPaint)
            canvas.drawPath(path, linePaint)
            if (q.size==4) for (p in q) canvas.drawCircle(p.x, p.y, 10f, dotPaint)
        }
    }
}
