package com.docscannerrn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * ML-based dewarping for curled/warped/bent pages, using UVDoc
 * (Verhoeven, Magne & Sorkine-Hornung, SIGGRAPH Asia 2023) — a small
 * (8M-param) grid-prediction CNN. Only the CNN forward pass runs through
 * ONNX Runtime; the actual pixel remap is done with plain OpenCV (the same
 * approach the paper's own reference code uses via PyTorch's grid_sample —
 * OpenCV's Imgproc.remap is the equivalent operation), so there's no exotic
 * op for the mobile runtime to support.
 *
 * Pipeline per photo:
 *   1. Resize input to the model's fixed 488x712 training resolution, normalize to [0,1]
 *   2. Run the ONNX model -> a small 31x45 grid of (x,y) sample coordinates in [-1,1]
 *   3. Upsample that grid to the ORIGINAL photo's full resolution (bilinear, corner-aligned)
 *   4. Convert normalized coords -> pixel coords in the original image
 *   5. Imgproc.remap the original full-resolution image using that pixel map
 *
 * This is a genuinely different capability from perspectiveCorrect() in
 * DocScannerModule: perspective correction only fixes a FLAT page shot at an
 * angle (4-corner homography). This fixes actual curvature/bending in the
 * page surface itself (e.g. a book page that won't lie flat). Run this
 * BEFORE perspectiveCorrect() if the page is both curled and at an angle.
 */
class DewarpModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val MODEL_INPUT_W = 488
        private const val MODEL_INPUT_H = 712
        private const val GRID_H = 45
        private const val GRID_W = 31
    }

    override fun getName(): String = "DewarpModule"

    private var cachedSession: OrtSession? = null
    private var cachedEnv: OrtEnvironment? = null

    private fun getSession(): OrtSession {
        cachedSession?.let { return it }

        val env = OrtEnvironment.getEnvironment()
        cachedEnv = env

        // Copy the model from assets to a real file the first time (ORT needs a path or byte array).
        val modelBytes = reactContext.assets.open("uvdoc.onnx").use { it.readBytes() }
        val session = env.createSession(modelBytes, OrtSession.SessionOptions())
        cachedSession = session
        return session
    }

    @ReactMethod
    fun dewarpDocument(imagePath: String, outputPath: String, promise: Promise) {
        try {
            val originalBitmap = BitmapFactory.decodeFile(imagePath)
                ?: return promise.reject("DECODE_FAILED", "Could not decode image at $imagePath")

            // 1. Preprocess: resize to the model's fixed training resolution, normalize to [0,1].
            val resized = Bitmap.createScaledBitmap(originalBitmap, MODEL_INPUT_W, MODEL_INPUT_H, true)
            val inputBuffer = bitmapToCHWFloatBuffer(resized)

            // 2. Run inference.
            val env = cachedEnv ?: OrtEnvironment.getEnvironment().also { cachedEnv = it }
            val session = getSession()
            val inputTensor = OnnxTensor.createTensor(
                env, inputBuffer, longArrayOf(1, 3, MODEL_INPUT_H.toLong(), MODEL_INPUT_W.toLong())
            )

            val results = session.run(mapOf("input_image" to inputTensor))
            // point_positions2D shape: [1, 2, GRID_H, GRID_W]
            @Suppress("UNCHECKED_CAST")
            val point_positions2D = (results[0].value as Array<Array<Array<FloatArray>>>)[0]
            inputTensor.close()
            results.close()

            val gridX = point_positions2D[0] // [GRID_H][GRID_W]
            val gridY = point_positions2D[1]

            // 3+4+5. Upsample grid to full-resolution + convert to pixel coords + remap.
            val outputBitmap = applyDewarpGrid(originalBitmap, gridX, gridY)

            val outFile = File(outputPath)
            outFile.parentFile?.mkdirs()
            FileOutputStream(outFile).use { outputBitmap.compress(Bitmap.CompressFormat.JPEG, 92, it) }

            promise.resolve(outputPath)
        } catch (e: Exception) {
            promise.reject("DEWARP_FAILED", e.message, e)
        }
    }

    /** Converts a Bitmap to a normalized [0,1] CHW float buffer (R plane, then G plane, then B plane). */
    private fun bitmapToCHWFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val buffer = FloatBuffer.allocate(3 * w * h)
        // R plane
        for (p in pixels) buffer.put((((p shr 16) and 0xFF) / 255f))
        // G plane
        for (p in pixels) buffer.put((((p shr 8) and 0xFF) / 255f))
        // B plane
        for (p in pixels) buffer.put(((p and 0xFF) / 255f))
        buffer.rewind()
        return buffer
    }

    /**
     * Upsamples the low-res (GRID_H x GRID_W) coordinate grid to the original image's
     * full resolution (bilinear, corner-aligned to match PyTorch's align_corners=True),
     * converts normalized [-1,1] coordinates to absolute pixel coordinates, and remaps
     * the original image using OpenCV — the exact equivalent of the reference
     * implementation's F.interpolate + F.grid_sample chain, validated numerically
     * against the original PyTorch code before this port.
     */
    private fun applyDewarpGrid(original: Bitmap, gridX: Array<FloatArray>, gridY: Array<FloatArray>): Bitmap {
        val imgW = original.width
        val imgH = original.height

        // Low-res grids as OpenCV Mats (CV_32F, single channel).
        val gridXMat = Mat(GRID_H, GRID_W, CvType.CV_32F)
        val gridYMat = Mat(GRID_H, GRID_W, CvType.CV_32F)
        for (y in 0 until GRID_H) {
            gridXMat.put(y, 0, gridX[y])
            gridYMat.put(y, 0, gridY[y])
        }

        // Corner-aligned upsample: build explicit sample-coordinate maps spanning
        // exactly [0, GRID_W-1] / [0, GRID_H-1] across the full image resolution,
        // then remap the low-res grid through them (equivalent to align_corners=True
        // bilinear interpolation).
        val upsampleMapX = Mat(imgH, imgW, CvType.CV_32F)
        val upsampleMapY = Mat(imgH, imgW, CvType.CV_32F)
        run {
            val rowX = FloatArray(imgW)
            for (x in 0 until imgW) {
                rowX[x] = if (imgW > 1) x.toFloat() * (GRID_W - 1) / (imgW - 1) else 0f
            }
            for (y in 0 until imgH) {
                upsampleMapX.put(y, 0, rowX)
                val yy = if (imgH > 1) y.toFloat() * (GRID_H - 1) / (imgH - 1) else 0f
                val rowY = FloatArray(imgW) { yy }
                upsampleMapY.put(y, 0, rowY)
            }
        }

        val upsampledGridX = Mat()
        val upsampledGridY = Mat()
        Imgproc.remap(gridXMat, upsampledGridX, upsampleMapX, upsampleMapY, Imgproc.INTER_LINEAR)
        Imgproc.remap(gridYMat, upsampledGridY, upsampleMapX, upsampleMapY, Imgproc.INTER_LINEAR)

        // Convert normalized [-1,1] coords -> absolute pixel coords in the ORIGINAL image
        // (align_corners=True convention: -1 -> pixel 0, +1 -> pixel size-1).
        val pixelMapX = Mat()
        val pixelMapY = Mat()
        org.opencv.core.Core.addWeighted(upsampledGridX, (imgW - 1) / 2.0, upsampledGridX, 0.0, (imgW - 1) / 2.0, pixelMapX)
        org.opencv.core.Core.addWeighted(upsampledGridY, (imgH - 1) / 2.0, upsampledGridY, 0.0, (imgH - 1) / 2.0, pixelMapY)

        val srcMat = Mat()
        Utils.bitmapToMat(original, srcMat)
        val dstMat = Mat()
        Imgproc.remap(srcMat, dstMat, pixelMapX, pixelMapY, Imgproc.INTER_LINEAR)

        val result = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dstMat, result)

        gridXMat.release(); gridYMat.release()
        upsampleMapX.release(); upsampleMapY.release()
        upsampledGridX.release(); upsampledGridY.release()
        pixelMapX.release(); pixelMapY.release()
        srcMat.release(); dstMat.release()

        return result
    }
}
