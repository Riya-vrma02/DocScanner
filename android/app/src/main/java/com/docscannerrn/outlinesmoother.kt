package com.docscannerrn

import android.graphics.PointF
import kotlin.math.hypot

/**
 * Stabilises the live outline so it stops flickering.
 *
 * Each frame's detection can have a different number of points (a clean page
 * snaps to 4 corners, a torn one gives 10-20) and a different starting point,
 * so frames can't be averaged directly. This class:
 *   1. resamples every outline to [N] evenly spaced points along its perimeter,
 *   2. rotates it so point i lines up with point i of the previous frame,
 *   3. holds still for tiny movements and blends for larger ones,
 *   4. snaps (no blending) when the outline jumps far, e.g. a different page.
 *
 * Only used for the on-screen overlay. The capture step runs its own fresh
 * detection, so smoothing never affects the saved crop.
 *
 * Not thread-safe: call from the analysis thread only.
 */
class OutlineSmoother {

    companion object {
        private const val N = 96
        private const val DEADBAND_PX = 3f   // below this average movement, keep the old outline
        private const val JUMP_PX = 150f     // above this, treat it as a new page and snap
    }

    private var prev: Array<PointF>? = null

    fun reset() {
        prev = null
    }

    fun update(raw: Array<PointF>): Array<PointF> {
        if (raw.size < 3) return raw
        val cur = resample(clockwise(raw), N)
        val p = prev
        if (p == null) {
            prev = cur
            return cur
        }

        // Find the rotation of `cur` that best matches the previous outline.
        var bestShift = 0
        var bestCost = Float.MAX_VALUE
        for (s in 0 until N) {
            var cost = 0f
            for (i in 0 until N) {
                val a = cur[(i + s) % N]
                val b = p[i]
                cost += hypot(a.x - b.x, a.y - b.y)
            }
            if (cost < bestCost) {
                bestCost = cost
                bestShift = s
            }
        }
        val meanDist = bestCost / N

        if (meanDist > JUMP_PX) {
            prev = cur
            return cur
        }
        if (meanDist < DEADBAND_PX) return p

        // Small moves are blended gently, larger moves follow faster.
        val alpha = (meanDist / 40f).coerceIn(0.15f, 0.6f)
        val out = Array(N) { i ->
            val a = cur[(i + bestShift) % N]
            val b = p[i]
            PointF(b.x + (a.x - b.x) * alpha, b.y + (a.y - b.y) * alpha)
        }
        prev = out
        return out
    }

    /** Makes the winding direction consistent between frames. */
    private fun clockwise(pts: Array<PointF>): Array<PointF> {
        var area = 0f
        for (i in pts.indices) {
            val a = pts[i]
            val b = pts[(i + 1) % pts.size]
            area += a.x * b.y - b.x * a.y
        }
        return if (area >= 0f) pts else pts.reversedArray()
    }

    /** Evenly spaced points along the closed outline. */
    private fun resample(pts: Array<PointF>, n: Int): Array<PointF> {
        val m = pts.size
        val lens = FloatArray(m)
        var total = 0f
        for (i in 0 until m) {
            val a = pts[i]
            val b = pts[(i + 1) % m]
            lens[i] = hypot(b.x - a.x, b.y - a.y)
            total += lens[i]
        }
        if (total <= 0f) return Array(n) { PointF(pts[0].x, pts[0].y) }

        val step = total / n
        var seg = 0
        var segStart = 0f
        return Array(n) { k ->
            val d = k * step
            while (seg < m - 1 && segStart + lens[seg] < d) {
                segStart += lens[seg]
                seg++
            }
            val a = pts[seg]
            val b = pts[(seg + 1) % m]
            val t = if (lens[seg] > 0f) ((d - segStart) / lens[seg]).coerceIn(0f, 1f) else 0f
            PointF(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        }
    }
}