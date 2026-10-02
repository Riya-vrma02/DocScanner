import React, { useRef, useState } from 'react';
import {
  View, Image, StyleSheet, PanResponder, TouchableOpacity, Text, LayoutChangeEvent, Alert,
} from 'react-native';
import Svg, { Path, Polygon, Circle } from 'react-native-svg';
import { DocScannerNative } from '../native/DocScannerNative';
import { usePagesStore } from '../store/pagesStore';

/**
 * Polygon-adjustment UI. Auto-detection hands us an outline with any number of
 * points (route.params.corners); the user can fix it before cropping.
 *
 *  - Drag a green dot to move that point.
 *  - Drag a white midpoint dot to add a new point on that edge (for curves/tears).
 *  - Double-tap a green dot to remove it (minimum MIN_POINTS).
 *
 * While a point is dragged, a circular MAGNIFIER (loupe) shows a zoomed view
 * around it so your fingertip doesn't hide where it lands.
 *
 * Confirm:
 *  - exactly 4 points -> DocScannerNative.perspectiveCorrect (flat crop, as before)
 *  - more than 4      -> DocScannerNative.cropToContour (irregular cutout, white outside)
 *
 * NOTE: page-flattening (dewarp / "fix curl") still happens on the FilterScreen,
 * AFTER this crop. The UVDoc model expects the page to fill the frame, so it works
 * best on the 4-point result.
 */

type Pt = { x: number; y: number };
type Hit = { type: 'vertex' | 'mid'; index: number };

const LOUPE_SIZE = 150;
const LOUPE_ZOOM = 2.4;
const HIT_RADIUS = 30;
const MIN_POINTS = 4;
const DOUBLE_TAP_MS = 300;
const TAP_SLOP = 6;

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

export default function CropScreen({ route, navigation }: any) {
  const rawPath = route.params.rawPath as string;
  const imageWidth = route.params.imageWidth as number;
  const imageHeight = route.params.imageHeight as number;

  const [area, setArea] = useState<{ w: number; h: number } | null>(null);
  const [corners, setCorners] = useState<Pt[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const addPage = usePagesStore((s) => s.addPage);

  const scale = area ? Math.min(area.w / imageWidth, area.h / imageHeight) : 0;
  const dispW = imageWidth * scale;
  const dispH = imageHeight * scale;
  const offsetX = area ? (area.w - dispW) / 2 : 0;
  const offsetY = area ? (area.h - dispH) / 2 : 0;

  // Gesture handlers read refs only, so they never see stale state.
  const cornersRef = useRef<Pt[]>([]);
  const initialRef = useRef<Pt[]>([]);
  const boundsRef = useRef({ minX: 0, minY: 0, maxX: 0, maxY: 0 });
  const dragRef = useRef<{ index: number; start: Pt; moved: boolean } | null>(null);
  const lastTapRef = useRef<{ index: number; t: number } | null>(null);

  function update(next: Pt[]) {
    cornersRef.current = next;
    setCorners(next);
  }

  function hitTest(p: Pt): Hit | null {
    const pts = cornersRef.current;
    let best: Hit | null = null;
    let bestD = HIT_RADIUS;
    for (let i = 0; i < pts.length; i++) {
      const d = Math.hypot(pts[i].x - p.x, pts[i].y - p.y);
      if (d < bestD) { bestD = d; best = { type: 'vertex', index: i }; }
    }
    if (best) return best;
    for (let i = 0; i < pts.length; i++) {
      const n = pts[(i + 1) % pts.length];
      const d = Math.hypot((pts[i].x + n.x) / 2 - p.x, (pts[i].y + n.y) / 2 - p.y);
      if (d < bestD) { bestD = d; best = { type: 'mid', index: i }; }
    }
    return best;
  }

  function finishDrag() {
    const d = dragRef.current;
    if (d && !d.moved) {
      // A tap on a point: two quick taps on the same point delete it.
      const now = Date.now();
      const last = lastTapRef.current;
      if (
        last && last.index === d.index && now - last.t < DOUBLE_TAP_MS &&
        cornersRef.current.length > MIN_POINTS
      ) {
        update(cornersRef.current.filter((_, i) => i !== d.index));
        lastTapRef.current = null;
      } else {
        lastTapRef.current = { index: d.index, t: now };
      }
    }
    dragRef.current = null;
    setActiveIndex(null);
  }

  // One responder for the whole image area; it decides which point was touched.
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (evt) => {
        const p = { x: evt.nativeEvent.locationX, y: evt.nativeEvent.locationY };
        const hit = hitTest(p);
        if (!hit) { dragRef.current = null; return; }
        let index = hit.index;
        if (hit.type === 'mid') {
          const pts = cornersRef.current;
          const a = pts[index];
          const b = pts[(index + 1) % pts.length];
          const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
          update([...pts.slice(0, index + 1), mid, ...pts.slice(index + 1)]);
          index += 1;
        }
        dragRef.current = {
          index,
          start: cornersRef.current[index],
          moved: hit.type === 'mid', // an inserted point is never treated as a tap
        };
        setActiveIndex(index);
      },
      onPanResponderMove: (_evt, g) => {
        const d = dragRef.current;
        if (!d) return;
        if (Math.abs(g.dx) > TAP_SLOP || Math.abs(g.dy) > TAP_SLOP) d.moved = true;
        const b = boundsRef.current;
        const pts = cornersRef.current.slice();
        pts[d.index] = {
          x: clamp(d.start.x + g.dx, b.minX, b.maxX),
          y: clamp(d.start.y + g.dy, b.minY, b.maxY),
        };
        update(pts);
      },
      onPanResponderRelease: finishDrag,
      onPanResponderTerminate: finishDrag,
    })
  ).current;

  function onAreaLayout(e: LayoutChangeEvent) {
    if (area) return; // measure once
    const { width, height } = e.nativeEvent.layout;
    const s = Math.min(width / imageWidth, height / imageHeight);
    const ox = (width - imageWidth * s) / 2;
    const oy = (height - imageHeight * s) / 2;

    const placed = (route.params.corners as Pt[]).map((c) => ({
      x: ox + c.x * s,
      y: oy + c.y * s,
    }));
    initialRef.current = placed;
    boundsRef.current = { minX: ox, minY: oy, maxX: ox + imageWidth * s, maxY: oy + imageHeight * s };
    setArea({ w: width, h: height });
    update(placed);
  }

  async function onConfirm() {
    try {
      const imageCorners = cornersRef.current.map((c) => ({
        x: (c.x - offsetX) / scale,
        y: (c.y - offsetY) / scale,
      }));

      const cacheDir = await DocScannerNative.getCacheDirectory();
      const correctedPath = `${cacheDir}/corrected_${Date.now()}.jpg`;

      if (imageCorners.length === 4) {
        await DocScannerNative.perspectiveCorrect(rawPath, imageCorners, correctedPath);
      } else {
        await DocScannerNative.cropToContour(rawPath, imageCorners, correctedPath);
      }

      const id = `page_${Date.now()}`;
      addPage({
        id,
        croppedImagePath: correctedPath,
        processedImagePath: correctedPath,
        rotationDegrees: 0,
        filterType: 'AUTO_ENHANCE',
        brightness: 0,
        contrast: 0,
      });

      navigation.replace('Filter', { pageId: id, isNewPage: true });
    } catch (e: any) {
      Alert.alert('Crop failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    }
  }

  function renderLoupe() {
    if (activeIndex === null || !area) return null;
    const c = corners[activeIndex];
    if (!c) return null;

    const relX = c.x - offsetX;
    const relY = c.y - offsetY;
    const imgLeft = LOUPE_SIZE / 2 - relX * LOUPE_ZOOM;
    const imgTop = LOUPE_SIZE / 2 - relY * LOUPE_ZOOM;

    const onLeftHalf = c.x < area.w / 2;
    const loupePos = onLeftHalf ? { right: 16 } : { left: 16 };

    return (
      <View style={[styles.loupe, loupePos]} pointerEvents="none">
        <Image
          source={{ uri: `file://${rawPath}` }}
          style={{ position: 'absolute', left: imgLeft, top: imgTop, width: dispW * LOUPE_ZOOM, height: dispH * LOUPE_ZOOM }}
          resizeMode="stretch"
        />
        <View style={styles.loupeCrosshairV} />
        <View style={styles.loupeCrosshairH} />
      </View>
    );
  }

  const outline = corners.map((c, i) => `${i ? 'L' : 'M'}${c.x},${c.y}`).join(' ') + ' Z';
  const polyPoints = corners.map((c) => `${c.x},${c.y}`).join(' ');

  return (
    <View style={styles.container}>
      <View style={styles.imageArea} onLayout={onAreaLayout}>
        {area && (
          <>
            <Image
              source={{ uri: `file://${rawPath}` }}
              style={{ position: 'absolute', left: offsetX, top: offsetY, width: dispW, height: dispH }}
              resizeMode="stretch"
            />

            <Svg width={area.w} height={area.h} style={StyleSheet.absoluteFill} pointerEvents="none">
              {/* dim everything outside the outline */}
              <Path
                d={`M0,0 H${area.w} V${area.h} H0 Z ${outline}`}
                fill="rgba(0,0,0,0.45)"
                fillRule="evenodd"
              />
              <Polygon points={polyPoints} fill="none" stroke="#4CAF50" strokeWidth={2} />

              {/* midpoint handles: drag to add a point */}
              {corners.map((c, i) => {
                const n = corners[(i + 1) % corners.length];
                return (
                  <Circle
                    key={`m${i}`}
                    cx={(c.x + n.x) / 2}
                    cy={(c.y + n.y) / 2}
                    r={7}
                    fill="#fff"
                    fillOpacity={0.85}
                    stroke="#4CAF50"
                    strokeWidth={2}
                  />
                );
              })}

              {/* vertex handles: drag to move, double-tap to remove */}
              {corners.map((c, i) => (
                <Circle
                  key={`v${i}`}
                  cx={c.x}
                  cy={c.y}
                  r={11}
                  fill="#4CAF50"
                  stroke="#fff"
                  strokeWidth={2}
                  opacity={activeIndex === i ? 0.6 : 1}
                />
              ))}
            </Svg>

            {renderLoupe()}

            {/* touch layer on top of everything */}
            <View style={StyleSheet.absoluteFill} {...panResponder.panHandlers} />
          </>
        )}
      </View>

      <View style={styles.buttonBar}>
        <Text style={styles.hint}>
          Drag dots to fit the edges. Drag a white dot to add a point. Double-tap a point to remove it.
        </Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={styles.resetButton}
            onPress={() => update(initialRef.current)}
            disabled={corners.length === 0}
          >
            <Text style={styles.buttonText}>Reset</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.confirmButton} onPress={onConfirm} disabled={corners.length === 0}>
            <Text style={styles.buttonText}>Confirm</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  imageArea: { flex: 1 },
  loupe: {
    position: 'absolute',
    top: 16,
    width: LOUPE_SIZE, height: LOUPE_SIZE, borderRadius: LOUPE_SIZE / 2,
    overflow: 'hidden',
    borderWidth: 3, borderColor: '#fff',
    backgroundColor: '#000',
  },
  loupeCrosshairV: {
    position: 'absolute', left: LOUPE_SIZE / 2 - 0.5, top: 0, bottom: 0, width: 1,
    backgroundColor: '#4CAF50AA',
  },
  loupeCrosshairH: {
    position: 'absolute', top: LOUPE_SIZE / 2 - 0.5, left: 0, right: 0, height: 1,
    backgroundColor: '#4CAF50AA',
  },
  buttonBar: { padding: 16, gap: 8 },
  hint: { color: '#ffffffAA', fontSize: 12, textAlign: 'center' },
  buttonRow: { flexDirection: 'row', gap: 8 },
  resetButton: { flex: 1, backgroundColor: '#424242', padding: 14, borderRadius: 8, alignItems: 'center' },
  confirmButton: { flex: 2, backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
