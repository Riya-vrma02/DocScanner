import React, { useRef, useState } from 'react';
import {
  View, Image, StyleSheet, PanResponder, TouchableOpacity, Text, LayoutChangeEvent, Alert,
} from 'react-native';
import { DocScannerNative } from '../native/DocScannerNative';
import { usePagesStore } from '../store/pagesStore';

/**
 * Corner-adjustment UI. The image is scaled to fit the available area (both
 * width AND height) and centered, so the button bar below always stays on
 * screen even for tall portrait photos.
 *
 * While a corner is being dragged, a circular MAGNIFIER (loupe) appears in a
 * top corner of the screen showing a zoomed-in view centered on that corner,
 * with a crosshair — so your fingertip doesn't hide exactly where the corner
 * is landing.
 *
 * NOTE: page-flattening (dewarp / "fix curl") happens on the FilterScreen,
 * AFTER this perspective crop — the UVDoc model expects the document to fill
 * the frame, so it must run on the cropped page, not the raw full photo.
 */
const LOUPE_SIZE = 150;
const LOUPE_ZOOM = 2.4;

export default function CropScreen({ route, navigation }: any) {
  const rawPath = route.params.rawPath as string;
  const imageWidth = route.params.imageWidth as number;
  const imageHeight = route.params.imageHeight as number;

  const [area, setArea] = useState<{ w: number; h: number } | null>(null);
  const [corners, setCorners] = useState<{ x: number; y: number }[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const addPage = usePagesStore((s) => s.addPage);

  const scale = area ? Math.min(area.w / imageWidth, area.h / imageHeight) : 0;
  const dispW = imageWidth * scale;
  const dispH = imageHeight * scale;
  const offsetX = area ? (area.w - dispW) / 2 : 0;
  const offsetY = area ? (area.h - dispH) / 2 : 0;

  const cornersRef = useRef<{ x: number; y: number }[]>([]);
  const boundsRef = useRef({ minX: 0, minY: 0, maxX: 0, maxY: 0 });
  const dragStartRef = useRef({ x: 0, y: 0 });
  const respondersRef = useRef<any[]>([]);

  function clamp(v: number, lo: number, hi: number) {
    return Math.max(lo, Math.min(hi, v));
  }

  function setCorner(i: number, p: { x: number; y: number }) {
    const next = cornersRef.current.slice();
    next[i] = p;
    cornersRef.current = next;
    setCorners(next);
  }

  function buildResponders(count: number) {
    return Array.from({ length: count }).map((_, i) =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderGrant: () => {
          setActiveIndex(i);
          dragStartRef.current = cornersRef.current[i];
        },
        onPanResponderMove: (_evt, g) => {
          const b = boundsRef.current;
          const start = dragStartRef.current;
          setCorner(i, {
            x: clamp(start.x + g.dx, b.minX, b.maxX),
            y: clamp(start.y + g.dy, b.minY, b.maxY),
          });
        },
        onPanResponderRelease: () => setActiveIndex(null),
        onPanResponderTerminate: () => setActiveIndex(null),
      })
    );
  }

  function onAreaLayout(e: LayoutChangeEvent) {
    if (area) return; // measure once
    const { width, height } = e.nativeEvent.layout;
    const s = Math.min(width / imageWidth, height / imageHeight);
    const ox = (width - imageWidth * s) / 2;
    const oy = (height - imageHeight * s) / 2;

    const placed = (route.params.corners as { x: number; y: number }[]).map((c) => ({
      x: ox + c.x * s,
      y: oy + c.y * s,
    }));
    cornersRef.current = placed;
    boundsRef.current = { minX: ox, minY: oy, maxX: ox + imageWidth * s, maxY: oy + imageHeight * s };
    respondersRef.current = buildResponders(placed.length);
    setArea({ w: width, h: height });
    setCorners(placed);
  }

  async function onConfirm() {
    try {
      const imageCorners = cornersRef.current.map((c) => ({
        x: (c.x - offsetX) / scale,
        y: (c.y - offsetY) / scale,
      }));

      const cacheDir = await DocScannerNative.getCacheDirectory();
      const correctedPath = `${cacheDir}/corrected_${Date.now()}.jpg`;
      await DocScannerNative.perspectiveCorrect(rawPath, imageCorners, correctedPath);

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
            {corners.map((c, i) => (
              <View
                key={i}
                {...(respondersRef.current[i]?.panHandlers ?? {})}
                style={[styles.handle, { left: c.x, top: c.y }, activeIndex === i && styles.handleActive]}
              >
                <View style={styles.handleDot} />
              </View>
            ))}
            {renderLoupe()}
          </>
        )}
      </View>

      <View style={styles.buttonBar}>
        <TouchableOpacity style={styles.confirmButton} onPress={onConfirm} disabled={corners.length === 0}>
          <Text style={styles.buttonText}>Confirm</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  imageArea: { flex: 1 },
  handle: {
    position: 'absolute',
    width: 44, height: 44,
    marginLeft: -22, marginTop: -22,
    alignItems: 'center', justifyContent: 'center',
  },
  handleActive: { opacity: 0.6 },
  handleDot: {
    width: 22, height: 22, borderRadius: 11,
    backgroundColor: '#4CAF50',
    borderWidth: 2, borderColor: '#fff',
  },
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
  confirmButton: { backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
