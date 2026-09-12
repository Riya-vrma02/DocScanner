import React, { useRef, useState } from 'react';
import {
  View, Image, StyleSheet, PanResponder, Animated, TouchableOpacity, Text, Dimensions, ActivityIndicator,
} from 'react-native';
import { DocScannerNative } from '../native/DocScannerNative';
import { usePagesStore } from '../store/pagesStore';

/**
 * Same corner-adjustment UI as before, plus a new "Fix curl" step: if the
 * page is bent/warped (not just shot at an angle), tapping it runs the
 * ML dewarping model BEFORE perspective correction, then re-detects the
 * corners on the dewarped result. Perspective correction alone can't fix
 * actual page curvature — it only handles a flat page photographed at an
 * angle — so this is a deliberately separate, opt-in step.
 */
export default function CropScreen({ route, navigation }: any) {
  const [rawPath, setRawPath] = useState(route.params.rawPath);
  const [imageWidth, setImageWidth] = useState(route.params.imageWidth);
  const [imageHeight, setImageHeight] = useState(route.params.imageHeight);
  const [dewarping, setDewarping] = useState(false);
  const [hasDewarped, setHasDewarped] = useState(false);

  const addPage = usePagesStore((s) => s.addPage);

  const screenW = Dimensions.get('window').width;
  const displayScale = screenW / imageWidth;
  const displayHeight = imageHeight * displayScale;

  const [corners, setCorners] = useState(() =>
    route.params.corners.map(
      (c: { x: number; y: number }) =>
        new Animated.ValueXY({ x: c.x * displayScale, y: c.y * displayScale })
    )
  );

  const panResponders = useRef(buildPanResponders(corners)).current;

  function buildPanResponders(cornerList: Animated.ValueXY[]) {
    return cornerList.map((corner: Animated.ValueXY) =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onPanResponderGrant: () => {
          corner.setOffset({ x: (corner.x as any)._value, y: (corner.y as any)._value });
          corner.setValue({ x: 0, y: 0 });
        },
        onPanResponderMove: Animated.event([null, { dx: corner.x, dy: corner.y }], {
          useNativeDriver: false,
        }),
        onPanResponderRelease: () => corner.flattenOffset(),
      })
    );
  }

  async function onFixCurl() {
    setDewarping(true);
    try {
      const cacheDir = await DocScannerNative.getCacheDirectory();
      const dewarpedPath = `${cacheDir}/dewarped_${Date.now()}.jpg`;
      await DocScannerNative.dewarpDocument(rawPath, dewarpedPath);

      // Re-run edge detection on the now-flattened image so the corner
      // handles line up with the new (dewarped) picture.
      const detected = await DocScannerNative.detectDocumentCorners(dewarpedPath);

      setRawPath(dewarpedPath);
      setImageWidth(detected.imageWidth);
      setImageHeight(detected.imageHeight);
      setHasDewarped(true);

      const newScale = screenW / detected.imageWidth;
      const newCorners = detected.corners.map(
        (c: { x: number; y: number }) => new Animated.ValueXY({ x: c.x * newScale, y: c.y * newScale })
      );
      setCorners(newCorners);
      panResponders.splice(0, panResponders.length, ...buildPanResponders(newCorners));
    } catch (e) {
      // If dewarping fails, just keep the original photo — perspective
      // correction on the un-dewarped image still works for flat pages.
    } finally {
      setDewarping(false);
    }
  }

  async function onConfirm() {
    const imageCorners = corners.map((c: Animated.ValueXY) => ({
      x: (c.x as any)._value / displayScale,
      y: (c.y as any)._value / displayScale,
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
  }

  return (
    <View style={styles.container}>
      <View style={{ width: screenW, height: displayHeight }}>
        <Image
          source={{ uri: `file://${rawPath}` }}
          style={{ width: screenW, height: displayHeight }}
          resizeMode="contain"
        />
        {corners.map((corner: Animated.ValueXY, i: number) => (
          <Animated.View
            key={i}
            {...panResponders[i].panHandlers}
            style={[styles.handle, { transform: corner.getTranslateTransform() }]}
          />
        ))}
        {dewarping && (
          <View style={styles.dewarpOverlay}>
            <ActivityIndicator size="large" color="#fff" />
            <Text style={styles.dewarpText}>Flattening page…</Text>
          </View>
        )}
      </View>

      <View style={styles.buttonBar}>
        {!hasDewarped && (
          <TouchableOpacity style={styles.curlButton} onPress={onFixCurl} disabled={dewarping}>
            <Text style={styles.buttonText}>Page is bent/curled — Fix it</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.confirmButton} onPress={onConfirm} disabled={dewarping}>
          <Text style={styles.buttonText}>Confirm</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  handle: {
    position: 'absolute',
    width: 32, height: 32, borderRadius: 16,
    backgroundColor: '#4CAF50', opacity: 0.8,
    borderWidth: 2, borderColor: '#fff',
    marginLeft: -16, marginTop: -16,
  },
  dewarpOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#000000AA',
    alignItems: 'center', justifyContent: 'center',
  },
  dewarpText: { color: '#fff', marginTop: 12 },
  buttonBar: { padding: 16, gap: 8 },
  curlButton: { backgroundColor: '#F57C00', padding: 14, borderRadius: 8, alignItems: 'center' },
  confirmButton: { backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
