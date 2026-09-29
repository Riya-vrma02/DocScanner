import React, { useState, useCallback, useRef } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, Alert } from 'react-native';
import Slider from '@react-native-community/slider';
import { usePagesStore } from '../store/pagesStore';
import { DocScannerNative } from '../native/DocScannerNative';
import { FilterType } from '../types';

const FILTERS: { label: string; value: FilterType }[] = [
  { label: 'Original', value: 'ORIGINAL' },
  { label: 'Auto', value: 'AUTO_ENHANCE' },
  { label: 'Gray', value: 'GRAYSCALE' },
  { label: 'B&W', value: 'BLACK_AND_WHITE' },
];

/**
 * Same UI/UX as before, but every render now calls the native applyFilter()
 * method (android.graphics ColorMatrix) instead of Skia.
 */
export default function FilterScreen({ route, navigation }: any) {
  const pageId: string | undefined = route.params?.pageId;
  const page = usePagesStore((s) => s.pages.find((p) => p.id === pageId));
  const updatePage = usePagesStore((s) => s.updatePage);

  // Hooks must run unconditionally, so fall back to safe defaults when the
  // page is missing; the guard below then renders a fallback instead.
  const [filterType, setFilterType] = useState<FilterType>(page?.filterType ?? 'AUTO_ENHANCE');
  const [rotation, setRotation] = useState(page?.rotationDegrees ?? 0);
  const [brightness, setBrightness] = useState(page?.brightness ?? 0);
  const [contrast, setContrast] = useState(page?.contrast ?? 0);
  const [previewPath, setPreviewPath] = useState(page?.processedImagePath ?? '');
  const [rendering, setRendering] = useState(false);
  const [flattened, setFlattened] = useState(false);
  // The cropped-image path from before flattening, so "Undo" can restore it.
  const preFlattenPathRef = useRef<string | null>(null);

  const rerender = useCallback(
    async (nextFilter = filterType, nextRotation = rotation, nextB = brightness, nextC = contrast) => {
      setRendering(true);
      try {
        const cacheDir = await DocScannerNative.getCacheDirectory();
        const previewFile = `${cacheDir}/preview_${pageId}.jpg`;
        await DocScannerNative.applyFilter(
          page!.croppedImagePath, nextFilter, nextRotation, nextB, nextC, previewFile
        );
        setPreviewPath(`${previewFile}?t=${Date.now()}`);
      } finally {
        setRendering(false);
      }
    },
    [filterType, rotation, brightness, contrast, page?.croppedImagePath, pageId]
  );

  // Guard: if we somehow landed here without a valid page (e.g. stale
  // navigation state after a hot reload), show a fallback instead of crashing.
  if (!page) {
    return (
      <View style={[styles.container, { alignItems: 'center', justifyContent: 'center', padding: 24 }]}>
        <Text style={{ color: '#555', textAlign: 'center', marginBottom: 16 }}>
          This page is no longer available.
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={() => navigation.navigate('Home')}>
          <Text style={styles.buttonText}>Back to Home</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function onSelectFilter(f: FilterType) {
    setFilterType(f);
    rerender(f, rotation, brightness, contrast);
  }

  function onRotate() {
    const next = (rotation + 90) % 360;
    setRotation(next);
    rerender(filterType, next, brightness, contrast);
  }

  /**
   * ML dewarp for a curled/bent page. Runs on the already-cropped page (which
   * fills the frame) — the UVDoc model needs the document to fill the image,
   * so running it here rather than on the raw full-frame photo avoids warping
   * the background into waves. The flattened image replaces the crop as the
   * new base for filters, so it can't be applied twice.
   */
  async function onFlatten() {
    setRendering(true);
    try {
      const cacheDir = await DocScannerNative.getCacheDirectory();
      const flatPath = `${cacheDir}/flattened_${pageId}_${Date.now()}.jpg`;
      // Remember the pre-flatten image so it can be restored via Undo.
      preFlattenPathRef.current = page.croppedImagePath;
      await DocScannerNative.dewarpDocument(page.croppedImagePath, flatPath);
      updatePage(pageId, { croppedImagePath: flatPath });
      setFlattened(true);

      const previewFile = `${cacheDir}/preview_${pageId}.jpg`;
      await DocScannerNative.applyFilter(flatPath, filterType, rotation, brightness, contrast, previewFile);
      setPreviewPath(`${previewFile}?t=${Date.now()}`);
    } catch (e: any) {
      Alert.alert('Flatten failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    } finally {
      setRendering(false);
    }
  }

  /** Reverts a flatten, restoring the cropped image from before dewarping. */
  async function onUndoFlatten() {
    const original = preFlattenPathRef.current;
    if (!original) return;
    setRendering(true);
    try {
      updatePage(pageId, { croppedImagePath: original });
      setFlattened(false);

      const cacheDir = await DocScannerNative.getCacheDirectory();
      const previewFile = `${cacheDir}/preview_${pageId}.jpg`;
      await DocScannerNative.applyFilter(original, filterType, rotation, brightness, contrast, previewFile);
      setPreviewPath(`${previewFile}?t=${Date.now()}`);
    } catch (e: any) {
      Alert.alert('Undo failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    } finally {
      setRendering(false);
    }
  }

  async function onDone() {
    const docDir = await DocScannerNative.getDocumentDirectory();
    await DocScannerNative.mkdir(`${docDir}/scans`);
    const finalPath = `${docDir}/scans/${pageId}.jpg`;

    await DocScannerNative.applyFilter(
      page.croppedImagePath, filterType, rotation, brightness, contrast, finalPath
    );

    updatePage(pageId, {
      processedImagePath: finalPath,
      rotationDegrees: rotation,
      filterType,
      brightness,
      contrast,
    });
    navigation.navigate('Home');
  }

  return (
    <View style={styles.container}>
      <Image source={{ uri: `file://${previewPath}` }} style={styles.preview} resizeMode="contain" />
      {rendering && <Text style={styles.renderingLabel}>Updating preview…</Text>}

      <View style={styles.filterRow}>
        {FILTERS.map((f) => (
          <TouchableOpacity
            key={f.value}
            style={[styles.filterChip, filterType === f.value && styles.filterChipActive]}
            onPress={() => onSelectFilter(f.value)}
          >
            <Text style={filterType === f.value ? styles.filterTextActive : styles.filterText}>{f.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={styles.sliderLabel}>Brightness</Text>
      <Slider
        value={brightness}
        minimumValue={-1}
        maximumValue={1}
        onSlidingComplete={(v) => { setBrightness(v); rerender(filterType, rotation, v, contrast); }}
      />

      <Text style={styles.sliderLabel}>Contrast</Text>
      <Slider
        value={contrast}
        minimumValue={-1}
        maximumValue={1}
        onSlidingComplete={(v) => { setContrast(v); rerender(filterType, rotation, brightness, v); }}
      />

      <TouchableOpacity
        style={[styles.flattenButton, rendering && styles.flattenButtonDisabled, flattened && styles.undoButton]}
        onPress={flattened ? onUndoFlatten : onFlatten}
        disabled={rendering}
      >
        <Text style={styles.buttonText}>{flattened ? '↩ Undo flatten' : 'Flatten curled page'}</Text>
      </TouchableOpacity>

      <View style={styles.bottomRow}>
        <TouchableOpacity style={styles.secondaryButton} onPress={onRotate}>
          <Text style={styles.buttonText}>Rotate 90°</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.primaryButton} onPress={onDone}>
          <Text style={styles.buttonText}>Done</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  preview: { flex: 1, backgroundColor: '#eee' },
  renderingLabel: { position: 'absolute', top: 12, alignSelf: 'center', color: '#888' },
  filterRow: { flexDirection: 'row', justifyContent: 'space-around', padding: 8 },
  filterChip: { paddingVertical: 8, paddingHorizontal: 16, borderRadius: 20, borderWidth: 1, borderColor: '#4CAF50' },
  filterChipActive: { backgroundColor: '#4CAF50' },
  filterText: { color: '#4CAF50' },
  filterTextActive: { color: '#fff' },
  sliderLabel: { paddingHorizontal: 16, color: '#555', marginTop: 4 },
  flattenButton: { backgroundColor: '#F57C00', marginHorizontal: 12, marginTop: 8, padding: 12, borderRadius: 8, alignItems: 'center' },
  flattenButtonDisabled: { opacity: 0.4 },
  undoButton: { backgroundColor: '#757575' },
  bottomRow: { flexDirection: 'row', padding: 12, gap: 8 },
  primaryButton: { flex: 1, backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  secondaryButton: { flex: 1, backgroundColor: '#757575', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
