import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
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
  const { pageId } = route.params;
  const page = usePagesStore((s) => s.pages.find((p) => p.id === pageId))!;
  const updatePage = usePagesStore((s) => s.updatePage);

  const [filterType, setFilterType] = useState<FilterType>(page.filterType);
  const [rotation, setRotation] = useState(page.rotationDegrees);
  const [brightness, setBrightness] = useState(page.brightness);
  const [contrast, setContrast] = useState(page.contrast);
  const [previewPath, setPreviewPath] = useState(page.processedImagePath);
  const [rendering, setRendering] = useState(false);

  const rerender = useCallback(
    async (nextFilter = filterType, nextRotation = rotation, nextB = brightness, nextC = contrast) => {
      setRendering(true);
      try {
        const cacheDir = await DocScannerNative.getCacheDirectory();
        const previewFile = `${cacheDir}/preview_${pageId}.jpg`;
        await DocScannerNative.applyFilter(
          page.croppedImagePath, nextFilter, nextRotation, nextB, nextC, previewFile
        );
        setPreviewPath(`${previewFile}?t=${Date.now()}`);
      } finally {
        setRendering(false);
      }
    },
    [filterType, rotation, brightness, contrast, page.croppedImagePath, pageId]
  );

  function onSelectFilter(f: FilterType) {
    setFilterType(f);
    rerender(f, rotation, brightness, contrast);
  }

  function onRotate() {
    const next = (rotation + 90) % 360;
    setRotation(next);
    rerender(filterType, next, brightness, contrast);
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
  bottomRow: { flexDirection: 'row', padding: 12, gap: 8 },
  primaryButton: { flex: 1, backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  secondaryButton: { flex: 1, backgroundColor: '#757575', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
