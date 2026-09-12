import React, { useEffect, useState } from 'react';
import { View, Image, StyleSheet, TouchableOpacity, Text, ActivityIndicator } from 'react-native';
import { DocScannerNative } from '../native/DocScannerNative';
import { usePagesStore } from '../store/pagesStore';

/**
 * For documents detectAllDocuments() flagged as torn/irregular-edged (not a
 * clean quadrilateral). There's no well-defined "4 corners" to drag/adjust
 * for a torn edge, so this auto-crops to the detected contour immediately
 * and shows the result for confirmation, rather than an interactive handle
 * UI. If the auto-detected shape looks wrong, "Use full photo instead"
 * falls back to the plain rectangular photo untouched.
 */
export default function IrregularCropScreen({ route, navigation }: any) {
  const { rawPath, points } = route.params;
  const addPage = usePagesStore((s) => s.addPage);
  const [croppedPath, setCroppedPath] = useState<string | null>(null);

  useEffect(() => {
    autoCrop();
  }, []);

  async function autoCrop() {
    const cacheDir = await DocScannerNative.getCacheDirectory();
    const outPath = `${cacheDir}/torn_crop_${Date.now()}.jpg`;
    await DocScannerNative.cropToContour(rawPath, points, outPath);
    setCroppedPath(outPath);
  }

  function proceedWith(path: string) {
    const id = `page_${Date.now()}`;
    addPage({
      id,
      croppedImagePath: path,
      processedImagePath: path,
      rotationDegrees: 0,
      filterType: 'AUTO_ENHANCE',
      brightness: 0,
      contrast: 0,
    });
    navigation.replace('Filter', { pageId: id, isNewPage: true });
  }

  if (!croppedPath) {
    return (
      <View style={styles.loading}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.loadingText}>Cropping to detected edge…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Torn/irregular edge detected — auto-cropped result:</Text>
      <Image source={{ uri: `file://${croppedPath}` }} style={styles.preview} resizeMode="contain" />

      <View style={styles.buttonBar}>
        <TouchableOpacity style={styles.secondaryButton} onPress={() => proceedWith(rawPath)}>
          <Text style={styles.buttonText}>Use full photo instead</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.primaryButton} onPress={() => proceedWith(croppedPath)}>
          <Text style={styles.buttonText}>Looks good — Continue</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  loading: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  loadingText: { color: '#fff', marginTop: 12 },
  title: { padding: 12, textAlign: 'center', color: '#555' },
  preview: { flex: 1, backgroundColor: '#eee' },
  buttonBar: { flexDirection: 'row', padding: 12, gap: 8 },
  primaryButton: { flex: 1, backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  secondaryButton: { flex: 1, backgroundColor: '#757575', padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
