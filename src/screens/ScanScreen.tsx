import React, { useEffect } from 'react';
import { View, ActivityIndicator, Text, StyleSheet, Alert } from 'react-native';
import { DocScannerNative } from '../native/DocScannerNative';

/**
 * Captures a photo, then detects EVERY document-shaped region in it (handles
 * multiple stacked/side-by-side pages in one shot):
 *   - 0 found  -> fall back to the full image as a single document
 *   - 1 found  -> go straight to processing it
 *   - 2+ found -> let the user pick which one(s) via MultiDocPicker
 * Each detected document also reports whether it's a clean quadrilateral
 * (perspective-correct it) or a torn/irregular edge (crop-to-contour it).
 */
export default function ScanScreen({ navigation }: any) {
  useEffect(() => {
    scan();
  }, []);

  async function scan() {
    try {
      const rawPath = await DocScannerNative.capturePhoto();
      const detected = await DocScannerNative.detectAllDocuments(rawPath);

      if (detected.documents.length === 0) {
        // Nothing confidently detected — fall back to the full frame as a
        // single quadrilateral so the user can still adjust corners manually.
        navigation.replace('Crop', {
          rawPath,
          corners: [
            { x: 0, y: 0 },
            { x: detected.imageWidth, y: 0 },
            { x: detected.imageWidth, y: detected.imageHeight },
            { x: 0, y: detected.imageHeight },
          ],
          imageWidth: detected.imageWidth,
          imageHeight: detected.imageHeight,
        });
      } else if (detected.documents.length === 1) {
        const doc = detected.documents[0];
        if (doc.isQuadrilateral) {
          navigation.replace('Crop', {
            rawPath,
            corners: doc.points,
            imageWidth: detected.imageWidth,
            imageHeight: detected.imageHeight,
          });
        } else {
          navigation.replace('IrregularCrop', {
            rawPath,
            points: doc.points,
          });
        }
      } else {
        navigation.replace('MultiDocPicker', {
          rawPath,
          documents: detected.documents,
          imageWidth: detected.imageWidth,
          imageHeight: detected.imageHeight,
        });
      }
    } catch (e: any) {
      console.error('ScanScreen failed:', e);
      Alert.alert(
        'Scan failed',
        `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`,
        [{ text: 'OK', onPress: () => navigation.goBack() }]
      );
    }
  }

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#4CAF50" />
      <Text style={styles.text}>Opening camera…</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  text: { color: '#fff', marginTop: 12 },
});