import React, { useState } from 'react';
import {
  View, Image, StyleSheet, TouchableOpacity, Text, Dimensions, ActivityIndicator,
} from 'react-native';
import { DocScannerNative, DetectedDocument } from '../native/DocScannerNative';
import { usePagesStore } from '../store/pagesStore';

/**
 * Shown when detectAllDocuments() found more than one page in a single
 * photo (stacked or side-by-side documents). Draws a bounding box over each
 * detected region on top of the full photo; tap one to process just that
 * page, or "Add all as separate pages" to process every detected region in
 * one go (e.g. a photo of 3 receipts side by side -> 3 pages at once).
 */
export default function MultiDocPickerScreen({ route, navigation }: any) {
  const { rawPath, documents, imageWidth, imageHeight } = route.params as {
    rawPath: string; documents: DetectedDocument[]; imageWidth: number; imageHeight: number;
  };
  const addPage = usePagesStore((s) => s.addPage);
  const [processingAll, setProcessingAll] = useState(false);

  const screenW = Dimensions.get('window').width;
  const scale = screenW / imageWidth;
  const displayHeight = imageHeight * scale;

  function boundingBoxStyle(doc: DetectedDocument) {
    const xs = doc.points.map((p) => p.x);
    const ys = doc.points.map((p) => p.y);
    const minX = Math.min(...xs) * scale;
    const minY = Math.min(...ys) * scale;
    const maxX = Math.max(...xs) * scale;
    const maxY = Math.max(...ys) * scale;
    return { left: minX, top: minY, width: maxX - minX, height: maxY - minY };
  }

  function onSelectOne(doc: DetectedDocument) {
    if (doc.isQuadrilateral) {
      navigation.replace('Crop', { rawPath, corners: doc.points, imageWidth, imageHeight });
    } else {
      navigation.replace('IrregularCrop', { rawPath, points: doc.points });
    }
  }

  async function onAddAll() {
    setProcessingAll(true);
    try {
      const cacheDir = await DocScannerNative.getCacheDirectory();
      for (const doc of documents) {
        const id = `page_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        const outPath = `${cacheDir}/${id}.jpg`;

        if (doc.isQuadrilateral) {
          await DocScannerNative.perspectiveCorrect(rawPath, doc.points, outPath);
        } else {
          await DocScannerNative.cropToContour(rawPath, doc.points, outPath);
        }

        addPage({
          id,
          croppedImagePath: outPath,
          processedImagePath: outPath,
          rotationDegrees: 0,
          filterType: 'AUTO_ENHANCE',
          brightness: 0,
          contrast: 0,
        });
      }
      navigation.navigate('Home');
    } finally {
      setProcessingAll(false);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{documents.length} pages detected — tap one to edit it, or add them all</Text>

      <View style={{ width: screenW, height: displayHeight }}>
        <Image
          source={{ uri: `file://${rawPath}` }}
          style={{ width: screenW, height: displayHeight }}
          resizeMode="contain"
        />
        {documents.map((doc, i) => (
          <TouchableOpacity
            key={i}
            style={[styles.box, boundingBoxStyle(doc)]}
            onPress={() => onSelectOne(doc)}
          >
            <Text style={styles.boxLabel}>{i + 1}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <TouchableOpacity style={styles.addAllButton} onPress={onAddAll} disabled={processingAll}>
        {processingAll ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Add all {documents.length} as separate pages</Text>
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  title: { color: '#fff', textAlign: 'center', padding: 12 },
  box: {
    position: 'absolute', borderWidth: 3, borderColor: '#4CAF50',
    alignItems: 'flex-start', justifyContent: 'flex-start',
  },
  boxLabel: {
    backgroundColor: '#4CAF50', color: '#fff', fontWeight: '700',
    paddingHorizontal: 6, paddingVertical: 2,
  },
  addAllButton: { backgroundColor: '#1976D2', margin: 16, padding: 14, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
