import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Image, useWindowDimensions, Alert, Modal, ActivityIndicator,
} from 'react-native';
import { usePagesStore } from '../store/pagesStore';
import { useExportFlow } from '../lib/useExportFlow';
import DraggableGrid from '../components/DraggableGrid';
import { pageUri } from './ReviewScreen';
import { ScanPage } from '../types';
import { importPdfPages } from '../lib/importPdf';

/**
 * Home: every page of the current scan as a grid with its page number as a watermark.
 * Long-press a page and drag it to move it; tap a page to open it full size (retake, crop, rotate,
 * filters, delete). "Keep scanning" adds pages, "Save PDF" exports them in the order shown.
 */
export default function HomeScreen({ navigation }: any) {
  const pages = usePagesStore((s) => s.pages);
  const reorder = usePagesStore((s) => s.reorder);
  const addPage = usePagesStore((s) => s.addPage);
  const [importing, setImporting] = useState(false);
  const exportFlow = useExportFlow(pages);
  const { width } = useWindowDimensions();
  const numColumns = width >= 600 ? 4 : 3;

  /** Pick a PDF and add its pages after the existing ones. */
  async function onImportPdf() {
    if (importing) return;
    setImporting(true);
    try {
      const res = await importPdfPages(addPage);
      if (res && res.added < res.totalPages) {
        Alert.alert('Large PDF', `Imported the first ${res.added} of ${res.totalPages} pages of ${res.name}.`);
      }
    } catch (e: any) {
      const msg = e?.code === 'PDF_PROTECTED'
        ? 'This PDF is password-protected, so its pages can\'t be opened.'
        : `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`;
      Alert.alert('Couldn\'t import PDF', msg);
    } finally {
      setImporting(false);
    }
  }

  useEffect(() => {
    navigation.setOptions({
      title: pages.length ? `${pages.length} page${pages.length === 1 ? '' : 's'}` : 'DocScanner',
      headerRight: () => (
        <TouchableOpacity onPress={onImportPdf} disabled={importing} style={styles.headerBtn}>
          <Text style={styles.headerBtnText}>Import PDF</Text>
        </TouchableOpacity>
      ),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pages.length, navigation, importing]);

  const importingOverlay = (
    <Modal transparent visible={importing} animationType="fade">
      <View style={styles.busyOverlay}>
        <View style={styles.busyCard}>
          <ActivityIndicator size="large" color={BLUE} />
          <Text style={styles.busyText}>Importing PDF…</Text>
        </View>
      </View>
    </Modal>
  );

  if (pages.length === 0) {
    return (
      <View style={[styles.container, styles.emptyWrap]}>
        <Text style={styles.emptyTitle}>No pages yet</Text>
        <Text style={styles.emptyText}>Scan a page to start a new document.</Text>
        <TouchableOpacity style={styles.emptyPrimary} onPress={() => navigation.navigate('Scan')}>
          <Text style={styles.primaryText}>Scan a page</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.emptySecondary} onPress={onImportPdf} disabled={importing}>
          <Text style={styles.secondaryText}>Import a PDF</Text>
        </TouchableOpacity>
        {importingOverlay}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.hint}>Tap a page to open it · long-press and drag to move it</Text>

      <DraggableGrid<ScanPage>
        data={pages}
        keyExtractor={(p) => p.id}
        numColumns={numColumns}
        onReorder={reorder}
        onPress={(_p, index) => navigation.navigate('Review', { initialIndex: index })}
        renderItem={(p, index, dragging) => (
          <View style={[styles.card, dragging && styles.cardDragging]}>
            <Image source={{ uri: pageUri(p) }} style={styles.thumb} resizeMode="contain" />
            {/* Page-number badge, bottom-right corner */}
            <View style={styles.badge} pointerEvents="none">
              <Text style={styles.badgeText}>{index + 1}</Text>
            </View>
          </View>
        )}
      />

      <View style={styles.bottomBar}>
        <TouchableOpacity style={styles.secondaryBtn} onPress={() => navigation.navigate('Scan')}>
          <Text style={styles.secondaryText}>Keep scanning</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.primaryBtn, exportFlow.exporting && styles.dim]}
          disabled={exportFlow.exporting}
          onPress={exportFlow.start}
        >
          <Text style={styles.primaryText}>{exportFlow.exporting ? 'Saving…' : 'Save PDF'}</Text>
        </TouchableOpacity>
      </View>

      {exportFlow.modals}
      {importingOverlay}
    </View>
  );
}

const BLUE = '#1A73E8';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#3A3C40' },
  hint: { color: '#BDC1C6', fontSize: 13, textAlign: 'center', paddingTop: 10 },

  card: {
    flex: 1, backgroundColor: '#FFFFFF', borderRadius: 6, padding: 4,
    borderWidth: 3, borderColor: 'transparent', overflow: 'hidden',
  },
  cardDragging: { borderColor: BLUE },
  thumb: { flex: 1 },
  badge: {
    position: 'absolute', bottom: 8, right: 8, minWidth: 26, height: 26, paddingHorizontal: 6,
    borderRadius: 5, backgroundColor: 'rgba(110,112,116,0.85)', alignItems: 'center', justifyContent: 'center',
  },
  badgeText: { color: '#FFFFFF', fontSize: 15, fontWeight: '500' },

  bottomBar: { flexDirection: 'row', gap: 12, padding: 16, backgroundColor: '#1F2125' },
  secondaryBtn: {
    flex: 1, height: 52, borderRadius: 26, borderWidth: 2, borderColor: '#FFFFFF',
    alignItems: 'center', justifyContent: 'center',
  },
  secondaryText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700' },
  primaryBtn: { flex: 1, height: 52, borderRadius: 26, backgroundColor: BLUE, alignItems: 'center', justifyContent: 'center' },
  primaryText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700' },
  dim: { opacity: 0.35 },

  emptyWrap: { alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyTitle: { color: '#FFFFFF', fontSize: 20, fontWeight: '700' },
  emptyText: { color: '#BDC1C6', fontSize: 15, marginTop: 8, textAlign: 'center' },
  // Own styles (no flex: 1 like the bottom-bar buttons) so they keep their size in the centred column.
  emptyPrimary: {
    alignSelf: 'stretch', height: 52, borderRadius: 26, backgroundColor: BLUE,
    alignItems: 'center', justifyContent: 'center', marginTop: 24,
  },
  emptySecondary: {
    alignSelf: 'stretch', height: 52, borderRadius: 26, borderWidth: 2, borderColor: '#FFFFFF',
    alignItems: 'center', justifyContent: 'center', marginTop: 14,
  },

  headerBtn: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16, borderWidth: 1, borderColor: '#FFFFFF66' },
  headerBtnText: { color: '#FFFFFF', fontSize: 14, fontWeight: '600' },
  busyOverlay: { flex: 1, backgroundColor: '#00000088', alignItems: 'center', justifyContent: 'center' },
  busyCard: { backgroundColor: '#FFFFFF', borderRadius: 12, padding: 24, alignItems: 'center', minWidth: 200 },
  busyText: { marginTop: 12, fontSize: 16, color: '#111' },
});