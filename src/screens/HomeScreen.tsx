import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Image, useWindowDimensions, Alert, Modal, ActivityIndicator,
} from 'react-native';
import { usePagesStore } from '../store/pagesStore';
import { useExportFlow } from '../lib/useExportFlow';
import DraggableGrid from '../components/DraggableGrid';
import ToolIcon, { ToolIconName } from '../components/ToolIcon';
import { ScanPage } from '../types';
import { importFiles } from '../lib/importFiles';
/** Image URI of a page; the version makes a re-processed file with the same name reload. */
function pageUri(p: ScanPage) {
  return `file://${p.processedImagePath}?v=${p.version ?? 0}`;
}

/**
 * Home: four options (Scan file, Import files, Merge files, Files) and, below them, every page of the
 * current scan as a grid with its page number. Long-press a page and drag it to move it; tap a page to
 * open it full size. "Save PDF" exports the pages in the order shown.
 */
export default function HomeScreen({ navigation }: any) {
  const pages = usePagesStore((s) => s.pages);
  const reorder = usePagesStore((s) => s.reorder);
  const addPage = usePagesStore((s) => s.addPage);
  const [importing, setImporting] = useState(false);
  const exportFlow = useExportFlow(pages);
  const { width } = useWindowDimensions();
  const numColumns = width >= 600 ? 4 : 3;

  /** Pick PDFs and/or images and add all their pages after the existing ones. */
  async function onImportFiles() {
    if (importing) return;
    setImporting(true);
    try {
      const res = await importFiles(addPage);
      if (!res) return;
      const notes: string[] = [];
      res.truncated.forEach((t) => notes.push(`${t.name}: only the first ${t.added} of ${t.total} pages were added.`));
      res.failed.forEach((f) => notes.push(`${f.name} ${f.reason === 'password-protected' ? 'is password-protected' : 'couldn\'t be read'}, so it was skipped.`));
      if (notes.length) Alert.alert(`${res.added} page${res.added === 1 ? '' : 's'} added`, notes.join('\n\n'));
    } catch (e: any) {
      Alert.alert('Couldn\'t import', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    } finally {
      setImporting(false);
    }
  }

  useEffect(() => {
    navigation.setOptions({
      title: pages.length ? `${pages.length} page${pages.length === 1 ? '' : 's'}` : 'DocScanner',
      headerRight: () => (
        <TouchableOpacity onPress={() => navigation.navigate('Latex')} style={styles.headerBtn}>
          <Text style={styles.headerBtnText}>To LaTeX</Text>
        </TouchableOpacity>
      ),
    });
  }, [pages.length, navigation]);

  const options: { icon: ToolIconName; label: string; onPress: () => void }[] = [
    { icon: 'scan', label: 'Scan file', onPress: () => navigation.navigate('Scan') },
    { icon: 'import', label: 'Import files', onPress: onImportFiles },
    { icon: 'merge', label: 'Merge files', onPress: () => navigation.navigate('Merge') },
    { icon: 'folder', label: 'Files', onPress: () => navigation.navigate('Files') },
  ];
  const empty = pages.length === 0;

  const optionTiles = (
    <View style={[styles.options, empty && styles.optionsBig]}>
      {options.map((o) => (
        <TouchableOpacity
          key={o.label}
          style={[styles.option, empty && styles.optionBig]}
          onPress={o.onPress}
          disabled={importing}
        >
          <View style={[styles.optionIcon, empty && styles.optionIconBig]}>
            <ToolIcon name={o.icon} size={empty ? 34 : 26} />
          </View>
          <Text style={[styles.optionText, empty && styles.optionTextBig]} numberOfLines={1}>{o.label}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const importingOverlay = (
    <Modal transparent visible={importing} animationType="fade">
      <View style={styles.busyOverlay}>
        <View style={styles.busyCard}>
          <ActivityIndicator size="large" color={BLUE} />
          <Text style={styles.busyText}>Importing files…</Text>
        </View>
      </View>
    </Modal>
  );

  if (empty) {
    return (
      <View style={[styles.container, styles.emptyWrap]}>
        <Text style={styles.emptyTitle}>What would you like to do?</Text>
        {optionTiles}
        {importingOverlay}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {optionTiles}
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
  hint: { color: '#BDC1C6', fontSize: 13, textAlign: 'center', paddingTop: 2 },

  // ---- the four options: a compact row above the pages, big 2x2 tiles when there are no pages
  options: { flexDirection: 'row', paddingHorizontal: 8, paddingTop: 12, paddingBottom: 8, backgroundColor: '#1F2125' },
  optionsBig: { alignSelf: 'stretch', flexWrap: 'wrap', backgroundColor: 'transparent', paddingHorizontal: 0, justifyContent: 'center', gap: 14 },
  option: { flex: 1, alignItems: 'center', paddingVertical: 4 },
  optionBig: {
    flex: 0, width: '44%', aspectRatio: 1.05, justifyContent: 'center', borderRadius: 16,
    backgroundColor: '#1F2125', borderWidth: 1, borderColor: '#5F6368',
  },
  optionIcon: { width: 48, height: 48, borderRadius: 24, backgroundColor: BLUE, alignItems: 'center', justifyContent: 'center' },
  optionIconBig: { width: 66, height: 66, borderRadius: 33 },
  optionText: { color: '#FFFFFF', fontSize: 13, fontWeight: '600', marginTop: 6 },
  optionTextBig: { fontSize: 16, marginTop: 12 },

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

  emptyWrap: { alignItems: 'center', justifyContent: 'center', padding: 16 },
  emptyTitle: { color: '#FFFFFF', fontSize: 20, fontWeight: '700', marginBottom: 20 },

  headerBtn: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16, borderWidth: 1, borderColor: '#FFFFFF66' },
  headerBtnText: { color: '#FFFFFF', fontSize: 14, fontWeight: '600' },
  busyOverlay: { flex: 1, backgroundColor: '#00000088', alignItems: 'center', justifyContent: 'center' },
  busyCard: { backgroundColor: '#FFFFFF', borderRadius: 12, padding: 24, alignItems: 'center', minWidth: 200 },
  busyText: { marginTop: 12, fontSize: 16, color: '#111' },
});