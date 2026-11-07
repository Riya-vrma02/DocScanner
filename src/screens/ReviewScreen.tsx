import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, Alert,
  FlatList, Image, ActivityIndicator, useWindowDimensions, NativeSyntheticEvent, NativeScrollEvent,
} from 'react-native';
import DraggableFlatList, { RenderItemParams } from 'react-native-draggable-flatlist';
import { usePagesStore } from '../store/pagesStore';
import { useExportFlow } from '../lib/useExportFlow';
import { DocScannerNative } from '../native/DocScannerNative';
import { ScanPage } from '../types';
import ToolIcon, { ToolIconName } from '../components/ToolIcon';

/** URI for a page image; the version suffix makes a re-processed file with the same name reload. */
export function pageUri(p: ScanPage) {
  return `file://${p.processedImagePath}?v=${p.version ?? 0}`;
}

/**
 * Page view (opened by tapping a page in the grid): one large page at a time (swipe or use the arrows),
 * a numbered thumbnail strip (tap to jump, long-press and drag to reorder), page tools, and
 * "Keep scanning" / "Save PDF".
 */
export default function ReviewScreen({ navigation, route }: any) {
  const pages = usePagesStore((s) => s.pages);
  const removePage = usePagesStore((s) => s.removePage);
  const reorder = usePagesStore((s) => s.reorder);
  const updatePage = usePagesStore((s) => s.updatePage);

  const { width } = useWindowDimensions();
  const [current, setCurrent] = useState<number>(route?.params?.initialIndex ?? 0);
  const [previewH, setPreviewH] = useState(0);
  const [busy, setBusy] = useState(false);
  const listRef = useRef<FlatList<ScanPage>>(null);
  const thumbsRef = useRef<any>(null);
  const prevCount = useRef(pages.length);

  const exportFlow = useExportFlow(pages);

  const page: ScanPage | undefined = pages[current];

  // A newly scanned page is shown straight away; after a delete, stay on a valid page.
  useEffect(() => {
    if (pages.length > prevCount.current) {
      goTo(pages.length - 1, false);
    } else if (current > pages.length - 1) {
      goTo(Math.max(0, pages.length - 1), false);
    }
    prevCount.current = pages.length;
    navigation.setOptions({ title: pages.length ? `${pages.length} page${pages.length === 1 ? '' : 's'}` : 'Pages' });
    if (pages.length === 0) navigation.goBack();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pages.length]);

  function goTo(i: number, animated = true) {
    if (pages.length === 0) return;
    const idx = Math.max(0, Math.min(pages.length - 1, i));
    setCurrent(idx);
    // Wait a frame so a just-added item exists in the list before scrolling to it.
    requestAnimationFrame(() => {
      listRef.current?.scrollToIndex({ index: idx, animated });
      thumbsRef.current?.scrollToIndex?.({ index: idx, animated, viewPosition: 0.5 });
    });
  }

  function onPreviewScrollEnd(e: NativeSyntheticEvent<NativeScrollEvent>) {
    const idx = Math.round(e.nativeEvent.contentOffset.x / width);
    if (idx !== current) {
      setCurrent(idx);
      thumbsRef.current?.scrollToIndex?.({ index: idx, animated: true, viewPosition: 0.5 });
    }
  }

  // ---------------------------------------------------------------- page tools

  function onRetake() {
    if (page) navigation.navigate('Scan', { replacePageId: page.id });
  }

  function onCrop() {
    if (!page) return;
    if (!page.rawPath || !page.cropCorners || !page.sourceWidth || !page.sourceHeight) {
      Alert.alert('Can\'t re-crop this page', 'Its original photo isn\'t available. Use Retake to scan it again.');
      return;
    }
    navigation.navigate('Crop', {
      rawPath: page.rawPath,
      corners: page.cropCorners,
      imageWidth: page.sourceWidth,
      imageHeight: page.sourceHeight,
      replacePageId: page.id,
    });
  }

  async function onRotate() {
    if (!page || busy) return;
    setBusy(true);
    try {
      const next = (page.rotationDegrees + 90) % 360;
      const docDir = await DocScannerNative.getDocumentDirectory();
      await DocScannerNative.mkdir(`${docDir}/scans`);
      const out = `${docDir}/scans/${page.id}.jpg`;
      await DocScannerNative.applyFilter(
        page.croppedImagePath, page.filterType, next, page.brightness, page.contrast, out
      );
      updatePage(page.id, { rotationDegrees: next, processedImagePath: out });
    } catch (e: any) {
      Alert.alert('Rotate failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  function onEdit() {
    if (page) navigation.navigate('Filter', { pageId: page.id });
  }

  function onDelete() {
    if (!page) return;
    Alert.alert('Delete page?', `Page ${current + 1} will be removed from this scan.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => removePage(page.id) },
    ]);
  }

  // ---------------------------------------------------------------- UI

  const tools: { key: string; label: string; icon: ToolIconName; onPress: () => void }[] = [
    { key: 'retake', label: 'Retake', icon: 'retake', onPress: onRetake },
    { key: 'crop', label: 'Crop', icon: 'crop', onPress: onCrop },
    { key: 'rotate', label: 'Rotate', icon: 'rotate', onPress: onRotate },
    { key: 'edit', label: 'Filters', icon: 'edit', onPress: onEdit },
    { key: 'delete', label: 'Delete', icon: 'delete', onPress: onDelete },
  ];

  if (pages.length === 0) return <View style={styles.container} />;

  return (
    <View style={styles.container}>
      {/* Large preview, one page per screen width */}
      <View style={styles.preview} onLayout={(e) => setPreviewH(e.nativeEvent.layout.height)}>
        {previewH > 0 && (
          <FlatList
            ref={listRef}
            data={pages}
            keyExtractor={(p) => p.id}
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            getItemLayout={(_, i) => ({ length: width, offset: width * i, index: i })}
            initialScrollIndex={Math.min(current, pages.length - 1)}
            onMomentumScrollEnd={onPreviewScrollEnd}
            renderItem={({ item }) => (
              <View style={{ width, height: previewH, padding: 20 }}>
                <Image source={{ uri: pageUri(item) }} style={styles.pageImage} resizeMode="contain" />
              </View>
            )}
          />
        )}
        {busy && (
          <View style={styles.busy}>
            <ActivityIndicator size="large" color="#FFFFFF" />
          </View>
        )}
      </View>

      {/* Navigation row */}
      <View style={styles.navRow}>
        <TouchableOpacity
          style={[styles.roundBtn, current === 0 && styles.dim]}
          disabled={current === 0}
          onPress={() => goTo(current - 1)}
        >
          <ToolIcon name="chevronLeft" size={22} />
        </TouchableOpacity>
        <View style={styles.navCenter}>
          <TouchableOpacity style={styles.roundBtn} onPress={() => navigation.navigate('Scan')}>
            <ToolIcon name="addPage" size={22} />
          </TouchableOpacity>
          <View style={styles.pagePill}>
            <Text style={styles.pagePillText}>Page {current + 1} of {pages.length}</Text>
          </View>
        </View>
        <TouchableOpacity
          style={[styles.roundBtn, current === pages.length - 1 && styles.dim]}
          disabled={current === pages.length - 1}
          onPress={() => goTo(current + 1)}
        >
          <ToolIcon name="chevronRight" size={22} />
        </TouchableOpacity>
      </View>

      {/* Thumbnail strip: tap to jump, long-press + drag to reorder */}
      <View style={styles.strip}>
        <DraggableFlatList
          ref={thumbsRef}
          data={pages}
          horizontal
          keyExtractor={(p) => p.id}
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.stripContent}
          onDragEnd={({ data }) => {
            const selectedId = page?.id;
            reorder(data);
            const idx = data.findIndex((p) => p.id === selectedId);
            if (idx >= 0) goTo(idx, false);
          }}
          renderItem={({ item, getIndex, drag, isActive }: RenderItemParams<ScanPage>) => {
            const idx = getIndex() ?? pages.findIndex((p) => p.id === item.id);
            const selected = idx === current;
            return (
              <TouchableOpacity
                onPress={() => goTo(idx)}
                onLongPress={drag}
                delayLongPress={200}
                style={[styles.thumbWrap, selected && styles.thumbSelected, isActive && styles.thumbDragging]}
              >
                <Image source={{ uri: pageUri(item) }} style={styles.thumb} resizeMode="contain" />
                <View style={styles.thumbBadge}>
                  <Text style={styles.thumbBadgeText}>{idx + 1}</Text>
                </View>
              </TouchableOpacity>
            );
          }}
        />
      </View>

      {/* Page tools */}
      <View style={styles.toolsRow}>
        {tools.map((t) => (
          <TouchableOpacity key={t.key} style={styles.tool} onPress={t.onPress} disabled={busy}>
            <ToolIcon name={t.icon} />
            <Text style={styles.toolLabel}>{t.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Main actions */}
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

    </View>
  );
}

const BLUE = '#1A73E8';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#3A3C40' },

  preview: { flex: 1 },
  pageImage: { flex: 1, width: '100%' },
  busy: { ...StyleSheet.absoluteFillObject, backgroundColor: '#00000055', alignItems: 'center', justifyContent: 'center' },

  navRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingVertical: 10,
  },
  navCenter: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  roundBtn: {
    width: 44, height: 44, borderRadius: 22, backgroundColor: '#1F2125',
    alignItems: 'center', justifyContent: 'center',
  },
  pagePill: { backgroundColor: '#1F2125', borderRadius: 22, paddingHorizontal: 22, height: 44, justifyContent: 'center' },
  pagePillText: { color: '#FFFFFF', fontSize: 16, fontWeight: '600' },
  dim: { opacity: 0.35 },

  strip: { backgroundColor: '#2A2C30', height: 124 },
  stripContent: { paddingHorizontal: 8, alignItems: 'center' },
  thumbWrap: {
    width: 76, height: 104, marginHorizontal: 5, borderRadius: 6, borderWidth: 3,
    borderColor: 'transparent', backgroundColor: '#1F2125', padding: 3,
  },
  thumbSelected: { borderColor: BLUE },
  thumbDragging: { opacity: 0.8, transform: [{ scale: 1.06 }] },
  thumb: { flex: 1 },
  thumbBadge: {
    position: 'absolute', bottom: 6, alignSelf: 'center', minWidth: 22, paddingHorizontal: 5,
    borderRadius: 4, backgroundColor: '#000000AA',
  },
  thumbBadgeText: { color: '#FFFFFF', fontSize: 13, fontWeight: '700', textAlign: 'center' },

  toolsRow: {
    flexDirection: 'row', justifyContent: 'space-around', backgroundColor: '#1F2125',
    paddingTop: 14, paddingBottom: 6,
  },
  tool: { alignItems: 'center', minWidth: 60, paddingVertical: 4 },
  toolLabel: { color: '#E8EAED', fontSize: 13, marginTop: 6 },

  bottomBar: { flexDirection: 'row', gap: 12, padding: 16, backgroundColor: '#1F2125' },
  secondaryBtn: {
    flex: 1, height: 52, borderRadius: 26, borderWidth: 2, borderColor: '#FFFFFF',
    alignItems: 'center', justifyContent: 'center',
  },
  secondaryText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700' },
  primaryBtn: { flex: 1, height: 52, borderRadius: 26, backgroundColor: BLUE, alignItems: 'center', justifyContent: 'center' },
  primaryText: { color: '#FFFFFF', fontSize: 17, fontWeight: '700' },

});