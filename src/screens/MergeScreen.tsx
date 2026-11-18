import React, { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, FlatList, Image, Alert, Modal, ActivityIndicator, TextInput,
  PermissionsAndroid, Platform,
} from 'react-native';
import Share from 'react-native-share';
import { DocScannerNative, MergeInput } from '../native/DocScannerNative';
import ToolIcon from '../components/ToolIcon';
import { filesDir, formatSize, stamp } from '../lib/files';

/**
 * Merge files: pick several PDFs, put them in order, join them into one PDF.
 * Pages are copied as they are (not re-rendered), so text stays sharp and selectable.
 * The result is saved in Files and can be opened, shared or saved to Downloads.
 */
export default function MergeScreen() {
  const [items, setItems] = useState<MergeInput[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [name, setName] = useState(`Merged_${stamp()}`);
  const [done, setDone] = useState<{ path: string; fileName: string; pages: number; size: number } | null>(null);

  async function addPdfs() {
    if (busy) return;
    try {
      const docDir = await DocScannerNative.getDocumentDirectory();
      const dir = `${docDir}/merge_inputs`;
      await DocScannerNative.mkdir(dir);
      setBusy('Opening PDFs…');
      const picked = await DocScannerNative.pickPdfsForMerge(dir);
      setItems((cur) => [...cur, ...picked]);
      const locked = picked.filter((p) => p.protected);
      if (locked.length) {
        Alert.alert('Password-protected', `${locked.map((p) => p.name).join(', ')} can't be merged. Remove ${locked.length === 1 ? 'it' : 'them'} to continue.`);
      }
    } catch (e: any) {
      if (e?.code !== 'PDF_CANCELLED') Alert.alert('Couldn\'t add PDFs', e?.message ?? String(e));
    } finally {
      setBusy(null);
    }
  }

  function move(i: number, d: -1 | 1) {
    const j = i + d;
    if (j < 0 || j >= items.length) return;
    const next = [...items];
    [next[i], next[j]] = [next[j], next[i]];
    setItems(next);
  }

  function remove(i: number) {
    const it = items[i];
    setItems(items.filter((_, k) => k !== i));
    DocScannerNative.deleteFile(it.path).catch(() => {});
  }

  async function merge() {
    if (items.length < 2) return;
    if (items.some((p) => p.protected)) {
      Alert.alert('Password-protected file', 'Remove the files marked "Password-protected" first.');
      return;
    }
    const base = (name.trim() || `Merged_${stamp()}`).replace(/[\\/:*?"<>|]/g, '_').replace(/\.pdf$/i, '');
    const fileName = `${base}.pdf`;
    setBusy('Merging…');
    try {
      const outPath = `${await filesDir()}/${fileName}`;
      const res = await DocScannerNative.mergePdfs(items.map((p) => p.path), items.map((p) => p.name), outPath);
      setDone({ path: res.path, fileName, pages: res.pages, size: res.size });
    } catch (e: any) {
      Alert.alert('Merge failed', e?.message ?? String(e));
    } finally {
      setBusy(null);
    }
  }

  async function saveToDownloads(path: string, fileName: string) {
    try {
      if (Platform.OS === 'android' && Number(Platform.Version) < 29) {
        const r = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE);
        if (r !== PermissionsAndroid.RESULTS.GRANTED) return;
      }
      const where = await DocScannerNative.savePdfToDownloads(path, fileName);
      Alert.alert('Saved', `Saved to ${where}`);
    } catch (e: any) {
      Alert.alert('Save failed', e?.message ?? String(e));
    }
  }

  function startOver() {
    items.forEach((p) => DocScannerNative.deleteFile(p.path).catch(() => {}));
    setItems([]);
    setDone(null);
    setName(`Merged_${stamp()}`);
  }

  const totalPages = items.reduce((n, p) => n + p.pages, 0);

  return (
    <View style={styles.container}>
      {items.length === 0 ? (
        <View style={styles.empty}>
          <ToolIcon name="merge" size={56} color="#BDC1C6" />
          <Text style={styles.emptyTitle}>Merge PDF files</Text>
          <Text style={styles.emptyText}>Pick two or more PDFs. You can put them in any order before merging.</Text>
          <TouchableOpacity style={styles.primaryWide} onPress={addPdfs}>
            <Text style={styles.primaryText}>Choose PDFs</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <Text style={styles.hint}>{items.length} files · {totalPages} pages · merged top to bottom</Text>
          <FlatList
            data={items}
            keyExtractor={(p) => p.path}
            contentContainerStyle={{ padding: 12, paddingBottom: 24 }}
            renderItem={({ item, index }) => (
              <View style={[styles.row, item.protected && styles.rowBad]}>
                <Text style={styles.order}>{index + 1}</Text>
                {item.thumb
                  ? <Image source={{ uri: `file://${item.thumb}` }} style={styles.thumb} resizeMode="contain" />
                  : <View style={[styles.thumb, styles.thumbBlank]}><ToolIcon name="pdf" size={26} color="#5F6368" /></View>}
                <View style={styles.info}>
                  <Text style={styles.name} numberOfLines={2}>{item.name}</Text>
                  <Text style={item.protected ? styles.bad : styles.meta}>
                    {item.protected ? 'Password-protected: can\'t be merged' : `${item.pages} page${item.pages === 1 ? '' : 's'} · ${formatSize(item.size)}`}
                  </Text>
                </View>
                <View style={styles.rowBtns}>
                  <TouchableOpacity onPress={() => move(index, -1)} disabled={index === 0} style={[styles.iconBtn, index === 0 && styles.dim]}>
                    <ToolIcon name="up" size={22} />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => move(index, 1)} disabled={index === items.length - 1} style={[styles.iconBtn, index === items.length - 1 && styles.dim]}>
                    <ToolIcon name="down" size={22} />
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => remove(index)} style={styles.iconBtn}>
                    <ToolIcon name="close" size={20} color="#F28B82" />
                  </TouchableOpacity>
                </View>
              </View>
            )}
            ListFooterComponent={
              <View style={styles.nameBox}>
                <Text style={styles.label}>File name</Text>
                <TextInput value={name} onChangeText={setName} style={styles.input} placeholder="Merged" placeholderTextColor="#80868B" />
              </View>
            }
          />
          <View style={styles.bottomBar}>
            <TouchableOpacity style={styles.secondaryBtn} onPress={addPdfs}>
              <Text style={styles.secondaryText}>Add PDFs</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.primaryBtn, items.length < 2 && styles.dim]} onPress={merge} disabled={items.length < 2}>
              <Text style={styles.primaryText}>{items.length < 2 ? 'Add 1 more' : `Merge ${items.length} files`}</Text>
            </TouchableOpacity>
          </View>
        </>
      )}

      <Modal transparent visible={busy !== null} animationType="fade">
        <View style={styles.overlay}>
          <View style={styles.busyCard}>
            <ActivityIndicator size="large" color={BLUE} />
            <Text style={styles.busyText}>{busy}</Text>
          </View>
        </View>
      </Modal>

      <Modal transparent visible={done !== null} animationType="slide">
        <View style={styles.sheetOverlay}>
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle}>Merged PDF ready</Text>
            <Text style={styles.sheetSub}>
              {done?.fileName} · {done?.pages} pages · {done ? formatSize(done.size) : ''}{'\n'}Also saved in Files.
            </Text>
            <TouchableOpacity style={styles.sheetOption} onPress={() => done && DocScannerNative.openFile(done.path, 'application/pdf').catch((e: any) => Alert.alert('Can\'t open', e?.message ?? String(e)))}>
              <Text style={styles.sheetText}>Open</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.sheetOption} onPress={() => done && Share.open({ url: `file://${done.path}`, type: 'application/pdf' }).catch(() => {})}>
              <Text style={styles.sheetText}>Share</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.sheetOption} onPress={() => done && saveToDownloads(done.path, done.fileName)}>
              <Text style={styles.sheetText}>Save to Downloads</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={startOver}>
              <Text style={styles.doneText}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const BLUE = '#1A73E8';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#3A3C40' },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyTitle: { color: '#FFFFFF', fontSize: 20, fontWeight: '700', marginTop: 14 },
  emptyText: { color: '#BDC1C6', fontSize: 15, marginTop: 8, textAlign: 'center' },
  primaryWide: { alignSelf: 'stretch', height: 52, borderRadius: 26, backgroundColor: BLUE, alignItems: 'center', justifyContent: 'center', marginTop: 24 },
  hint: { color: '#BDC1C6', fontSize: 13, textAlign: 'center', paddingTop: 10 },
  row: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#1F2125', borderRadius: 10,
    padding: 10, marginBottom: 10, borderWidth: 1, borderColor: 'transparent',
  },
  rowBad: { borderColor: '#F28B82' },
  order: { color: '#BDC1C6', fontSize: 15, fontWeight: '700', width: 22, textAlign: 'center' },
  thumb: { width: 46, height: 62, backgroundColor: '#FFFFFF', borderRadius: 3, marginHorizontal: 8 },
  thumbBlank: { alignItems: 'center', justifyContent: 'center', backgroundColor: '#E8EAED' },
  info: { flex: 1 },
  name: { color: '#FFFFFF', fontSize: 15, fontWeight: '600' },
  meta: { color: '#BDC1C6', fontSize: 13, marginTop: 3 },
  bad: { color: '#F28B82', fontSize: 13, marginTop: 3 },
  rowBtns: { flexDirection: 'row' },
  iconBtn: { padding: 6 },
  dim: { opacity: 0.3 },
  nameBox: { marginTop: 6 },
  label: { color: '#BDC1C6', fontSize: 13, marginBottom: 6 },
  input: { backgroundColor: '#1F2125', color: '#FFFFFF', borderRadius: 8, paddingHorizontal: 12, height: 46, fontSize: 15 },
  bottomBar: { flexDirection: 'row', gap: 12, padding: 16, backgroundColor: '#1F2125' },
  secondaryBtn: { flex: 1, height: 52, borderRadius: 26, borderWidth: 2, borderColor: '#FFFFFF', alignItems: 'center', justifyContent: 'center' },
  secondaryText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
  primaryBtn: { flex: 1.3, height: 52, borderRadius: 26, backgroundColor: BLUE, alignItems: 'center', justifyContent: 'center' },
  primaryText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
  overlay: { flex: 1, backgroundColor: '#00000088', alignItems: 'center', justifyContent: 'center' },
  busyCard: { backgroundColor: '#FFFFFF', borderRadius: 12, padding: 24, alignItems: 'center', minWidth: 200 },
  busyText: { marginTop: 12, fontSize: 16, color: '#111' },
  sheetOverlay: { flex: 1, backgroundColor: '#00000066', justifyContent: 'flex-end' },
  sheet: { backgroundColor: '#FFFFFF', padding: 20, borderTopLeftRadius: 16, borderTopRightRadius: 16 },
  sheetTitle: { fontSize: 18, fontWeight: '600', color: '#111' },
  sheetSub: { fontSize: 13, color: '#666', marginTop: 4, marginBottom: 8 },
  sheetOption: { paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#EEE' },
  sheetText: { fontSize: 16, color: BLUE, fontWeight: '600' },
  doneText: { textAlign: 'center', color: '#111', marginTop: 14, fontSize: 16, fontWeight: '600' },
});