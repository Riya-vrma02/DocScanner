import React, { useCallback, useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, FlatList, Image, Alert, Modal, TextInput, PermissionsAndroid, Platform,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import Share from 'react-native-share';
import { DocScannerNative, SavedFile } from '../native/DocScannerNative';
import ToolIcon from '../components/ToolIcon';
import { filesDir, formatSize, mimeFor } from '../lib/files';

type Row = SavedFile & { thumb?: string | null };

/**
 * Files: everything the app has made (saved scans, merged PDFs, LaTeX), newest first.
 * Tap to open in a viewer app; ⋮ for share, save to Downloads, rename, delete.
 */
export default function FilesScreen() {
  const [files, setFiles] = useState<Row[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [menu, setMenu] = useState<Row | null>(null);
  const [renaming, setRenaming] = useState<Row | null>(null);
  const [newName, setNewName] = useState('');

  const load = useCallback(async () => {
    const dir = await filesDir();
    const list: Row[] = await DocScannerNative.listFiles(dir);
    setFiles(list);
    setLoaded(true);
    // Page-1 thumbnails for PDFs (cached next to the app's cache files).
    const cache = await DocScannerNative.getCacheDirectory();
    for (const f of list) {
      if (!f.name.toLowerCase().endsWith('.pdf')) continue;
      const thumb = await DocScannerNative.pdfThumbnail(f.path, `${cache}/thumbs/${f.name}_${f.modified}.jpg`).catch(() => null);
      if (thumb) setFiles((cur) => cur.map((x) => (x.path === f.path ? { ...x, thumb } : x)));
    }
  }, []);

  useFocusEffect(useCallback(() => { load().catch(() => setLoaded(true)); }, [load]));

  function open(f: Row) {
    DocScannerNative.openFile(f.path, mimeFor(f.name)).catch((e: any) =>
      Alert.alert('Can\'t open', e?.message ?? String(e)));
  }

  function share(f: Row) {
    Share.open({ url: `file://${f.path}`, type: mimeFor(f.name), filename: f.name }).catch(() => {});
  }

  async function saveToDownloads(f: Row) {
    try {
      if (Platform.OS === 'android' && Number(Platform.Version) < 29) {
        const r = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE);
        if (r !== PermissionsAndroid.RESULTS.GRANTED) return;
      }
      const where = await DocScannerNative.saveFileToDownloads(f.path, f.name, mimeFor(f.name));
      Alert.alert('Saved', `Saved to ${where}`);
    } catch (e: any) {
      Alert.alert('Save failed', e?.message ?? String(e));
    }
  }

  function confirmDelete(f: Row) {
    Alert.alert('Delete file?', f.name, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive',
        onPress: async () => {
          await DocScannerNative.deleteFile(f.path).catch(() => {});
          setFiles((cur) => cur.filter((x) => x.path !== f.path));
        },
      },
    ]);
  }

  async function doRename() {
    const f = renaming;
    if (!f) return;
    const ext = f.name.includes('.') ? f.name.slice(f.name.lastIndexOf('.')) : '';
    let n = newName.trim();
    if (!n) return;
    if (ext && !n.toLowerCase().endsWith(ext.toLowerCase())) n += ext;
    try {
      await DocScannerNative.renameFile(f.path, n);
      setRenaming(null);
      load();
    } catch (e: any) {
      Alert.alert('Couldn\'t rename', e?.message ?? String(e));
    }
  }

  function kindLabel(name: string) {
    const ext = name.toLowerCase().split('.').pop();
    return ext === 'pdf' ? 'PDF' : ext === 'zip' ? 'LaTeX + images' : ext === 'tex' ? 'LaTeX' : (ext ?? '').toUpperCase();
  }

  if (loaded && files.length === 0) {
    return (
      <View style={[styles.container, styles.empty]}>
        <ToolIcon name="folder" size={56} color="#BDC1C6" />
        <Text style={styles.emptyTitle}>No files yet</Text>
        <Text style={styles.emptyText}>PDFs you save or merge and LaTeX you create appear here.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={files}
        keyExtractor={(f) => f.path}
        contentContainerStyle={{ padding: 12 }}
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.row} onPress={() => open(item)} onLongPress={() => setMenu(item)}>
            {item.thumb
              ? <Image source={{ uri: `file://${item.thumb}` }} style={styles.thumb} resizeMode="contain" />
              : <View style={[styles.thumb, styles.thumbBlank]}><ToolIcon name="file" size={26} color="#5F6368" /></View>}
            <View style={styles.info}>
              <Text style={styles.name} numberOfLines={2}>{item.name}</Text>
              <Text style={styles.meta}>
                {kindLabel(item.name)} · {formatSize(item.size)} · {new Date(item.modified).toLocaleString()}
              </Text>
            </View>
            <TouchableOpacity style={styles.moreBtn} onPress={() => setMenu(item)}>
              <ToolIcon name="more" size={22} />
            </TouchableOpacity>
          </TouchableOpacity>
        )}
      />

      <Modal transparent visible={menu !== null} animationType="slide" onRequestClose={() => setMenu(null)}>
        <View style={styles.sheetOverlay}>
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle} numberOfLines={2}>{menu?.name}</Text>
            {[
              { label: 'Open', run: open },
              { label: 'Share', run: share },
              { label: 'Save to Downloads', run: saveToDownloads },
              { label: 'Rename', run: (f: Row) => { setNewName(f.name.replace(/\.[^.]+$/, '')); setRenaming(f); } },
            ].map((o) => (
              <TouchableOpacity key={o.label} style={styles.sheetOption} onPress={() => { const f = menu; setMenu(null); if (f) o.run(f); }}>
                <Text style={styles.sheetText}>{o.label}</Text>
              </TouchableOpacity>
            ))}
            <TouchableOpacity style={styles.sheetOption} onPress={() => { const f = menu; setMenu(null); if (f) confirmDelete(f); }}>
              <Text style={[styles.sheetText, styles.danger]}>Delete</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setMenu(null)}>
              <Text style={styles.cancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal transparent visible={renaming !== null} animationType="fade" onRequestClose={() => setRenaming(null)}>
        <View style={styles.centerOverlay}>
          <View style={styles.dialog}>
            <Text style={styles.sheetTitle}>Rename</Text>
            <TextInput value={newName} onChangeText={setNewName} style={styles.input} autoFocus selectTextOnFocus />
            <View style={styles.dialogBtns}>
              <TouchableOpacity onPress={() => setRenaming(null)}><Text style={styles.dialogBtn}>Cancel</Text></TouchableOpacity>
              <TouchableOpacity onPress={doRename}><Text style={[styles.dialogBtn, styles.dialogOk]}>Rename</Text></TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const BLUE = '#1A73E8';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#3A3C40' },
  empty: { alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyTitle: { color: '#FFFFFF', fontSize: 20, fontWeight: '700', marginTop: 14 },
  emptyText: { color: '#BDC1C6', fontSize: 15, marginTop: 8, textAlign: 'center' },
  row: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#1F2125', borderRadius: 10, padding: 10, marginBottom: 10 },
  thumb: { width: 46, height: 62, backgroundColor: '#FFFFFF', borderRadius: 3, marginRight: 12 },
  thumbBlank: { alignItems: 'center', justifyContent: 'center', backgroundColor: '#E8EAED' },
  info: { flex: 1 },
  name: { color: '#FFFFFF', fontSize: 15, fontWeight: '600' },
  meta: { color: '#BDC1C6', fontSize: 12, marginTop: 4 },
  moreBtn: { padding: 8 },
  sheetOverlay: { flex: 1, backgroundColor: '#00000066', justifyContent: 'flex-end' },
  sheet: { backgroundColor: '#FFFFFF', padding: 20, borderTopLeftRadius: 16, borderTopRightRadius: 16 },
  sheetTitle: { fontSize: 17, fontWeight: '600', color: '#111', marginBottom: 6 },
  sheetOption: { paddingVertical: 15, borderBottomWidth: 1, borderBottomColor: '#EEE' },
  sheetText: { fontSize: 16, color: BLUE, fontWeight: '600' },
  danger: { color: '#D32F2F' },
  cancelText: { textAlign: 'center', color: '#111', marginTop: 14, fontSize: 16 },
  centerOverlay: { flex: 1, backgroundColor: '#00000088', alignItems: 'center', justifyContent: 'center', padding: 24 },
  dialog: { backgroundColor: '#FFFFFF', borderRadius: 12, padding: 20, alignSelf: 'stretch' },
  input: { borderWidth: 1, borderColor: '#DADCE0', borderRadius: 8, paddingHorizontal: 12, height: 46, fontSize: 16, color: '#111', marginTop: 8 },
  dialogBtns: { flexDirection: 'row', justifyContent: 'flex-end', gap: 24, marginTop: 18 },
  dialogBtn: { fontSize: 16, color: '#5F6368', fontWeight: '600' },
  dialogOk: { color: BLUE },
});