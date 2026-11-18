import React, { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator, Alert, Image,
  PermissionsAndroid, Platform,
} from 'react-native';
import Share from 'react-native-share';
import { DocScannerNative, LatexResult } from '../native/DocScannerNative';
import { usePagesStore } from '../store/pagesStore';
import { filesDir, stamp } from '../lib/files';

/**
 * Image to LaTeX: pick a photo/screenshot of a page (or use the pages scanned in the app) and get a
 * LaTeX document of it. Text, headings and lists become LaTeX; pictures, charts and equations are kept
 * as images. Everything runs on the phone (OpenCV only).
 */
export default function LatexScreen() {
  const pages = usePagesStore((s) => s.pages);
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<string | null>(null);   // preview of the picked image
  const [result, setResult] = useState<(LatexResult & { baseName: string }) | null>(null);

  async function outDir() {
    const docDir = await DocScannerNative.getDocumentDirectory();
    const dir = `${docDir}/latex`;
    await DocScannerNative.mkdir(dir);
    return dir;
  }

  async function convert(paths: string[], preview: string | null) {
    setBusy(true);
    setResult(null);
    setSource(preview);
    try {
      const baseName = `LaTeX_${stamp()}`;
      // Saved in the Files folder so it shows on the Files screen.
      const res = await DocScannerNative.imageToLatex(paths, await filesDir(), baseName);
      setResult({ ...res, baseName });
    } catch (e: any) {
      Alert.alert('Couldn\'t convert', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function onPickImage() {
    if (busy) return;
    try {
      const img = await DocScannerNative.pickImage(await outDir());
      await convert([img.path], img.path);
    } catch (e: any) {
      if (e?.code === 'IMAGE_CANCELLED') return;
      Alert.alert('Couldn\'t open image', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    }
  }

  function onUseScans() {
    if (busy || pages.length === 0) return;
    convert(pages.map((p) => p.processedImagePath), pages[0].processedImagePath);
  }

  /** The file to hand out: the .zip when there are figures (the .tex alone would miss them). */
  function outputFile() {
    if (!result) return null;
    return result.zipPath
      ? { path: result.zipPath, name: `${result.baseName}.zip`, mime: 'application/zip' }
      : { path: result.texPath, name: `${result.baseName}.tex`, mime: 'application/x-tex' };
  }

  async function onCopy() {
    if (!result) return;
    await DocScannerNative.copyToClipboard(result.tex);
    Alert.alert('Copied', result.figures
      ? 'LaTeX copied. The pictures it refers to are in the .zip (use Share or Save).'
      : 'LaTeX copied to the clipboard.');
  }

  function onShare() {
    const f = outputFile();
    if (f) Share.open({ url: `file://${f.path}`, type: f.mime, filename: f.name }).catch(() => {});
  }

  async function onSave() {
    const f = outputFile();
    if (!f) return;
    try {
      if (Platform.OS === 'android' && Number(Platform.Version) < 29) {
        const res = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE);
        if (res !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert('Permission needed', 'Storage permission is required to save to Downloads.');
          return;
        }
      }
      const where = await DocScannerNative.saveFileToDownloads(f.path, f.name, f.mime);
      Alert.alert('Saved', `Saved to ${where}`);
    } catch (e: any) {
      Alert.alert('Save failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    }
  }

  return (
    <View style={styles.container}>
      <View style={styles.sourceRow}>
        <TouchableOpacity style={[styles.sourceBtn, busy && styles.dim]} onPress={onPickImage} disabled={busy}>
          <Text style={styles.sourceTitle}>Pick an image</Text>
          <Text style={styles.sourceSub}>Photo, screenshot or saved file</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.sourceBtn, (busy || pages.length === 0) && styles.dim]}
          onPress={onUseScans}
          disabled={busy || pages.length === 0}
        >
          <Text style={styles.sourceTitle}>Use scanned pages</Text>
          <Text style={styles.sourceSub}>
            {pages.length ? `${pages.length} page${pages.length === 1 ? '' : 's'} in the app` : 'Scan pages first'}
          </Text>
        </TouchableOpacity>
      </View>

      {busy && (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={BLUE} />
          <Text style={styles.busyText}>Reading the page…</Text>
        </View>
      )}

      {!busy && !result && (
        <View style={styles.center}>
          {source ? <Image source={{ uri: `file://${source}` }} style={styles.preview} resizeMode="contain" /> : null}
          <Text style={styles.help}>
            Works best on a straight, well-lit photo of printed text. Headings, paragraphs and lists become
            LaTeX; pictures, charts and equations are kept as images.
          </Text>
        </View>
      )}

      {!busy && result && (
        <>
          <Text style={styles.summary}>
            {result.pages} page{result.pages === 1 ? '' : 's'} read
            {result.figures ? ` · ${result.figures} picture${result.figures === 1 ? '' : 's'} kept as images (in the .zip)` : ''}
          </Text>
          <ScrollView style={styles.codeBox} contentContainerStyle={styles.codeContent}>
            <ScrollView horizontal>
              <Text selectable style={styles.code}>{result.tex}</Text>
            </ScrollView>
          </ScrollView>
          <View style={styles.bottomBar}>
            <TouchableOpacity style={styles.actionBtn} onPress={onCopy}>
              <Text style={styles.actionText}>Copy</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionBtn} onPress={onShare}>
              <Text style={styles.actionText}>Share {result.zipPath ? '.zip' : '.tex'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, styles.primary]} onPress={onSave}>
              <Text style={styles.primaryText}>Save</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </View>
  );
}

const BLUE = '#1A73E8';

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#3A3C40' },
  sourceRow: { flexDirection: 'row', gap: 12, padding: 16 },
  sourceBtn: {
    flex: 1, backgroundColor: '#1F2125', borderRadius: 12, paddingVertical: 16, paddingHorizontal: 12,
    borderWidth: 1, borderColor: '#5F6368',
  },
  sourceTitle: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
  sourceSub: { color: '#BDC1C6', fontSize: 13, marginTop: 4 },
  dim: { opacity: 0.4 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  busyText: { color: '#FFFFFF', fontSize: 16, marginTop: 14 },
  preview: { width: '70%', height: '55%', marginBottom: 16 },
  help: { color: '#BDC1C6', fontSize: 14, textAlign: 'center', lineHeight: 20 },
  summary: { color: '#BDC1C6', fontSize: 13, paddingHorizontal: 16, paddingBottom: 8 },
  codeBox: { flex: 1, marginHorizontal: 16, backgroundColor: '#17181B', borderRadius: 8 },
  codeContent: { padding: 12 },
  code: { color: '#E8EAED', fontFamily: 'monospace', fontSize: 12, lineHeight: 17 },
  bottomBar: { flexDirection: 'row', gap: 10, padding: 16, backgroundColor: '#1F2125', marginTop: 12 },
  actionBtn: {
    flex: 1, height: 48, borderRadius: 24, borderWidth: 2, borderColor: '#FFFFFF',
    alignItems: 'center', justifyContent: 'center',
  },
  actionText: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' },
  primary: { backgroundColor: BLUE, borderColor: BLUE },
  primaryText: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' },
});