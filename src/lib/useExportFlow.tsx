import React, { useState } from 'react';
import { Modal, View, Text, TouchableOpacity, StyleSheet, Alert, PermissionsAndroid, Platform } from 'react-native';
import Share from 'react-native-share';
import { exportPagesToPdf } from './pdfExport';
import { DocScannerNative } from '../native/DocScannerNative';
import { PageSizeName, ScanPage } from '../types';

/**
 * "Save PDF" flow shared by the grid and the page view: page-size picker -> build the PDF ->
 * "Share" / "Save to Downloads" chooser. Render `modals` somewhere in the screen and call `start()`.
 */
export function useExportFlow(pages: ScanPage[]) {
  const [pickerVisible, setPickerVisible] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState<{ path: string; fileName: string } | null>(null);

  /** Public Downloads needs WRITE_EXTERNAL_STORAGE on API <= 28; API 29+ uses MediaStore. */
  async function ensureLegacyStoragePermission(): Promise<boolean> {
    if (Platform.OS !== 'android' || Number(Platform.Version) >= 29) return true;
    const res = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE, {
      title: 'Storage permission',
      message: 'DocScanner needs storage access to save the PDF to Downloads.',
      buttonPositive: 'OK',
      buttonNegative: 'Cancel',
    });
    return res === PermissionsAndroid.RESULTS.GRANTED;
  }

  async function saveToDownloads(path: string, fileName: string) {
    if (!(await ensureLegacyStoragePermission())) {
      Alert.alert('Permission needed', 'Storage permission is required to save to Downloads.');
      return;
    }
    const location = await DocScannerNative.savePdfToDownloads(path, fileName);
    Alert.alert('Saved', `PDF saved to ${location}`);
  }

  async function doExport(pageSize: PageSizeName) {
    setPickerVisible(false);
    setExporting(true);
    try {
      const fileName = `Scan_${Date.now()}.pdf`;
      const path = await exportPagesToPdf(pages, pageSize, fileName);
      // A Modal rather than Alert.alert: Android lays out an alert's buttons in one row and clips
      // them on narrow screens, so "Share" + "Save to Downloads" + "Cancel" didn't fit.
      setExported({ path, fileName });
    } catch (e: any) {
      Alert.alert('Export failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`);
    } finally {
      setExporting(false);
    }
  }

  const modals = (
    <>
      <Modal transparent visible={pickerVisible} animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Choose page size</Text>
            {(['A4', 'LEGAL', 'LETTER'] as PageSizeName[]).map((size) => (
              <TouchableOpacity key={size} style={styles.sizeOption} onPress={() => doExport(size)}>
                <Text style={styles.sizeOptionText}>{size}</Text>
              </TouchableOpacity>
            ))}
            <TouchableOpacity onPress={() => setPickerVisible(false)}>
              <Text style={styles.cancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal transparent visible={exported !== null} animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>PDF ready</Text>
            <Text style={styles.modalSubtitle}>{exported?.fileName ?? ''}</Text>
            <TouchableOpacity
              style={styles.actionOption}
              onPress={() => {
                const f = exported;
                setExported(null);
                if (f) Share.open({ url: `file://${f.path}`, type: 'application/pdf' }).catch(() => {});
              }}
            >
              <Text style={styles.actionText}>Share</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.actionOption}
              onPress={() => {
                const f = exported;
                setExported(null);
                if (f) {
                  saveToDownloads(f.path, f.fileName).catch((e: any) =>
                    Alert.alert('Save failed', `${e?.code ?? 'UNKNOWN'}: ${e?.message ?? String(e)}`)
                  );
                }
              }}
            >
              <Text style={styles.actionText}>Save to Downloads</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setExported(null)}>
              <Text style={styles.cancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );

  return { start: () => setPickerVisible(true), exporting, modals };
}

const styles = StyleSheet.create({
  modalOverlay: { flex: 1, backgroundColor: '#00000066', justifyContent: 'flex-end' },
  modalCard: { backgroundColor: '#fff', padding: 20, borderTopLeftRadius: 16, borderTopRightRadius: 16 },
  modalTitle: { fontSize: 18, fontWeight: '600', marginBottom: 12, color: '#111' },
  modalSubtitle: { fontSize: 13, color: '#777', marginBottom: 8 },
  sizeOption: { paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: '#eee' },
  sizeOptionText: { fontSize: 16, color: '#111' },
  actionOption: { paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#eee' },
  actionText: { fontSize: 16, color: '#1A73E8', fontWeight: '600' },
  cancelText: { textAlign: 'center', color: '#D32F2F', marginTop: 12, fontSize: 16 },
});