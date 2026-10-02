import React, { useEffect, useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, PermissionsAndroid, Platform, ActivityIndicator, Alert, Linking,
} from 'react-native';
import NativeDocumentScannerView, { CapturedDocument } from '../native/DocumentScannerView';

/**
 * Live in-app scanner: shows the CameraX preview with a real-time edge overlay,
 * and on shutter tap captures a full-res still. The native side detects the
 * page and returns the image path + corners, which we hand to the existing
 * CropScreen for manual corner adjustment (unchanged flow from there on).
 */
export default function CameraScanScreen({ navigation }: any) {
  const [granted, setGranted] = useState<boolean | null>(null);
  // True when Android reports NEVER_ASK_AGAIN — the system dialog won't show
  // again, so the only way back is the app's Settings page.
  const [blocked, setBlocked] = useState(false);
  const [captureNonce, setCaptureNonce] = useState(0);
  const [busy, setBusy] = useState(false);

  async function requestPermission() {
    if (Platform.OS !== 'android') {
      setGranted(true);
      return;
    }
    try {
      const res = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA, {
        title: 'Camera permission',
        message: 'DocScanner needs the camera to scan documents.',
        buttonPositive: 'OK',
        buttonNegative: 'Cancel',
      });
      setGranted(res === PermissionsAndroid.RESULTS.GRANTED);
      setBlocked(res === PermissionsAndroid.RESULTS.NEVER_ASK_AGAIN);
    } catch {
      setGranted(false);
    }
  }

  useEffect(() => {
    requestPermission();
  }, []);

  function onCaptured(e: { nativeEvent: CapturedDocument }) {
    setBusy(false);
    const doc = e.nativeEvent;

    const go = () =>
      navigation.replace('Crop', {
        rawPath: doc.path,
        corners: doc.corners,
        imageWidth: doc.imageWidth,
        imageHeight: doc.imageHeight,
      });

    // ---- TEMP DIAGNOSTIC: is the ML segmenter actually being used? ----
    if (doc.detector !== 'ml') {
      Alert.alert(
        'Detector diagnostic',
        `detector: ${doc.detector}\nmodel: ${doc.segStatus}\nreason: ${doc.segError ?? 'none'}`,
        [{ text: 'Continue', onPress: go }]
      );
      return;
    }
    // ---- END TEMP DIAGNOSTIC ----
    go();
  }

  function onError(e: { nativeEvent: { message: string } }) {
    setBusy(false);
    Alert.alert('Scanner error', e.nativeEvent?.message ?? 'Unknown error');
  }

  function shoot() {
    if (busy) return;
    setBusy(true);
    setCaptureNonce((n) => n + 1);
  }

  if (granted === null) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#fff" size="large" />
      </View>
    );
  }

  if (!granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.msg}>
          {blocked
            ? 'Camera access is blocked. Enable it in Settings to scan documents.'
            : 'Camera permission is required to scan documents.'}
        </Text>
        <TouchableOpacity
          style={styles.button}
          onPress={() => (blocked ? Linking.openSettings() : requestPermission())}
        >
          <Text style={styles.buttonText}>{blocked ? 'Open Settings' : 'Grant camera access'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.button, styles.secondary]} onPress={() => navigation.goBack()}>
          <Text style={styles.buttonText}>Go back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <NativeDocumentScannerView
        style={StyleSheet.absoluteFill as any}
        captureTrigger={captureNonce}
        onDocumentCaptured={onCaptured}
        onScannerError={onError}
      />
      <View style={styles.hint} pointerEvents="none">
        <Text style={styles.hintText}>Align the document within the frame</Text>
      </View>
      <View style={styles.controls}>
        <TouchableOpacity style={styles.shutter} onPress={shoot} disabled={busy}>
          {busy ? <ActivityIndicator color="#000" /> : <View style={styles.shutterInner} />}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000', padding: 24 },
  msg: { color: '#fff', textAlign: 'center', marginBottom: 16 },
  hint: { position: 'absolute', top: 24, left: 0, right: 0, alignItems: 'center' },
  hintText: { color: '#fff', backgroundColor: '#00000066', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 12 },
  controls: { position: 'absolute', bottom: 32, left: 0, right: 0, alignItems: 'center' },
  shutter: {
    width: 72, height: 72, borderRadius: 36, backgroundColor: '#ffffffcc',
    alignItems: 'center', justifyContent: 'center', borderWidth: 4, borderColor: '#fff',
  },
  shutterInner: { width: 56, height: 56, borderRadius: 28, backgroundColor: '#fff' },
  button: { backgroundColor: '#1976D2', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 8, marginTop: 8 },
  secondary: { backgroundColor: '#757575' },
  buttonText: { color: '#fff', fontWeight: '600' },
});
