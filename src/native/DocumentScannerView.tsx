import { requireNativeComponent, ViewStyle } from 'react-native';

export interface CapturedDocument {
  path: string;
  imageWidth: number;
  imageHeight: number;
  corners: { x: number; y: number }[];
  /** Which detector produced the corners: 'ml' | 'cv' | 'fullframe'. */
  detector?: string;
  /** ONNX segmenter state: 'loaded' | 'not_loaded' | 'unavailable'. */
  segStatus?: string;
  /** Why the ML segmenter returned nothing, if it didn't. */
  segError?: string | null;
}

export interface DocumentScannerProps {
  style?: ViewStyle;
  /** Increment this nonce to trigger a full-resolution capture. */
  captureTrigger?: number;
  onDocumentCaptured?: (e: { nativeEvent: CapturedDocument }) => void;
  onScannerError?: (e: { nativeEvent: { message: string } }) => void;
}

/**
 * Native CameraX-backed live document scanner view (Android).
 * Backed by DocumentScannerViewManager ("DocumentScannerView").
 */
const NativeDocumentScannerView =
  requireNativeComponent<DocumentScannerProps>('DocumentScannerView');

export default NativeDocumentScannerView;
