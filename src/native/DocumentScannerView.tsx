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
  /**
   * True when the detected blob was multiple stacked/overlapping sheets that
   * had to be cut apart — only the TOP sheet is in this capture.
   */
  overlapping?: boolean;
  /** Count of clearly detached paper regions (pages side by side). */
  separateSheets?: number;
  /** Hull vs min-area rect (diagnostic only). */
  blobRectangularity?: number;
  /** Paper area / hull area — the value that decides `overlapping`. */
  blobSolidity?: number;
  /** True when a cut was actually applied to isolate the top sheet. */
  splitApplied?: boolean;
}

/** Live sheet state from the preview, emitted only when it changes. */
export interface DetectionState {
  /** A page is currently detected in frame. */
  detected: boolean;
  /** Sheets are stacked/overlapping; only the top one would be scanned. */
  overlapping: boolean;
  /** Clearly detached paper regions in frame (pages side by side). */
  separateSheets: number;
  /**
   * Hull vs min-area rect. Reads ~1.0 for a pile too, so it is diagnostic only
   * and does NOT drive the overlap warning.
   */
  blobRectangularity: number;
  /**
   * Paper area / convex-hull area. A single sheet fills its own hull (~0.95+);
   * angled sheets leave background wedges inside the hull and read lower. This
   * is the value the overlap warning is based on.
   */
  blobSolidity: number;
  /** True when a cut was actually applied to isolate the top sheet. */
  splitApplied: boolean;
}

export interface DocumentScannerProps {
  style?: ViewStyle;
  /** Increment this nonce to trigger a full-resolution capture. */
  captureTrigger?: number;
  onDocumentCaptured?: (e: { nativeEvent: CapturedDocument }) => void;
  onScannerError?: (e: { nativeEvent: { message: string } }) => void;
  /** Fires when the live sheet state changes (detected / overlapping / count). */
  onDetectionState?: (e: { nativeEvent: DetectionState }) => void;
}

/**
 * Native CameraX-backed live document scanner view (Android).
 * Backed by DocumentScannerViewManager ("DocumentScannerView").
 */
const NativeDocumentScannerView =
  requireNativeComponent<DocumentScannerProps>('DocumentScannerView');

export default NativeDocumentScannerView;
