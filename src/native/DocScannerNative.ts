import { NativeModules } from 'react-native';
import { FilterType, PageSizeName } from '../types';

const { DocScannerModule, DewarpModule } = NativeModules;

export interface DetectedCorners {
  corners: { x: number; y: number }[];
  imageWidth: number;
  imageHeight: number;
}

export interface DetectedDocument {
  points: { x: number; y: number }[];
  isQuadrilateral: boolean;
  area: number;
}

export interface DetectedDocuments {
  documents: DetectedDocument[];
  imageWidth: number;
  imageHeight: number;
}

export const DocScannerNative = {
  capturePhoto: (): Promise<string> => DocScannerModule.capturePhoto(),

  detectDocumentCorners: (imagePath: string): Promise<DetectedCorners> =>
    DocScannerModule.detectDocumentCorners(imagePath),

  /**
   * Finds EVERY significant document-shaped region in the photo (handles
   * multiple stacked/side-by-side pages), and reports for each one whether
   * it's a clean quadrilateral (perspectiveCorrect it) or an irregular/torn
   * shape (cropToContour it instead — there's no well-defined "4 corners"
   * to perspective-warp for a torn edge).
   */
  detectAllDocuments: (imagePath: string): Promise<DetectedDocuments> =>
    DocScannerModule.detectAllDocuments(imagePath),

  /** Crops to an arbitrary (possibly non-convex, torn-edge) polygon, masking everything outside it to white. */
  cropToContour: (
    imagePath: string,
    points: { x: number; y: number }[],
    outputPath: string
  ): Promise<string> => DocScannerModule.cropToContour(imagePath, points, outputPath),

  perspectiveCorrect: (
    imagePath: string,
    corners: { x: number; y: number }[],
    outputPath: string
  ): Promise<string> => DocScannerModule.perspectiveCorrect(imagePath, corners, outputPath),

  /**
   * ML-based dewarping for curled/bent pages (e.g. a book page that won't
   * lie flat). This is a SEPARATE fix from perspectiveCorrect — perspective
   * correction only handles a flat page shot at an angle; this handles
   * actual curvature in the page surface. Run this first if the page is
   * both curled and at an angle, then perspectiveCorrect the result.
   */
  dewarpDocument: (imagePath: string, outputPath: string): Promise<string> =>
    DewarpModule.dewarpDocument(imagePath, outputPath),

  applyFilter: (
    imagePath: string,
    filterType: FilterType,
    rotationDegrees: number,
    brightness: number,
    contrast: number,
    outputPath: string
  ): Promise<string> =>
    DocScannerModule.applyFilter(imagePath, filterType, rotationDegrees, brightness, contrast, outputPath),

  mkdir: (path: string): Promise<boolean> => DocScannerModule.mkdir(path),
  readFileBase64: (path: string): Promise<string> => DocScannerModule.readFileBase64(path),
  deleteFile: (path: string): Promise<boolean> => DocScannerModule.deleteFile(path),
  getDocumentDirectory: (): Promise<string> => DocScannerModule.getDocumentDirectory(),
  getCacheDirectory: (): Promise<string> => DocScannerModule.getCacheDirectory(),

  exportToPdf: (
    imagePaths: string[],
    pageSize: PageSizeName,
    outputPath: string
  ): Promise<string> => DocScannerModule.exportToPdf(imagePaths, pageSize, outputPath),

  /**
   * Copies a generated PDF into the device's public Downloads folder.
   * Resolves with a user-facing location (e.g. "Downloads/Scan_123.pdf").
   */
  savePdfToDownloads: (sourcePath: string, displayName: string): Promise<string> =>
    DocScannerModule.savePdfToDownloads(sourcePath, displayName),
};
