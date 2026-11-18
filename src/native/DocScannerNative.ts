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

export interface ImportedPdf {
  /** File name of the chosen PDF, e.g. "notes.pdf". */
  name: string;
  /** Page count of the PDF (pages beyond the import limit are not rendered). */
  totalPages: number;
  pages: { path: string; width: number; height: number }[];
}

export interface PickedImage {
  path: string;
  width: number;
  height: number;
}

export interface LatexResult {
  /** The LaTeX source. */
  tex: string;
  /** The .tex file written to the output folder. */
  texPath: string;
  /** .zip with main.tex + figures/ (only when parts of the pages were kept as pictures), else null. */
  zipPath: string | null;
  /** Number of pictures cut out of the pages (equations, charts, photos, unreadable bits). */
  figures: number;
  pages: number;
}

export interface ImportedFile {
  name: string;
  kind: 'pdf' | 'image';
  /** Pages in the file (a PDF longer than the import limit has more than `pages`). */
  totalPages: number;
  pages: { path: string; width: number; height: number }[];
  /** Why the file gave no pages, or null. */
  error: null | 'PDF_PROTECTED' | 'UNREADABLE';
}

export interface MergeInput {
  /** App-private copy of the picked PDF. */
  path: string;
  name: string;
  pages: number;
  size: number;
  /** Small JPEG of page 1, or null. */
  thumb: string | null;
  protected: boolean;
}

export interface SavedFile {
  path: string;
  name: string;
  size: number;
  /** ms since epoch */
  modified: number;
}

export const DocScannerNative = {
  capturePhoto: (): Promise<string> => DocScannerModule.capturePhoto(),

  /**
   * Opens the system file picker for a PDF and renders each page to a JPEG in outputDir.
   * Rejects with code PDF_CANCELLED if the user closes the picker, PDF_PROTECTED for password-protected files.
   */
  pickAndImportPdf: (outputDir: string): Promise<ImportedPdf> =>
    DocScannerModule.pickAndImportPdf(outputDir),

  /** System picker for one image; saves an upright JPEG copy in outputDir. Rejects IMAGE_CANCELLED if closed. */
  pickImage: (outputDir: string): Promise<PickedImage> => DocScannerModule.pickImage(outputDir),

  /** Reads the page images (on-device, OpenCV only) and writes <baseName>.tex (+ .zip with figures) to outputDir. */
  imageToLatex: (imagePaths: string[], outputDir: string, baseName: string): Promise<LatexResult> =>
    DocScannerModule.imageToLatex(imagePaths, outputDir, baseName),

  copyToClipboard: (text: string): Promise<boolean> => DocScannerModule.copyToClipboard(text),

  /** Copies any file into the public Downloads folder. */
  saveFileToDownloads: (sourcePath: string, displayName: string, mimeType: string): Promise<string> =>
    DocScannerModule.saveFileToDownloads(sourcePath, displayName, mimeType),

  /** Pick any number of PDFs and images; each becomes pages. Rejects IMPORT_CANCELLED if closed. */
  pickAndImportFiles: (outputDir: string): Promise<{ files: ImportedFile[] }> =>
    DocScannerModule.pickAndImportFiles(outputDir),

  /** Pick PDFs to merge (copied into outputDir). Rejects PDF_CANCELLED if closed. */
  pickPdfsForMerge: (outputDir: string): Promise<MergeInput[]> => DocScannerModule.pickPdfsForMerge(outputDir),

  /** Joins PDFs in order without re-rendering (text stays selectable). Rejects PDF_PROTECTED / PDF_UNREADABLE. */
  mergePdfs: (paths: string[], names: string[], outputPath: string): Promise<{ path: string; pages: number; size: number }> =>
    DocScannerModule.mergePdfs(paths, names, outputPath),

  listFiles: (dir: string): Promise<SavedFile[]> => DocScannerModule.listFiles(dir),
  renameFile: (path: string, newName: string): Promise<string> => DocScannerModule.renameFile(path, newName),
  /** Opens the file in another app (PDF viewer etc.). Rejects NO_APP if none can open it. */
  openFile: (path: string, mimeType: string): Promise<boolean> => DocScannerModule.openFile(path, mimeType),
  /** Page 1 of a PDF as a small JPEG at outPath (cached). Resolves the path, or null. */
  pdfThumbnail: (path: string, outPath: string): Promise<string | null> => DocScannerModule.pdfThumbnail(path, outPath),

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

  /** ocr: run on-device text recognition and add an invisible text layer, making the PDF searchable. */
  exportToPdf: (
    imagePaths: string[],
    pageSize: PageSizeName,
    outputPath: string,
    ocr: boolean = true
  ): Promise<string> => DocScannerModule.exportToPdf(imagePaths, pageSize, outputPath, ocr),

  /**
   * Copies a generated PDF into the device's public Downloads folder.
   * Resolves with a user-facing location (e.g. "Downloads/Scan_123.pdf").
   */
  savePdfToDownloads: (sourcePath: string, displayName: string): Promise<string> =>
    DocScannerModule.savePdfToDownloads(sourcePath, displayName),
};