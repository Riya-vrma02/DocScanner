import { DocScannerNative } from '../native/DocScannerNative';
import { PageSizeName, ScanPage } from '../types';
import { filesDir } from './files';

/**
 * Replaces pdf-lib + react-native-fs: the native exportToPdf() method uses
 * Android's own android.graphics.pdf.PdfDocument to combine every page into
 * one multi-page PDF sized to the chosen paper format.
 */
export async function exportPagesToPdf(
  pages: ScanPage[],
  pageSizeName: PageSizeName,
  outputFileName: string
): Promise<string> {
  // Saved in the app's Files folder so it shows on the Files screen.
  const outputPath = `${await filesDir()}/${outputFileName}`;
  const imagePaths = pages.map((p) => p.processedImagePath);
  // ocr = true: pages are run through on-device text recognition so the PDF is searchable.
  return DocScannerNative.exportToPdf(imagePaths, pageSizeName, outputPath, true);
}