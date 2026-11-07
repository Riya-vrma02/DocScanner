import { DocScannerNative } from '../native/DocScannerNative';
import { ScanPage } from '../types';

export interface ImportResult {
  name: string;
  added: number;
  totalPages: number;
}

/**
 * Lets the user pick a PDF, renders its pages and adds them as pages of the current scan
 * (after any existing pages). Returns null if the user closed the picker without choosing a file.
 */
export async function importPdfPages(addPage: (p: ScanPage) => void): Promise<ImportResult | null> {
  const docDir = await DocScannerNative.getDocumentDirectory();
  const outDir = `${docDir}/imports`;
  await DocScannerNative.mkdir(outDir);
  let pdf;
  try {
    pdf = await DocScannerNative.pickAndImportPdf(outDir);
  } catch (e: any) {
    if (e?.code === 'PDF_CANCELLED') return null;
    throw e;
  }
  const stamp = Date.now();
  pdf.pages.forEach((pg, i) => {
    addPage({
      id: `page_${stamp}_${i}`,
      croppedImagePath: pg.path,
      processedImagePath: pg.path,
      rotationDegrees: 0,
      // Already a clean digital page: no colour enhancement by default (it can still be applied in Filters).
      filterType: 'ORIGINAL',
      brightness: 0,
      contrast: 0,
      // The whole page as the "photo" with its four corners, so Crop works on imported pages too.
      rawPath: pg.path,
      cropCorners: [
        { x: 0, y: 0 },
        { x: pg.width, y: 0 },
        { x: pg.width, y: pg.height },
        { x: 0, y: pg.height },
      ],
      sourceWidth: pg.width,
      sourceHeight: pg.height,
    });
  });
  return { name: pdf.name, added: pdf.pages.length, totalPages: pdf.totalPages };
}