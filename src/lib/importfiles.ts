import { DocScannerNative } from '../native/DocScannerNative';
import { ScanPage } from '../types';

export interface ImportFilesResult {
  /** Pages added to the scan. */
  added: number;
  /** Files that gave no pages, with the reason. */
  failed: { name: string; reason: string }[];
  /** PDFs longer than the import limit (only their first pages were added). */
  truncated: { name: string; added: number; total: number }[];
}

/**
 * Lets the user pick several PDFs and/or images at once and adds all their pages to the current scan,
 * in the order picked. Returns null if the picker was closed without choosing anything.
 */
export async function importFiles(addPage: (p: ScanPage) => void): Promise<ImportFilesResult | null> {
  const docDir = await DocScannerNative.getDocumentDirectory();
  const outDir = `${docDir}/imports`;
  await DocScannerNative.mkdir(outDir);
  let res;
  try {
    res = await DocScannerNative.pickAndImportFiles(outDir);
  } catch (e: any) {
    if (e?.code === 'IMPORT_CANCELLED') return null;
    throw e;
  }
  const out: ImportFilesResult = { added: 0, failed: [], truncated: [] };
  const t = Date.now();
  res.files.forEach((f, fi) => {
    if (f.error) {
      out.failed.push({ name: f.name, reason: f.error === 'PDF_PROTECTED' ? 'password-protected' : 'couldn\'t be read' });
      return;
    }
    if (f.kind === 'pdf' && f.pages.length < f.totalPages) {
      out.truncated.push({ name: f.name, added: f.pages.length, total: f.totalPages });
    }
    f.pages.forEach((pg, i) => {
      addPage({
        id: `page_${t}_${fi}_${i}`,
        croppedImagePath: pg.path,
        processedImagePath: pg.path,
        rotationDegrees: 0,
        // Digital pages / photos as they are; Filters can still enhance them.
        filterType: 'ORIGINAL',
        brightness: 0,
        contrast: 0,
        // The whole image as the "photo" with its four corners, so Crop works on imported pages too.
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
      out.added++;
    });
  });
  return out;
}