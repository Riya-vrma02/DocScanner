import { DocScannerNative } from '../native/DocScannerNative';

/** Folder of everything the app makes (exported scans, merged PDFs, LaTeX), shown on the Files screen. */
export async function filesDir(): Promise<string> {
  const docDir = await DocScannerNative.getDocumentDirectory();
  const dir = `${docDir}/files`;
  await DocScannerNative.mkdir(dir);
  return dir;
}

/** "2026-11-18_10-42-05": safe in file names and share URLs. */
export function stamp(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

export function mimeFor(name: string): string {
  const ext = name.toLowerCase().split('.').pop();
  switch (ext) {
    case 'pdf': return 'application/pdf';
    case 'zip': return 'application/zip';
    case 'tex': return 'application/x-tex';
    case 'jpg': case 'jpeg': return 'image/jpeg';
    case 'png': return 'image/png';
    default: return 'application/octet-stream';
  }
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}