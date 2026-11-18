import { ScanPage } from '../types';

/** Image URI of a page; the version makes a re-processed file with the same name reload. */
export function pageUri(p: ScanPage) {
  return `file://${p.processedImagePath}?v=${p.version ?? 0}`;
}