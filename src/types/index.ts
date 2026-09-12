export type FilterType = 'ORIGINAL' | 'AUTO_ENHANCE' | 'GRAYSCALE' | 'BLACK_AND_WHITE';

export type PageSizeName = 'A4' | 'LEGAL' | 'LETTER';

// Standard paper sizes in PDF points (1 pt = 1/72 inch), portrait orientation.
export const PAGE_SIZES: Record<PageSizeName, { widthPt: number; heightPt: number }> = {
  A4: { widthPt: 595.28, heightPt: 841.89 },
  LEGAL: { widthPt: 612, heightPt: 1008 },
  LETTER: { widthPt: 612, heightPt: 792 },
};

export interface ScanPage {
  id: string;
  /** Path to the image AFTER the native scanner plugin's auto-crop + perspective correction. */
  croppedImagePath: string;
  /** Path to the image after filters/rotation are baked in — this is what gets exported. */
  processedImagePath: string;
  rotationDegrees: number; // 0, 90, 180, 270
  filterType: FilterType;
  brightness: number; // -1..1
  contrast: number;   // -1..1
}
