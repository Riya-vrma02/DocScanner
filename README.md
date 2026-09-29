# DocScanner (React Native / TypeScript)

Cross-platform (iOS + Android) rebuild of the same Adobe Scan-style pipeline,
now in React Native.

**Capture (native auto-crop) → color/intensity filter + rotate (Skia) → add to
page list (reorderable) → combine all pages into one PDF (A4 / Legal / Letter)**

## Why these libraries
Pure JS/RN can't do real-time contour detection or true perspective warping on
its own — that needs either native code or a heavy WASM/OpenCV bundle. Instead
this app leans on libraries that already wrap the right native APIs, so you get
a real cross-platform app without hand-rolling a native module:

| Feature | Library | What it does |
|---|---|---|
| Capture + auto edge-detect + perspective-correct crop | [`react-native-document-scanner-plugin`](https://www.npmjs.com/package/react-native-document-scanner-plugin) | Wraps Apple's WeScan (iOS) and an ML-Kit-based scanner (Android). Detects the document, lets the user drag corners, and returns an already flattened/perspective-corrected image — this **is** the "correcting page orientation and distortion" step. |
| Color/intensity filters, rotation | [`@shopify/react-native-skia`](https://shopify.github.io/react-native-skia/) | Renders the cropped image through a `ColorMatrix` (grayscale, auto-enhance, brightness/contrast) on an **offscreen Skia surface**, then rotates and encodes the result to a real JPEG file (`renderProcessedImage.ts`) — so the edit is baked in, not just a live-preview effect. |
| Page size (A4/Legal/Letter) + combining pages into one PDF | [`pdf-lib`](https://pdf-lib.js.org/) | Pure JS, no native PDF module needed. Creates a page at exact A4/Legal/Letter point dimensions per scan, auto-picks portrait/landscape to match that scan, and embeds the JPEG scaled-to-fit — all pages go into one `PDFDocument`. |
| Reorder pages before export | [`react-native-draggable-flatlist`](https://github.com/computerjazz/react-native-draggable-flatlist) | Long-press-drag reordering in `HomeScreen`. |
| Filesystem | `react-native-fs` | Reading/writing the intermediate and final image/PDF files. |

## Project layout
```
App.tsx                        navigation shell (Home → Scan → Filter)
src/
  types/index.ts                ScanPage model, PAGE_SIZES (A4/Legal/Letter in pt)
  store/pagesStore.ts            zustand store holding the session's pages
  screens/
    HomeScreen.tsx                page list, drag-to-reorder, export dialog
    ScanScreen.tsx                launches the native scanner plugin
    FilterScreen.tsx              filter presets, brightness/contrast, rotate
  lib/
    imageFilters.ts               color-matrix math for each filter preset
    renderProcessedImage.ts       Skia offscreen render: rotation + filter -> JPEG file
    pdfExport.ts                  pdf-lib: combine all pages into one sized PDF
  components/
    PageListItem.tsx              thumbnail row in the page list
```

## Setup
This is a **ready-to-run bare React Native project** — the `ios/` and `android/`
native folders are already scaffolded and wired up (camera permission added to
`Info.plist` / `AndroidManifest.xml`, bundle ID / package set to `com.docscannerrn`).
It needs the full native toolchain — it won't run in plain Expo Go.

```bash
npm install
cd ios && pod install && cd ..   # iOS only, needs a Mac + Xcode
npx react-native run-ios         # or: npx react-native run-android
```

Android needs `ANDROID_HOME` set and either an emulator running or a device
connected with USB debugging on. iOS needs Xcode + CocoaPods installed.

## Notes / next steps
- **Persist documents across app restarts**: add zustand's `persist` middleware
  (AsyncStorage/MMKV) to `pagesStore.ts`, or move to SQLite for multiple saved documents.
- **OCR / searchable PDF text layer**: add `@react-native-ml-kit/text-recognition`
  and draw an invisible text layer into each `pdf-lib` page alongside the image.
- **Batch capture**: `DocumentScanner.scanDocument({ maxNumDocuments: N })` can
  return several pages from one session if you want rapid multi-page capture
  instead of one-scan-then-edit-then-repeat.
