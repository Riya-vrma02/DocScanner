# DocScanner — React Native / TypeScript

A cross-platform document scanning application built with **React Native and TypeScript**.

The application follows an Adobe Scan-style document workflow, allowing users to capture document images, correct document perspective and orientation, apply image processing, manage multiple pages, and generate a final PDF.

## Features

- 📷 Document image capture
- 🔲 Document boundary detection and auto-cropping
- 📐 Perspective correction / dewarping
- 🔄 Image rotation
- 🎨 Image enhancement and color/intensity processing
- 📄 Multiple-page document management
- ↕️ Page reordering
- 📑 PDF generation
- 📏 A4, Legal, and Letter page sizes
- 📱 Android and iOS support
- ⚡ React Native + TypeScript architecture

---

## Document Processing Pipeline

The application follows the following workflow:

```text
Camera / Image Input
        ↓
Document Detection
        ↓
Document Boundary Detection
        ↓
Perspective Correction / Dewarping
        ↓
Crop & Orientation Correction
        ↓
Image Processing
        ↓
Add Page to Document
        ↓
Reorder Pages
        ↓
Select PDF Page Size
        ↓
Generate Final PDF
