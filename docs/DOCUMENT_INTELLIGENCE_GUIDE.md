# Document Intelligence Guide

## The stack

Documents = content + layout + structure + metadata. Extract each.

```
PDF/DOCX/image → Preprocessing → Layout analysis
              → Text extraction (PDF layer / OCR / VLM)
              → Structure recognition
              → Table extraction
              → Figure/chart understanding
              → Multi-page consolidation
              → Semantic indexing (ColPali + text)
              → Application (QA, extraction, summary)
```

## Choosing extraction method

- Native text, simple layout → text-layer extraction (`pdfplumber`, `PyMuPDF`).
- Scanned, simple layout → OCR (Tesseract, PaddleOCR).
- Complex layout with tables/figures → OCR + layout model (LayoutLMv3) + VLM for complex regions.
- Full document QA → VLM directly (ColPali retrieval + VLM QA).

## Real-world attrition (10K PDFs)

- 60% cleanly text-layer extractable.
- 30% need OCR + layout.
- 10% only work with VLM on rasterized pages.

Budget accordingly.

## Tables

Hardest common element. Approaches:
- Table transformer (TATR) + OCR.
- VLM with schema in prompt.
- Cloud services (Textract, Document AI, Form Recognizer).

For high-value tables, use two methods and reconcile.

## ColPali visual RAG

For documents with heavy visual content (financial reports, papers with figures):
- Rasterize page → embed as image.
- Retrieve pages by visual similarity.
- VLM QA on retrieved.

Higher cost than text RAG but captures layout + figures + charts.
