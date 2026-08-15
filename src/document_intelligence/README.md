# 📄 Document Intelligence — The Enterprise Killer App

> *Every enterprise has documents. Every enterprise wants to search them, extract from them, answer questions about them. Document intelligence is where multimodal AI generates the most measurable business value in 2026.*

---

## Document Understanding Stack (`document_understanding_stack.py`)

**Documents have:** content (words, numbers, figures); layout (columns, tables, headers); structure (sections, subsections, lists); metadata.

**Full stack:**
```
Document Input (PDF, DOCX, image, scan)
    ↓
Preprocessing (format conversion, page extraction)
    ↓
Layout Analysis (columns, tables, figures, reading order)
    ↓
Text Extraction (PDF text layer, OCR, VLM)
    ↓
Structure Recognition (headings, sections, lists)
    ↓
Table Extraction (rows, columns, cells)
    ↓
Figure/Chart Understanding
    ↓
Multi-Page Consolidation (cross-page refs, TOCs)
    ↓
Semantic Indexing (embed, store)
    ↓
Application-Specific (QA, summary, extraction, review)
```

Each stage has failure modes.

**Distinctions:**
- **Text-only:** flat text. Loses structure. Easy.
- **Layout-aware:** preserves spatial + logical. Harder, more useful.
- **Visual (VLM-based):** doc as image. Highest fidelity for complex.

---

## PDF Extraction Strategies (`pdf_extraction_strategies.py`)

**PDFs are the hardest common format.** Can be: pure text, scanned images, mixed, encrypted, malformed.

**Approaches:**

**Text-layer extraction:** `pdfplumber`, `PyMuPDF`, `pdfminer.six`. Works when PDF has text layer. Fast, cheap. Loses layout beyond coords.

**OCR:** Tesseract, PaddleOCR, EasyOCR, AWS Textract, Google Document AI. Required for scanned. Quality varies enormously.

**Layout models:** LayoutParser, PubLayNet, Detectron2 with document models. Detect regions (title, paragraph, table, figure).

**VLM directly on rasterized pages:** page → image → VLM. Highest quality for complex. 10-100× cost of OCR.

**Practical tree:**
1. Native text, simple layout → text extraction.
2. Complex layout with tables/figures → layout model + OCR + VLM for complex regions.
3. Full document QA → VLM directly (ColPali-style).

**Real-world attrition** in 10K PDFs: 60% text-layer extractable, 30% need OCR+layout, 10% only work with VLM.

---

## Layout-Aware Models (`layout_aware_models.py`)

**LayoutLM family (Microsoft):** v1 (2019) text + 2D positions; v2 adds image features; v3 unified pretraining. Strong on forms, invoices, standardized docs.

**Donut (Kim et al. 2022):** OCR-free. ViT takes doc image directly. Output structured JSON or text. Good for structured extraction.

**Others:** DocFormer, LiLT, Nougat (scientific papers).

**Recent VLM-based:** Nougat, ColPali. General VLMs (Qwen2.5-VL, Claude, GPT-4o) often outperform specialized when strong VLM available.

**Trade-off:** specialized cheaper/faster but plateau in quality. VLMs more expensive but higher quality on complex.

---

## OCR When Still Needed (`ocr_when_still_needed.py`)

**Reasons to still use OCR:**
1. Cost: $0.001-0.01/page vs VLM $0.01-0.10.
2. Latency: 100ms-1s vs VLM 2-10s.
3. Scale: millions of pages/day → OCR is only economically feasible.
4. Simple documents.
5. Multi-language.

**Modern OCR engines:** Tesseract (mature, mediocre on messy); PaddleOCR (multilingual, strong); AWS Textract (layout, tables); Google Document AI (form parsing, tables); Azure Form Recognizer.

**OCR + LLM hybrid:**
1. OCR extracts text + bounding boxes.
2. Layout heuristics/model organize into sections.
3. LLM processes structured text.
4. VLM fallback on ambiguous image regions.

**Verify quality:** CER on labeled sample. 95%+ char accuracy typical clean scans; 80% degraded.

---

## Table Extraction (`table_extraction.py`)

**Hardest common element.** Cells span rows/columns; nested tables; multi-level headers; merged cells; no consistent formatting.

**Approaches:**

**Rule-based:** detect ruling lines. Fragile — fails without visible borders.

**Table transformer / TATR (Microsoft):** deep model detects table structures. Cell bounding boxes. Combine with OCR for text.

**VLM-based:** "Extract this table as JSON with columns X, Y, Z." Handles complex layouts. Better with explicit schema.

**Specialized services:** AWS Textract, Google Document AI, Azure Form Recognizer have table modes. Higher quality than open-source generally.

**Quality checks:** row/column count as expected; content complete (no truncation); merged cells correctly; type check (numeric in numeric columns).

**Rule:** for high-value tables, two independent methods, reconcile disagreements.

---

## Multi-Page Reasoning (`multi_page_reasoning.py`)

**Cross-page context matters.**

**Challenges:** table headers on page N, data on pages N+1 through N+5; references to earlier sections; TOC + content pages.

**Strategies:**

**Long-context single-shot:** feed entire doc. Simple but expensive. Not viable beyond ~100 pages.

**Chunked with cross-reference resolution:** chunk by section/page. Two-pass: extract references/metadata; answer using metadata.

**RAG over chunks:** chunk, embed, retrieve per query. Add page metadata for citations.

**Hierarchical:** summarize each page → summary of doc. Query summaries first, drill down.

**Visual (ColPali-style):** embed each page as image. Retrieve pages by visual similarity. VLM does QA.

**In practice:** RAG with page-level chunking + VLM for complex pages is the workhorse.

---

## ColPali and Visual RAG (`colpali_and_visual_rag.py`)

**ColPali (Faysse et al. 2024)** — retrieval based on visual page embeddings.

**Idea:**
- Standard doc RAG: OCR → text → embed → retrieve.
- ColPali: rasterize page → embed as image → retrieve by visual similarity.

**Why:** layout is signal ("page with a table" looks different from "page of prose"). Charts and figures OCR-invisible but semantically meaningful. Multi-modal similarity captures both content and structure.

**Architecture:** ColPali model produces multi-vector embeddings per page (one per patch region). Late-interaction retrieval (like ColBERT). Query embedded, best matched to page embeddings.

**When:** documents with heavy visual content (financial reports, scientific papers, presentations); OCR-based RAG misses relevant content; need to show *page image* with answer.

**Cost:** higher than text RAG. Multi-vector needs more storage. Retrieval slightly more expensive.

**Adoption:** growing rapidly since late 2024. Modern document-heavy RAG systems increasingly use ColPali or similar.

---

## Document QA Systems (`document_qa_systems.py`)

**End-to-end architecture:**
```
User Query
    ↓
Query Rewriting (multi-hop)
    ↓
Retrieval
    ├── Text embedding (chunked text)
    ├── Visual retrieval (ColPali, over pages)
    └── Metadata filter (date, doc type)
    ↓
Reranking
    ↓
Context Assembly (top-K chunks/pages + metadata)
    ↓
Answer Generation (LLM or VLM)
    ├── With page-level citations
    └── With "unable to answer" behavior
    ↓
Answer + Citations
```

**Design choices:** chunk granularity (page, paragraph, semantic); retrieval modality (text, visual, hybrid); answer LLM (text-only cheaper; VLM better for complex layouts); citation format; "not found" handling.

**Common failures:** hallucinated citations; missed table/figure info; wrong document retrieved; multi-hop only getting one part.

**Metrics:** answer accuracy (human-judged); citation accuracy (does cited page actually contain answer?); retrieval recall; answer completeness.

---

## Forms and Structured Extraction (`forms_and_structured_extraction.py`)

**Most common enterprise use case.** Given a form (invoice, PO, tax form, contract), extract fields into structured data.

**Pipeline:**

1. **Classification.** What kind of form?
2. **Field detection.** What fields does this type have?
3. **Field extraction.** Extract per field. Constrained decoding for schema. Validation (dates parse, numbers numeric).
4. **Confidence scoring.** Per-field. Low-confidence flagged for review.
5. **Human-in-the-loop review.** UI showing extracted fields + source region. Human confirms/edits. Confirmed feeds back to training.

**Frameworks:** Docling (IBM), Unstructured, custom VLM pipelines.

**Metrics:** field accuracy per form type; full-document accuracy; human touch rate; cost per document.

**Real-world numbers:**
- Simple invoices: 95%+ automated (5% human).
- Complex contracts: 60-80% automated.
- Custom forms regularly added: 40-70%.

**Rule:** always have a human-in-the-loop review path for high-value extraction. "Correct + auditable" beats "fully automated but occasionally wrong."

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `document_understanding_stack.py` | Layout, structure, content |
| `pdf_extraction_strategies.py` | Text, tables, figures |
| `layout_aware_models.py` | LayoutLM, Donut, Nougat |
| `ocr_when_still_needed.py` | OCR + LLM hybrid |
| `table_extraction.py` | The hard problem |
| `multi_page_reasoning.py` | Cross-page context |
| `colpali_and_visual_rag.py` | Image-based RAG |
| `document_qa_systems.py` | End-to-end architecture |
| `forms_and_structured_extraction.py` | Enterprise pattern |

---

*Previous: [← Vision-Language Models](../vision_language_models/README.md) · Next: [Audio and Speech →](../audio_and_speech/README.md)*  ·  *Back to [main README](../../README.md)*
