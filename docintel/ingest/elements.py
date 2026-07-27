from dataclasses import dataclass, field
import pathlib
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

pipeline_options = PdfPipelineOptions()

pipeline_options.do_table_structure = True       # structured table extraction

_CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend
        )
    }
)

@dataclass
class Element:
    kind: str            # "heading" | "text" | "table" | "list" | ...
    text: str            # NL content (table -> markdown/blurb)
    page: int | None = None
    meta: dict = field(default_factory=dict)


def extract_elements(src: pathlib.Path) -> list[Element]:
    doc = _CONVERTER.convert(str(src)).document
    out: list[Element] = []

    # Tables: keep BOTH the grid (markdown) and a short NL description,
    # because embedders match prose better than pipe-delimited cells.
    for t in doc.tables:
        page = t.prov[0].page_no if t.prov else None
        grid_md = t.export_to_markdown(doc)
        out.append(Element(
            kind="table", text=grid_md, page=page,
            meta={"n_rows": len(t.data.grid) if t.data else 0},
        ))

    # Text-like items in reading order, typed by Docling's labels.
    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        txt = getattr(item, "text", None)
        if not txt:
            continue
        kind = "heading" if label and "title" in str(label).lower() \
               else "heading" if label and "section" in str(label).lower() \
               else "text"
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        out.append(Element(kind=kind, text=txt, page=page,
                           meta={"label": str(label)}))
    return out


if __name__ == "__main__":
    els = extract_elements(pathlib.Path("corpus/NVDA_10-K.pdf"))
    from collections import Counter
    print(Counter(e.kind for e in els))
    tbl = next(e for e in els if e.kind == "table")
    print(f"\nsample table (page {tbl.page}):\n", tbl.text)
