import hashlib
import pathlib
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True       # structured table extraction

_converter_instance = None

def get_document_converter() -> DocumentConverter:
    """Returns a singleton instance of DocumentConverter."""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
    return _converter_instance

def content_hash(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

def parse_and_cache(pdf_path: pathlib.Path, out_dir: pathlib.Path) -> DoclingDocument:
    """PDF -> DoclingDocument, cached to disk as Docling's own JSON.
    This is the ONLY place that pays for layout + table-structure
    inference -- every later stage (today's scoring, Day 21's
    HybridChunker, any future re-chunk) reloads the cache instead
    of re-parsing the PDF."""
    # h = content_hash(pdf_path)
    out = out_dir / f"{pdf_path.stem}.json"

    if out.exists():
        print(f"Cache hit: {out.name}"); 
        return out
    else:
        print(f"parsing {pdf_path.name}...")
            
    result = get_document_converter().convert(str(pdf_path))
    doc = result.document
    doc.save_as_json(out)          # native serialization -- nothing hand-rolled
    n_tables = len(doc.tables)
    
    print(f"{pdf_path.name:20s} {len(doc.texts):4d} text items   {n_tables:3d} tables")
    return doc


def load_cached(json_path: pathlib.Path) -> DoclingDocument:
    """Reload a cached parse -- structurally identical to a fresh
    convert(), just without paying for the layout/table models
    again. This is what Day 21 calls before chunking."""
    return DoclingDocument.load_from_json(json_path)