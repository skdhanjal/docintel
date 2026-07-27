import pathlib
from docling.document_converter import DocumentConverter

# One converter, reused — model load is the expensive part.
_CONVERTER = DocumentConverter()

def parse_document(src: pathlib.Path):
    """PDF/HTML/DOCX -> DoclingDocument (typed elements + tables)."""
    result = _CONVERTER.convert(str(src))
    return result.document

if __name__ == "__main__":
    doc = parse_document(pathlib.Path("corpus/NVDA_10-K.pdf"))
    md = doc.export_to_markdown()
    print(md[:1200])
    print("...")
    print("# tables:", len(doc.tables))