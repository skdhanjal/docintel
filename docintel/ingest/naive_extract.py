import pathlib
import pdfplumber
from docintel.config import settings

def naive_text(pdf_path: pathlib.Path) -> str:
    """The 'just extract the text' approach. Deliberately dumb —
    no layout awareness, no table structure, no reading order fix.
    This is the baseline every RAG tutorial silently assumes works."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)

def find_a_table_page(pdf_path: pathlib.Path) -> int | None:
    """Locate a page pdfplumber thinks has a table, so we can
    eyeball exactly how the naive text mangles it."""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if page.find_tables():
                return i
    return None

if __name__ == "__main__":
    print("Starting extract...")
    corpus = pathlib.Path(settings.corpus_dir)
    for pdf in sorted(corpus.glob("*.pdf")):
        print(f"\n=== {pdf.name} ===")
        text = naive_text(pdf)
        page = find_a_table_page(pdf)
        print(f"  chars extracted : {len(text):,}")
        print(f"  first table page : {page}")
        # Dump the naive text of that page for inspection.
        if page is not None:
            with pdfplumber.open(pdf) as p:
                snippet = (p.pages[page].extract_text() or "")[:700]
            print("  --- naive text of a table page ---")
            print("\n".join("  " + ln for ln in snippet.splitlines()))