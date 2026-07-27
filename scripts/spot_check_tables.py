
import pathlib, random
from docintel.ingest.parse_pipeline import load_cached

doc = load_cached(pathlib.Path("parsed/NVDA_10-K.json"))   # swap filenames to sample across filings

def sample(n_per_file: int = 4):
    """Print random parsed tables with their page, for hand-scoring."""
    tables = doc.tables

    for i, table in enumerate(random.sample(tables, min(n_per_file, len(tables)))):
        page = table.prov[0].page_no if table.prov else None
        print(f"[{i}] page {page}")
        print(table.export_to_dataframe(doc).head())
        print()
    
    
if __name__ == "__main__":
    random.seed(20)
    sample()    