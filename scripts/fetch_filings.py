import json, time, pathlib, requests
from docintel.config import settings

HEADERS = {"User-Agent": settings.sec_user_agent}
CORPUS  = pathlib.Path(settings.corpus_dir); CORPUS.mkdir(exist_ok=True)

# Five industries on purpose — each stresses parsing differently.
TICKERS = ["JPM", "PFE", "WMT", "NVDA", "CVX"]

def cik_for(ticker: str) -> str:
    """SEC publishes the ticker→CIK map as a single JSON file."""
    m = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS).json()
    for row in m.values():
        if row["ticker"] == ticker:
            return f"{row['cik_str']:010d}"   # zero-padded to 10 digits
    raise ValueError(f"no CIK for {ticker}")

def latest_10k(cik: str) -> tuple[str, str]:
    """Return (accession_no_nodashes, primary_document) of newest 10-K."""
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                       headers=HEADERS).json()["filings"]["recent"]
    for form, acc, doc in zip(sub["form"], sub["accessionNumber"],
                            sub["primaryDocument"]):
        if form == "10-K":
            return acc.replace("-", ""), doc
    raise ValueError("no 10-K in recent filings")

def main():
    for t in TICKERS:
        cik = cik_for(t); time.sleep(0.2)
        acc, doc = latest_10k(cik); time.sleep(0.2)
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{int(cik)}/{acc}/{doc}")
        out = CORPUS / f"{t}_10-K.html"
        out.write_bytes(requests.get(url, headers=HEADERS).content)
        print(f"{t}: saved {out.name}  ({url})")
        time.sleep(0.3)

if __name__ == "__main__":
    main()