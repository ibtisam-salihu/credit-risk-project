# Simple Company Resolver - Searches FAME database for UK companies

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import pandas as pd
from rapidfuzz import process, fuzz


@dataclass(frozen=True)
class CompanyMatch:
    # represents matched company from fame 
    company_name: str
    ticker_symbol: Optional[str]
    isin: Optional[str]
    sic_code: Optional[str]
    sic_description: Optional[str]
    credit_score: Optional[float]
    similarity: float

    @property
    def yahoo_ticker(self) -> Optional[str]:
        # Converts uk ticker to yahoon finance ticker 
        if not self.ticker_symbol:
            return None
        ticker = str(self.ticker_symbol).strip().upper()
        if ticker.lower() in {"n.a.", "n.s.", "na", ""}:
            return None
        return f"{ticker}.L"


class CompanyResolver:
    # searches for companies in fame with fuzzy matching 

    # important colours in fame civ
    REQUIRED_COLUMNS = [
        "Company name",
        "Ticker symbol",
        "ISIN number",
        "Primary UK SIC (2007) code",
        "Primary UK SIC (2007) description",
        "Credit score",
    ]

    def __init__(self, fame_csv_path: str | Path):
        # Load FAME database from CSV file
        self.fame_csv_path = Path(fame_csv_path)
        
        # to check data exists
        if not self.fame_csv_path.exists():
            raise FileNotFoundError(f"FAME CSV not found: {self.fame_csv_path}")

        # load csv
        self.df = pd.read_csv(self.fame_csv_path)
        
        # removes any unamed index coloums 
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]

        
        # check required colours exist 
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        
        # clean company names 
        self.df["Company name"] = self.df["Company name"].fillna("").astype(str).str.strip()
        
        # clean the ticker symbols 
        self.df["Ticker symbol"] = self.df["Ticker symbol"].fillna("").astype(str).str.strip().str.upper()
        
        # searchable by combination of ticker and name 
        self.df["__search_key"] = self.df["Company name"] + " | " + self.df["Ticker symbol"]
        self.search_keys = self.df["__search_key"].tolist()

    def _clean_value(self, value) -> Optional[str]:
        # Cleans string values from csv and returns none if invalid 
        if pd.isna(value):
            return None
        cleaned = str(value).strip()
        return None if cleaned.lower() in {"n.a.", "n.s.", "na", ""} else cleaned

    def _clean_number(self, value) -> Optional[float]:
        # Cleans numeric values from csv returns none if invalid 
        if pd.isna(value):
            return None
        try:
            return float(value)
        except:
            return None

    def _row_to_match(self, row: pd.Series, similarity: float) -> CompanyMatch:
        # Converts DataFrame row to CompanyMatch object
        return CompanyMatch(
            company_name=self._clean_value(row["Company name"]) or "",
            ticker_symbol=self._clean_value(row["Ticker symbol"]),
            isin=self._clean_value(row["ISIN number"]),
            sic_code=self._clean_value(row["Primary UK SIC (2007) code"]),
            sic_description=self._clean_value(row["Primary UK SIC (2007) description"]),
            credit_score=self._clean_number(row["Credit score"]),
            similarity=float(similarity),
        )

    def search(self, query: str, limit: int = 10) -> List[CompanyMatch]:
        # Search for companies by name or ticker, return top matches
        query = (query or "").strip()
        if not query:
            return []
        
        # Try exact ticker match first
        exact = self.df[self.df["Ticker symbol"] == query.upper()]
        if len(exact) > 0:
            return [self._row_to_match(exact.iloc[0], similarity=100.0)]
        
        # Fuzzy search on company name and ticker
        results = process.extract(query, self.search_keys, scorer=fuzz.WRatio, limit=limit)
        
        # Convert results to CompanyMatch objects
        matches = []
        for _, score, idx in results:
            matches.append(self._row_to_match(self.df.iloc[idx], similarity=float(score)))
        
        return matches

    def resolve_one(self, query: str, min_similarity: float = 85.0) -> Optional[CompanyMatch]:
        # Get single best match if confidence is above threshold, else return None
        results = self.search(query, limit=1)
        if not results:
            return None
        
        best = results[0]
        return best if best.similarity >= min_similarity else None