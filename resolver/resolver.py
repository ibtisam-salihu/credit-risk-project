# resolver/resolver.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import process, fuzz


@dataclass(frozen=True)
class CompanyMatch:
    company_name: str
    ticker_symbol: Optional[str]
    isin: Optional[str]
    sic_code: Optional[str]
    sic_description: Optional[str]
    credit_score: Optional[float]
    similarity: float

    @property
    def yahoo_ticker(self) -> Optional[str]:
        """Convert UK EPIC/ticker to Yahoo Finance format (e.g., TSCO -> TSCO.L)."""
        if not self.ticker_symbol:
            return None
        t = str(self.ticker_symbol).strip().upper()
        if t.lower() in {"n.a.", "n.s.", "na", ""}:
            return None
        return f"{t}.L"


class CompanyResolver:
    """
    Resolves user-entered company names (or tickers) to canonical records
    using the FAME public universe CSV.
    """

    REQUIRED_COLS = [
        "Company name",
        "Ticker symbol",
        "ISIN number",
        "Primary UK SIC (2007) code",
        "Primary UK SIC (2007) description",
        "Credit score",
    ]

    def __init__(self, fame_csv_path: str | Path):
        self.fame_csv_path = Path(fame_csv_path)
        if not self.fame_csv_path.exists():
            raise FileNotFoundError(f"FAME CSV not found: {self.fame_csv_path}")

        # Load CSV
        self.df = pd.read_csv(self.fame_csv_path)

        # Drop accidental unnamed index column (common in exports)
        if len(self.df.columns) > 0 and (
            str(self.df.columns[0]).startswith("Unnamed") or str(self.df.columns[0]).strip() in {"", " "}
        ):
            self.df = self.df.drop(columns=[self.df.columns[0]])

        # Validate required columns exist
        missing = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(
                "FAME CSV missing required columns:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nRe-export from FAME including these columns."
            )

        # Clean key columns
        self.df["Company name"] = (
            self.df["Company name"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        self.df["Ticker symbol"] = (
            self.df["Ticker symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Combined key: allows searching by name OR ticker
        self.df["__search_key"] = self.df["Company name"] + " | " + self.df["Ticker symbol"]
        self.search_keys = self.df["__search_key"].tolist()

    def _row_to_match(self, row: pd.Series, similarity: float) -> CompanyMatch:
        def _clean_str(v) -> Optional[str]:
            if pd.isna(v):
                return None
            s = str(v).strip()
            return None if s.lower() in {"n.a.", "n.s.", "na", ""} else s

        def _clean_float(v) -> Optional[float]:
            if pd.isna(v):
                return None
            s = str(v).strip().lower()
            if s in {"n.a.", "n.s.", "na", ""}:
                return None
            try:
                return float(v)
            except Exception:
                return None

        return CompanyMatch(
            company_name=_clean_str(row["Company name"]) or "",
            ticker_symbol=_clean_str(row["Ticker symbol"]),
            isin=_clean_str(row["ISIN number"]),
            sic_code=_clean_str(row["Primary UK SIC (2007) code"]),
            sic_description=_clean_str(row["Primary UK SIC (2007) description"]),
            credit_score=_clean_float(row["Credit score"]),
            similarity=float(similarity),
        )

    def search(self, query: str, limit: int = 10) -> list[CompanyMatch]:
        """
        Returns top matches for a query.
        - If query matches a ticker exactly, returns that immediately with similarity=100.
        - Otherwise fuzzy matches against 'Company name | TICKER'.
        """
        query = (query or "").strip()
        if not query:
            return []

        q_upper = query.upper()

        # 1) Exact ticker shortcut (fast + accurate)
        exact = self.df[self.df["Ticker symbol"] == q_upper]
        if len(exact) > 0:
            row = exact.iloc[0]
            return [self._row_to_match(row, similarity=100.0)]

        # 2) Fuzzy match
        results = process.extract(
            query,
            self.search_keys,
            scorer=fuzz.WRatio,
            limit=limit,
        )

        matches: list[CompanyMatch] = []
        for _, score, idx in results:
            row = self.df.iloc[idx]
            matches.append(self._row_to_match(row, similarity=float(score)))

        return matches

    def resolve_one(self, query: str, min_similarity: float = 85.0) -> Optional[CompanyMatch]:
        """
        Convenience method: returns top match if similarity is above threshold,
        otherwise None (so UI can ask user to choose).
        """
        results = self.search(query, limit=1)
        if not results:
            return None
        best = results[0]
        return best if best.similarity >= min_similarity else None