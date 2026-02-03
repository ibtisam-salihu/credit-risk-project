from resolver.resolver import CompanyResolver

resolver = CompanyResolver("data/raw/fame_public_universe.csv")

while True:
    q = input("Search company (blank to exit): ").strip()
    if not q:
        break

    matches = resolver.search(q, limit=5)
    for i, m in enumerate(matches, 1):
        print(
            f"{i}. {m.company_name} | "
            f"ticker={m.ticker_symbol} | "
            f"yahoo={m.yahoo_ticker} | "
            f"credit={m.credit_score} | "
            f"similarity={m.similarity:.1f}"
        )
    print("-" * 80)
