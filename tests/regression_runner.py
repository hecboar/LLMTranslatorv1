import asyncio, yaml, sys
from pathlib import Path
from app.pipelines.translate_graph import run_pipeline_graph
from app.stores.term_store import TermStore, DNTItem

THRESH_NUM = 1.0
THRESH_TERM = 0.98

def load_yaml(p: Path): 
    return yaml.safe_load(p.read_text(encoding="utf-8"))

async def seed_from_fixture(ts: TermStore, fx: dict, client_id: str):
    for concept_key, langs in fx.get("glossary", {}).items():
        for lang, preferred in langs.items():
            await ts.set_global_preferred(concept_key, lang, preferred)
    for t in fx.get("dnt", []):
        await ts.add_dnt(DNTItem(client_id=client_id, term=t))

async def run_one_fixture(path: Path) -> bool:
    from app.config import settings
    ts = TermStore(settings.db_path)
    fx = load_yaml(path)
    client_id = fx.get("client_id","qa_client")
    await seed_from_fixture(ts, fx, client_id)
    ok = True
    for case in fx["cases"]:
        res = await run_pipeline_graph(
            text=case["text"],
            client_id=client_id,
            targets=fx.get("targets", ["en","fr","de"]),
            src_lang_override=case.get("src_lang"),
            domain_override=fx.get("domain"),
            enable_rag=False,
            save_tm=False
        )
        for L in fx.get("targets", ["en","fr","de"]):
            qa = res["results"][L]["qa"]
            n_ok = (qa["numeric_consistency"] == THRESH_NUM)
            t_ok = (qa["term_coverage"] >= THRESH_TERM)
            print(f"[{path.name}][{L}] num={qa['numeric_consistency']:.2f} term={qa['term_coverage']:.2f} -> {'OK' if (n_ok and t_ok) else 'FAIL'}")
            ok = ok and n_ok and t_ok
    return ok

async def main():
    fx_dir = Path(__file__).parent / "fixtures"
    failures = 0
    for p in sorted(fx_dir.glob("*.yaml")):
        ok = await run_one_fixture(p)
        if not ok:
            failures += 1
    if failures:
        print(f"\nFAILED fixtures: {failures}", file=sys.stderr)
        sys.exit(1)
    print("\nAll fixtures passed ✓")

if __name__ == "__main__":
    asyncio.run(main())
