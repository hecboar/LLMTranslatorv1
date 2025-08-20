import os, asyncio, pytest, shutil
from pathlib import Path
import yaml

@pytest.fixture(scope="session", autouse=True)
def isolate_db(tmp_path_factory):
    base = tmp_path_factory.mktemp("dbs")
    os.environ["MT_DB_PATH"] = str(base / "mt.sqlite")
    os.environ["MT_CHECKPOINT_DB"] = str(base / "mt_check.sqlite")
    yield

class SeedEnv:
    def __init__(self, base):
        self.base = Path(base)
    async def load(self, fixture_path: str):
        p = self.base / fixture_path
        return yaml.safe_load(p.read_text(encoding="utf-8"))

@pytest.fixture
def seeded_env():
    return SeedEnv(base=Path.cwd())

@pytest.fixture(autouse=True)
def _silence_network(monkeypatch):
    # tests de regresión usan enable_rag=False; si algo llama DDG por error, mejor que no falle catastróficamente
    yield
