from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import os

class Settings(BaseSettings):
    # 
    #  declara el API key para que pydantic no lo trate como "extra"
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Modelos
    model_translate: str = Field(default="gpt-4o-mini", alias="MT_MODEL_TRANSLATE")
    model_review: str    = Field(default="gpt-4o-mini", alias="MT_MODEL_REVIEW")
    model_classify: str  = Field(default="gpt-4o-mini", alias="MT_MODEL_CLASSIFY")
    model_embed: str     = Field(default="text-embedding-3-large", alias="MT_MODEL_EMBED")

    # Umbrales QA (compat y nuevos nombres usados en pipelines/graph)
    # Compat: se mantienen estos por si algún código externo los usa
    term_min: float = Field(default=0.98, alias="MT_TERM_MIN")
    num_min: float  = Field(default=0.98, alias="MT_NUM_MIN")
    dom_min: float  = Field(default=0.85, alias="MT_DOM_MIN")
    # Nombres efectivos usados por translate_graph._qa_ok
    qa_term_min: float = Field(default=0.98, alias="MT_QA_TERM_MIN")
    qa_num_min: float  = Field(default=0.98, alias="MT_QA_NUM_MIN")
    qa_dom_min: float  = Field(default=0.85, alias="MT_QA_DOM_MIN")
    qa_max_loops: int  = Field(default=1,    alias="MT_QA_MAX_LOOPS")
    qa_use_llm: bool = Field(default=False, alias="MT_QA_USE_LLM")
    qa_llm_model: str = Field(default="gpt-4o-mini", alias="MT_QA_LLM_MODEL")
    qa_weight_llm: float = Field(default=0.6, alias="MT_QA_WEIGHT_LLM")  # mezcla LLM vs reglas


    # Concurrencia
    max_conc_translate: int = Field(default=6, alias="MT_MAX_CONCURRENCY_TRANSLATE")
    max_conc_embed: int     = Field(default=4, alias="MT_MAX_CONCURRENCY_EMBED")
    # Rate limits (aiolimiter)
    llm_rps: float = Field(default=4.0, alias="MT_LLM_RPS")
    embed_rps: float = Field(default=2.0, alias="MT_EMBED_RPS")

    # SQLite
    db_path: str = Field(default=".local/mt.sqlite", alias="MT_DB_PATH")
    checkpoint_db: str = Field(default=".local/mt_checkpoints.sqlite", alias="MT_CHECKPOINT_DB")

    # RAG/TM
    rag_topk: int = Field(default=3, alias="MT_RAG_TOPK")
    tm_topk: int  = Field(default=1, alias="MT_TM_TOPK")


        # --- NUEVO: control de extractor/validadores LLM y trazas ---
    llm_term_extractor_enabled: bool = Field(default=True, alias="MT_LLM_TERM_EXTRACTOR")
    term_cand_topk: int = Field(default=12, alias="MT_TERM_CAND_TOPK")
    term_min_len: int = Field(default=2, alias="MT_TERM_MIN_LEN")

    llm_validators_enabled: bool = Field(default=True, alias="MT_LLM_VALIDATORS")
    llm_domain_weight: float = Field(default=0.7, alias="MT_LLM_DOMAIN_WEIGHT")  # mezcla LLM/reglas para domain_score

    # Búsqueda web
    ddg_max_results: int = Field(default=4, alias="MT_DDG_MAX_RESULTS")
    rag_trusted_domains: str = Field(
        default="eur-lex.europa.eu,oecd.org,esma.europa.eu,ilpa.org,investeurope.eu,inrev.org,epra.com,rics.org",
        alias="MT_RAG_TRUSTED_DOMAINS"
    )

    # Trazas (dev)
    trace_prompts: bool = Field(default=True, alias="MT_TRACE_PROMPTS")  # si quieres ver prompts en logs/response(debug)


    #  configuración pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",  # o "allow" si prefieres aceptar cualquier extra
    )

settings = Settings()

#  Propaga OPENAI_API_KEY al entorno por si no está ya exportado
if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

# Crea los directorios para las bases de datos si no existen
Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
