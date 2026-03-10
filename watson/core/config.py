import os
from dotenv import load_dotenv

load_dotenv()
# Load from project root (two levels up from watson/core/config.py)
ROOT_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path=ROOT_ENV, override=True)

_ONTOLOGY_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ontology"))


class Config:
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Gemini (Google) Settings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")

    # Claude (Anthropic) Settings
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "65536"))

    # Mapping to resolve model/key by provider
    @classmethod
    def get_provider_config(cls, provider=None):
        p = provider or cls.LLM_PROVIDER
        if p == "openai":
            return {"model": cls.OPENAI_MODEL, "api_key": cls.OPENAI_API_KEY}
        elif p == "gemini":
            return {"model": cls.GOOGLE_MODEL, "api_key": cls.GOOGLE_API_KEY}
        elif p == "claude":
            return {"model": cls.ANTHROPIC_MODEL, "api_key": cls.ANTHROPIC_API_KEY}
        elif p == "ollama":
            return {"model": cls.OLLAMA_MODEL, "base_url": cls.OLLAMA_BASE_URL}
        return {}

    # MCP Settings
    MCP_SERVER_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp", "universal-ontology-mcp", "main.py")
    )

    # Ontology schema directories
    ONTOLOGY_DIRS = {
        "uco":    os.path.join(_ONTOLOGY_BASE, "uco"),
        "stix":   os.path.join(_ONTOLOGY_BASE, "stix"),
        "malont": os.path.join(_ONTOLOGY_BASE, "malont"),
        "all":    _ONTOLOGY_BASE,
    }
    ONTOLOGY_DIR = ONTOLOGY_DIRS["uco"]  # default

    # Default Pipeline Settings
    DEFAULT_CHUNK_SIZE = 4000
    DEFAULT_CHUNK_OVERLAP = 400

    # Evaluation LLM (used by LLMMatcher; can be separate from prediction LLM)
    EVAL_LLM_PROVIDER = os.getenv("EVAL_LLM_PROVIDER", "ollama")
    EVAL_LLM_MODEL    = os.getenv("EVAL_LLM_MODEL",    "llama3.1:8b")
    EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", "http://localhost:11434")

    @classmethod
    def set_schema(cls, schema: str):
        """Switch the active ontology schema (uco | stix | all)."""
        if schema not in cls.ONTOLOGY_DIRS:
            raise ValueError(f"Unknown schema '{schema}'. Choose from: {list(cls.ONTOLOGY_DIRS)}")
        cls.ONTOLOGY_DIR = cls.ONTOLOGY_DIRS[schema]

    @classmethod
    def validate(cls):
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")


config = Config()
