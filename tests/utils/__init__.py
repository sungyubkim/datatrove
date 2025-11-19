# Re-export decorators from parent-level utils.py module
# Note: tests/utils.py (module) exists alongside tests/utils/ (package)
# We need to load the utils.py module directly to avoid circular imports

import importlib.util
from pathlib import Path

# Load utils.py module directly by file path
utils_py_path = Path(__file__).parent.parent / "utils.py"
spec = importlib.util.spec_from_file_location("_tests_utils_module", utils_py_path)
_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_utils_module)

# Re-export all decorators
require_boto3 = _utils_module.require_boto3
require_datasets = _utils_module.require_datasets
require_fasttext = _utils_module.require_fasttext
require_inscriptis = _utils_module.require_inscriptis
require_lighteval = _utils_module.require_lighteval
require_moto = _utils_module.require_moto
require_nltk = _utils_module.require_nltk
require_pyarrow = _utils_module.require_pyarrow
require_readability = _utils_module.require_readability
require_s3fs = _utils_module.require_s3fs
require_tldextract = _utils_module.require_tldextract
require_tokenizers = _utils_module.require_tokenizers
require_trafilatura = _utils_module.require_trafilatura
require_xxhash = _utils_module.require_xxhash
use_hash_configs = _utils_module.use_hash_configs

__all__ = [
    "require_boto3",
    "require_datasets",
    "require_fasttext",
    "require_inscriptis",
    "require_lighteval",
    "require_moto",
    "require_nltk",
    "require_pyarrow",
    "require_readability",
    "require_s3fs",
    "require_tldextract",
    "require_tokenizers",
    "require_trafilatura",
    "require_xxhash",
    "use_hash_configs",
]
