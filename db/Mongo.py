from datetime import datetime
import os
from urllib.parse import quote_plus, unquote_plus
from urllib.parse import urlsplit, urlunsplit

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, PyMongoError


load_dotenv()
_client = None
_translations_collection = None
APP_NAME = os.getenv("MONGO_APP_NAME", "Transformer0")
DEFAULT_DB_NAME = os.getenv("MONGO_DB_NAME", "Transformer0")
DEFAULT_COLLECTION = os.getenv("MONGO_COLLECTION_NAME", "translations")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_mongo_uri(uri: str) -> str:
    """Escape username/password in Mongo URI if needed."""
    parsed = urlsplit(uri)

    if "@" not in parsed.netloc:
        return uri

    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    if ":" not in userinfo:
        return uri

    username, password = userinfo.split(":", 1)
    safe_user = quote_plus(unquote_plus(username))
    safe_password = quote_plus(unquote_plus(password))

    return urlunsplit(
        (parsed.scheme, f"{safe_user}:{safe_password}@{hostinfo}", parsed.path, parsed.query, parsed.fragment)
    )


def _get_translations_collection():
    global _client, _translations_collection

    if _translations_collection is not None:
        return _translations_collection

    mongo_uri = os.getenv("MONGO_DB")
    if not mongo_uri:
        return None

    normalized_mongo_uri = _normalize_mongo_uri(mongo_uri)

    mongo_client_kwargs = {
        "appname": APP_NAME,
        "serverSelectionTimeoutMS": 12000,
        "connectTimeoutMS": 12000,
        "socketTimeoutMS": 12000,
        "retryWrites": True,
        "tls": True,
        "tlsCAFile": certifi.where(),
    }

    # Optional local troubleshooting fallback. Keep disabled by default.
    if _env_flag("MONGO_TLS_ALLOW_INVALID_CERTS", default=False):
        mongo_client_kwargs["tlsAllowInvalidCertificates"] = True

    _client = MongoClient(
        normalized_mongo_uri,
        **mongo_client_kwargs,
    )

    try:
        db = _client.get_default_database()
    except ConfigurationError:
        db = None

    if db is None:
        db = _client.get_database(DEFAULT_DB_NAME)

    # Force early connectivity/auth validation so submit errors are explicit.
    _client.admin.command("ping")

    _translations_collection = db.get_collection(DEFAULT_COLLECTION)
    return _translations_collection


def save_translation_record(
    en_text: str,
    hindi_options: list[str],
    selected: str,
    selected_index: int | None = None,
):
    try:
        translations_collection = _get_translations_collection()
        if translations_collection is None:
            raise RuntimeError(
                "MongoDB is not configured. Set MONGO_DB in .env to enable saving."
            )

        record = {
            "english_text": en_text,
            "hindi_options": hindi_options,
            "selected_hindi_text": selected,
            "selected_option_index": selected_index,
            "total_options": len(hindi_options),
            "created_at": datetime.utcnow(),
        }
        result = translations_collection.insert_one(record)
        return str(result.inserted_id)
    except PyMongoError as error:
        error_text = str(error)
        if "SSL handshake failed" in error_text or "tlsv1 alert" in error_text.lower():
            raise RuntimeError(
                "MongoDB TLS handshake failed. Try: "
                "1) update certifi/pymongo, "
                "2) check system date/time, "
                "3) if your network intercepts TLS, set MONGO_TLS_ALLOW_INVALID_CERTS=true for local testing."
            ) from error
        raise RuntimeError(f"MongoDB write failed: {error}") from error




