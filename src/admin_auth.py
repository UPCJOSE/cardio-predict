"""Credenciales del administrador (JSON local, contraseña hasheada)."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path

from src.config import (
    ADMIN_CREDENTIALS_JSON,
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
)


def _hash_password(password: str, salt: str) -> str:
    """SHA-256(salt + password); suficiente para demo de aula."""
    digest = hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()
    return digest


def _default_credentials() -> dict:
    salt = secrets.token_hex(16)
    return {
        "username": ADMIN_USERNAME,
        "salt": salt,
        "password_hash": _hash_password(ADMIN_PASSWORD, salt),
    }


def load_credentials() -> dict:
    """Carga credenciales; si no existen, crea el archivo con la clave por defecto."""
    path = Path(ADMIN_CREDENTIALS_JSON)
    if not path.exists():
        creds = _default_credentials()
        save_credentials(creds)
        return creds
    return json.loads(path.read_text(encoding="utf-8"))


def save_credentials(creds: dict) -> None:
    path = Path(ADMIN_CREDENTIALS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(creds, indent=2), encoding="utf-8")


def verify_login(username: str, password: str) -> bool:
    creds = load_credentials()
    if username != creds.get("username"):
        return False
    expected = creds.get("password_hash", "")
    salt = creds.get("salt", "")
    got = _hash_password(password, salt)
    return hmac.compare_digest(got, expected)


def change_password(
    current_password: str,
    new_password: str,
    new_username: str | None = None,
) -> tuple[bool, str]:
    """Cambia la contraseña (y opcionalmente el usuario). Exige la clave actual."""
    if not current_password:
        return False, "Indique la contraseña actual."
    if not new_password or len(new_password) < 6:
        return False, "La nueva contraseña debe tener al menos 6 caracteres."
    creds = load_credentials()
    if not verify_login(creds.get("username", ADMIN_USERNAME), current_password):
        return False, "La contraseña actual no es correcta."

    salt = secrets.token_hex(16)
    username = (new_username or creds.get("username") or ADMIN_USERNAME).strip()
    if not username:
        return False, "El usuario no puede quedar vacío."
    save_credentials(
        {
            "username": username,
            "salt": salt,
            "password_hash": _hash_password(new_password, salt),
        }
    )
    return True, "Credenciales actualizadas."


def current_username() -> str:
    return str(load_credentials().get("username", ADMIN_USERNAME))
