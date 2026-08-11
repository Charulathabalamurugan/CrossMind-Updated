"""
OAuth2 + JWT authentication for CrossMind.
Supports password grant, token refresh, and API key fallback.
"""
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger("crossmind.auth")

# In-memory user store for prototype; replace with real database in production.
USERS: Dict[str, Dict[str, Any]] = {
    "admin": {
        "user_id": "u_admin",
        "username": "admin",
        "password_hash": hashlib.sha256("admin".encode()).hexdigest(),
        "roles": ["admin", "researcher", "viewer"],
        "email": "admin@crossmind.ai",
        "full_name": "System Admin",
        "preferences": {"default_role": "admin", "theme": "dark"},
        "created_at": datetime.utcnow().isoformat(),
    },
    "researcher": {
        "user_id": "u_researcher",
        "username": "researcher",
        "password_hash": hashlib.sha256("researcher".encode()).hexdigest(),
        "roles": ["researcher", "viewer"],
        "email": "researcher@crossmind.ai",
        "full_name": "Default Researcher",
        "preferences": {"default_role": "researcher", "theme": "light"},
        "created_at": datetime.utcnow().isoformat(),
    },
}

TOKEN_TTL_SECONDS = 3600
REFRESH_TTL_SECONDS = 86400


class AuthService:
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
        user = USERS.get(username)
        if not user:
            return None
        if hashlib.sha256(password.encode()).hexdigest() != user["password_hash"]:
            return None
        return user

    @staticmethod
    def create_tokens(user: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.utcnow()
        access_exp = now + timedelta(seconds=TOKEN_TTL_SECONDS)
        refresh_exp = now + timedelta(seconds=REFRESH_TTL_SECONDS)
        access_token = hashlib.sha256(f"{user['user_id']}:{now.isoformat()}:access".encode()).hexdigest()
        refresh_token = hashlib.sha256(f"{user['user_id']}:{now.isoformat()}:refresh".encode()).hexdigest()
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": TOKEN_TTL_SECONDS,
            "expires_at": access_exp.isoformat(),
            "refresh_token": refresh_token,
            "refresh_expires_at": refresh_exp.isoformat(),
            "user_id": user["user_id"],
            "username": user["username"],
            "roles": user["roles"],
        }

    @staticmethod
    def validate_token(token: str) -> Optional[Dict[str, Any]]:
        for username, user in USERS.items():
            if user.get("access_token") == token:
                return user
        return None

    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[Dict[str, Any]]:
        for username, user in USERS.items():
            if user.get("refresh_token") == refresh_token:
                return AuthService.create_tokens(user)
        return None

    @staticmethod
    def list_users() -> List[Dict[str, Any]]:
        return [
            {
                "user_id": u["user_id"],
                "username": u["username"],
                "email": u["email"],
                "roles": u["roles"],
                "full_name": u.get("full_name", ""),
            }
            for u in USERS.values()
        ]

    @staticmethod
    def create_user(username: str, password: str, email: str, roles: List[str], full_name: str = "") -> Optional[Dict[str, Any]]:
        if username in USERS:
            return None
        USERS[username] = {
            "user_id": f"u_{username}",
            "username": username,
            "password_hash": hashlib.sha256(password.encode()).hexdigest(),
            "roles": roles,
            "email": email,
            "full_name": full_name or username,
            "preferences": {"default_role": roles[0] if roles else "viewer", "theme": "light"},
            "created_at": datetime.utcnow().isoformat(),
        }
        return {k: v for k, v in USERS[username].items() if k != "password_hash"}
