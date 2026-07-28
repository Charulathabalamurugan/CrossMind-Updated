import logging
from typing import Dict, Any, List, Optional
from config import settings

logger = logging.getLogger("crossmind.opa")

class OPAEnforcer:
    def __init__(self):
        self.enabled = settings.OPA_ENABLED
        self._cache: Dict[str, bool] = {}
        self._cache_ttl = 300

    def check_access(self, user_role: str, resource_domain: str, action: str = "read") -> Dict[str, Any]:
        cache_key = f"{user_role}:{resource_domain}:{action}"
        if cache_key in self._cache:
            return {"allowed": self._cache[cache_key], "method": "cache"}
        
        if not self.enabled:
            result = self._default_policy(user_role, resource_domain, action)
            self._cache[cache_key] = result["allowed"]
            return {**result, "method": "default_policy"}

        try:
            import requests
            policy_url = f"{settings.OPA_URL}/v1/data/crossmind/authz/allow" if hasattr(settings, "OPA_URL") else None
            if not policy_url:
                result = self._default_policy(user_role, resource_domain, action)
                self._cache[cache_key] = result["allowed"]
                return {**result, "method": "opa_unavailable"}
            
            resp = requests.post(
                policy_url,
                json={"input": {"user_role": user_role, "resource_domain": resource_domain, "action": action}},
                timeout=2.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                allowed = data.get("result", {}).get("allow", False)
                self._cache[cache_key] = allowed
                return {"allowed": allowed, "method": "opa"}
            else:
                result = self._default_policy(user_role, resource_domain, action)
                self._cache[cache_key] = result["allowed"]
                return {**result, "method": "opa_error_fallback"}
        except Exception as exc:
            logger.warning(f"OPA check failed: {exc}. Using default policy.")
            result = self._default_policy(user_role, resource_domain, action)
            self._cache[cache_key] = result["allowed"]
            return {**result, "method": "error_fallback"}

    def _default_policy(self, user_role: str, resource_domain: str, action: str) -> Dict[str, Any]:
        allowed = True
        if user_role == "public" and resource_domain in ("pharmacology", "clinical") and action == "write":
            allowed = False
        return {"allowed": allowed, "user_role": user_role, "resource_domain": resource_domain, "action": action}

    def invalidate_cache(self):
        self._cache.clear()

_opa_instance = None

def get_opa_enforcer() -> OPAEnforcer:
    global _opa_instance
    if _opa_instance is None:
        _opa_instance = OPAEnforcer()
    return _opa_instance