import logging
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.deepseek")

class ZAYA1_8BAgent:
    def __init__(self):
        self.enabled = settings.ZAYA1_8B_VLLM_ENABLED
        self.vllm_enabled = settings.ZAYA1_8B_VLLM_ENABLED
        self.api_base = settings.ZAYA1_8B_API_BASE
        self.model_name = settings.ZAYA1_8B_MODEL_NAME
        self.quantization = settings.ZAYA1_8B_QUANTIZATION

    def generate(self, prompt: str, evidence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled or not self.vllm_enabled:
            return {"response": "", "model": "disabled", "token_count": 0}
        
        try:
            import httpx
            messages = [
                {"role": "system", "content": "You are ZAYA1-8B, an 8.4B MoE model with 760M active parameters for scientific discovery. Provide grounded, evidence-based responses with confidence scores."},
                {"role": "user", "content": prompt},
            ]
            if evidence:
                evidence_text = "\n".join(
                    f"- [{ev.get('id','')}] {ev.get('payload',{}).get('title','')}: {ev.get('payload',{}).get('content','')[:500]}"
                    for ev in evidence[:10]
                )
                messages[1]["content"] += f"\n\nEvidence:\n{evidence_text}"

            response = httpx.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 131072,
                },
                timeout=15.0,
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "response": content,
                    "model": f"ZAYA1-8B ({self.quantization})",
                    "token_count": usage.get("total_tokens", 0),
                    "model_name": self.model_name,
                }
        except Exception as exc:
            logger.warning(f"DeepSeek-R1 generation failed: {exc}")

        return {"response": "", "model": "zaya1-8b-unavailable", "token_count": 0}

zaya1_8b_agent = None

def get_zaya1_8b_agent() -> ZAYA1_8BAgent:
    global zaya1_8b_agent
    if zaya1_8b_agent is None:
        zaya1_8b_agent = ZAYA1_8BAgent()
    return zaya1_8b_agent