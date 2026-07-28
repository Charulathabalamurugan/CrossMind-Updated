import logging
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.deepseek")

class DeepSeekR1Agent:
    def __init__(self):
        self.enabled = settings.DEEPSEEK_R1_ENABLED
        self.vllm_enabled = settings.vLLM_ENABLED
        self.api_base = settings.DEEPSEEK_R1_API_BASE
        self.model_name = settings.DEEPSEEK_R1_MODEL
        self.quantization = settings.DEEPSEEK_R1_QUANTIZATION

    def generate(self, prompt: str, evidence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.enabled or not self.vllm_enabled:
            return {"response": "", "model": "disabled", "token_count": 0}
        
        try:
            import httpx
            messages = [
                {"role": "system", "content": "You are DeepSeek-R1-Distill-Qwen-14B, a reasoning model for scientific hypothesis generation. Provide grounded, evidence-based responses with confidence scores."},
                {"role": "user", "content": prompt},
            ]
            if evidence:
                evidence_text = "\n".join(
                    f"- [{ev.get('id','')}] {ev.get('payload',{}).get('title','')}: {ev.get('payload',{}).get('content','')[:500]}"
                    for ev in evidence[:10]
                )
                messages[1]["content"] += f"\n\nEvidence:\n{evidence_text}"

            response = httpx.post(
                f"{self.api_base}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "extra_body": {"quantization": self.quantization},
                },
                timeout=15.0,
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "response": content,
                    "model": f"deepseek-r1-distill ({self.quantization})",
                    "token_count": usage.get("total_tokens", 0),
                    "model_name": self.model_name,
                }
        except Exception as exc:
            logger.warning(f"DeepSeek-R1 generation failed: {exc}")

        return {"response": "", "model": "deepseek-r1-unavailable", "token_count": 0}

deepseek_agent = None

def get_deepseek_agent() -> DeepSeekR1Agent:
    global deepseek_agent
    if deepseek_agent is None:
        deepseek_agent = DeepSeekR1Agent()
    return deepseek_agent