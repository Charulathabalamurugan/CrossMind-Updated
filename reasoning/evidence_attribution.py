import re
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("crossmind.evidence_attribution")

class EvidenceAttributor:
    def __init__(self):
        self.stop_words = {
            "a", "an", "and", "are", "between", "for", "find", "how", "in",
            "is", "of", "the", "to", "what", "with", "that", "this", "it",
        }

    def attribute(self, hypothesis_text: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        sentences = re.split(r'(?<=[.!?])\s+', hypothesis_text)
        attributions = []
        overall_confidence = 0.0
        
        query_terms = {
            term.lower() for term in re.findall(r'[\w\'-]+', hypothesis_text)
            if len(term) > 3 and term.lower() not in self.stop_words
        }
        
        evidence_pool = []
        for ev in evidence:
            payload = ev.get("payload", {})
            content = payload.get("content", "")
            title = payload.get("title", "")
            ev_terms = {t.lower() for t in re.findall(r'[\w\'-]+', f"{title} {content}") if len(t) > 3 and t.lower() not in self.stop_words}
            overlap = len(query_terms & ev_terms)
            evidence_pool.append({
                "id": ev.get("id"),
                "title": title,
                "domain": payload.get("domain", "general"),
                "support_score": round(float(ev.get("score", 0.0)), 4),
                "matched_terms": list(query_terms & ev_terms),
                "overlap_count": overlap,
                "passage": content[:300] if content else "",
                "citation": payload.get("citation", ""),
            })
        
        evidence_pool.sort(key=lambda x: (x["overlap_count"], x["support_score"]), reverse=True)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            sent_terms = {t.lower() for t in re.findall(r'[\w\'-]+', sentence) if len(t) > 3 and t.lower() not in self.stop_words}
            best_ev = None
            best_score = 0
            for ev in evidence_pool:
                overlap = len(sent_terms & set(ev["matched_terms"]))
                if overlap > best_score:
                    best_score = overlap
                    best_ev = ev
            attributions.append({
                "claim": sentence.strip(),
                "attributed_evidence_id": best_ev["id"] if best_ev else None,
                "attribution_confidence": round(min(1.0, best_score / max(len(sent_terms), 1)), 3) if sent_terms else 0.0,
                "matched_terms": best_ev["matched_terms"][:5] if best_ev else [],
                "evidence_domain": best_ev["domain"] if best_ev else "unattributed",
            })
        
        coverage = sum(1 for a in attributions if a["attributed_evidence_id"]) / max(len(attributions), 1)
        overall_confidence = round(coverage, 3)
        
        return {
            "attributions": attributions,
            "overall_attribution_coverage": overall_confidence,
            "total_claims": len(attributions),
            "supported_claims": sum(1 for a in attributions if a["attributed_evidence_id"]),
            "evidence_pool_size": len(evidence),
        }

_attributor_instance: Optional[EvidenceAttributor] = None

def get_evidence_attributor() -> EvidenceAttributor:
    global _attributor_instance
    if _attributor_instance is None:
        _attributor_instance = EvidenceAttributor()
    return _attributor_instance
