import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("crossmind.conflict_detector")

class ConflictDetector:
    """
    ConflictDetector scans retrieved evidence documents for scientific contradictions
    or opposing claims (e.g. increases vs decreases, activates vs inhibits, toxic vs protective).
    """
    
    # Pairs of contrasting action words
    CONFLICT_PAIRS = [
        (r"\bincrease(s|d)?\b", r"\bdecrease(s|d)?\b"),
        (r"\bactivate(s|d)?\b", r"\binhibit(s|d)?\b"),
        (r"\bstimulate(s|d)?\b", r"\bsuppress(es|ed)?\b"),
        (r"\bupregulate(s|d)?\b", r"\bdownregulate(s|d)?\b"),
        (r"\bprotective\b", r"\btoxic\b"),
        (r"\benhance(s|d)?\b", r"\breduce(s|d)?\b"),
        (r"\bpositive\b", r"\bnegative\b"),
    ]
    
    @classmethod
    def detect_conflicts(cls, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts = []
        if len(evidence) < 2:
            return conflicts
            
        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                doc1 = evidence[i]
                doc2 = evidence[j]
                
                id1 = doc1.get("id", f"doc_{i}")
                id2 = doc2.get("id", f"doc_{j}")
                
                title1 = doc1.get("payload", {}).get("title", "Doc A")
                title2 = doc2.get("payload", {}).get("title", "Doc B")
                
                content1 = doc1.get("payload", {}).get("content", "").lower()
                content2 = doc2.get("payload", {}).get("content", "").lower()
                
                if not content1 or not content2:
                    continue
                
                # Check overlapping terms (substantive words) to see if they discuss the same topics
                words1 = set(re.findall(r'\b\w{4,}\b', content1))
                words2 = set(re.findall(r'\b\w{4,}\b', content2))
                overlap = words1.intersection(words2)
                
                # Filter out extremely common structural words
                stop_words = {"this", "that", "with", "from", "their", "have", "were", "about", "study", "results", "effect", "treatment"}
                overlap = overlap - stop_words
                
                # If they discuss the same entity/concept (overlap > 2), check for contrasting actions
                if len(overlap) >= 2:
                    for pattern1, pattern2 in cls.CONFLICT_PAIRS:
                        match1_p1 = re.search(pattern1, content1)
                        match1_p2 = re.search(pattern2, content1)
                        match2_p1 = re.search(pattern1, content2)
                        match2_p2 = re.search(pattern2, content2)
                        
                        # Case 1: Doc1 has pattern1, Doc2 has pattern2
                        # Case 2: Doc1 has pattern2, Doc2 has pattern1
                        if (match1_p1 and match2_p2) or (match1_p2 and match2_p1):
                            kw = list(overlap)[:3]
                            conflicts.append({
                                "source_id_1": id1,
                                "source_title_1": title1,
                                "source_id_2": id2,
                                "source_title_2": title2,
                                "overlapping_entities": kw,
                                "conflict_type": f"Opposing claims detected around: {', '.join(kw)}",
                                "details": f"Doc A mentions matching pattern '{pattern1 if match1_p1 else pattern2}' whereas Doc B mentions opposing pattern '{pattern2 if match2_p2 else pattern1}'."
                            })
                            # Break to prevent logging multiple conflicts for the same document pair
                            break
                            
        return conflicts

def get_conflict_detector() -> ConflictDetector:
    return ConflictDetector()
