"""Trace what the pre-filter detects."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning.symbolic_filter import SymbolicPreFilter
pf = SymbolicPreFilter()
q = "How does silicon improve lithium battery anodes?"
fm = pf.process(q)
print("Detected domains:", fm.get("detected_domains"))
print("Entities:", fm.get("extracted_entities"))
print("Full filter metadata keys:", list(fm.keys()))
print()
print("Full metadata:")
for k, v in fm.items():
    if k != "extracted_entities":
        print(f"  {k}: {v}")
