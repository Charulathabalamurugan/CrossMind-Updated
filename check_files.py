import os
base = 'E:\\CrossMind-Updated'
for f in ['reasoning/decision_tree.py', 'reasoning/scallop.py', 'reasoning/deforest_vis.py']:
    full = os.path.join(base, f)
    status = 'EXISTS' if os.path.exists(full) else 'MISSING'
    print(status + ': ' + f)