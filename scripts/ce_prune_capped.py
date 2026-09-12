"""Delete ce_out/<dom>_<idx>.json for every prompt where any arm hit the token
cap, so that re-running ce_gen.py with a higher MAXNEW regenerates exactly
those prompts (all arms, same seeds).  Run on the pod in /workspace.

  python3 ce_prune_capped.py            # dry run: lists files
  python3 ce_prune_capped.py --delete   # removes them
"""
import json, os, sys

OUT = os.environ.get("OUT", "ce_out")
todo = []
for f in sorted(os.listdir(OUT)):
    if not f.endswith(".json"):
        continue
    recs = json.load(open(os.path.join(OUT, f)))
    if any(r.get("hit_cap") for r in recs):
        todo.append(f)
print(f"{len(todo)} prompt files with a capped arm:")
print("  " + " ".join(todo))
if "--delete" in sys.argv:
    for f in todo:
        os.remove(os.path.join(OUT, f))
    print("deleted")
else:
    print("(dry run; add --delete to remove)")
