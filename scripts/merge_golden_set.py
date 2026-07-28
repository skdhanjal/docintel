import json

with open("golden_set_synthetic.json") as f:
    synthetic = json.load(f)
with open("golden_set_handwritten.json") as f:
    handwritten = json.load(f)

reviewed_synthetic = [g for g in synthetic if g["reviewed"]]
golden = reviewed_synthetic + handwritten
for i, g in enumerate(golden):
    g["id"] = f"q{i:04d}"

with open("golden_set.json", "w") as f:
    json.dump(golden, f, indent=2)

print(f"golden_set.json: {len(golden)} questions")
from collections import Counter
print(Counter(g["archetype"] for g in golden))