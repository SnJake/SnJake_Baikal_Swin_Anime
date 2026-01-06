import json
from collections import defaultdict

acc = defaultdict(list)
with open(r"G:\Programs\UpscalerAnime\runs\perceptual_convnextv2\train_log.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        acc[row["epoch"]].append(row["loss"])

for e in sorted(acc):
    avg = sum(acc[e]) / len(acc[e])
    print(e, avg)
