#!/usr/bin/env -S uv run --no-project

# /// script
# dependencies = [
#   "pgmpy",
# ]
# ///

from pgmpy.examples import load_model
df = load_model("asia").simulate(n_samples=1000)
df.to_csv("asia.csv", index=False) 
