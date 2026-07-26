The user chose analysis-only (no code), so this is a research deliverable rather than an implementation plan — I'll present the findings directly. I've written the full write-up to /Users/vinicius/.claude/plans/is-it-possible-to-prancy-firefly.md. Summary:

Q1 — Keep only a NetworkX structure in memory? Feasible, with one caveat.

The graphviz.Digraph plays two separate roles:

- Intermediate representation — generate_dot() → to_networkx() is a wasteful round-trip: the dfg dict {(src_label, tgt_label): count} is serialized to DOT and then regex-parsed back into nx.DiGraph (core.py:484-512). You can build the nx.DiGraph + _nodes list directly from dfg, deleting all DOT parsing. Clean win.
- Styling/render carrier — the whole visualizer.py is bound to the stored _dot object: it mutates dot.body strings and renders via Source(dot.source).pipe("png"). To drop graphviz from memory entirely, you'd regenerate the Digraph lazily from the nx graph at render time.

So: networkx-only-in-memory is achievable; it removes the round-trip cheaply but pushes Digraph construction into the visualizer. Three invariants must hold: SHA-1 node ids, weight edge attribute, and the _nodes = sorted([id,label]) list.

Q2 — Replace with igraph's to_graphviz? No.

python-igraph has no to_graphviz() returning a graphviz.Digraph. It only has the lossy Graph.write_dot() (no reliable per-node fillcolor, rankdir, edge penwidth/label) plus Cairo plotting. The visualizer relies on graphviz-specific dot.body/.source/.pipe/.attr behavior that igraph can't supply. igraph is already a dep (pyproject.toml:40) but only as a fast centrality engine in metrics/nodes.py — keep it there. Keep graphviz as the render backend.

Q3 — Properties generate_dot produces (full table in the file)

- Graph: bgcolor, rankdir (config), overlap, fontsize (hardcoded) → re-applied by visualizer layout templates.
- Nodes: id = str(int(sha1(activity),16)) (the system-wide key, used in every metrics module + recomputed in explainer._label_to_node_id); label = escaped activity (id↔label maps, node_lookup); style=filled, fontsize=20; fillcolor (config, mutated by change_node_color/highlight_class_node); shape.
- Edges: source/target id (SHA-1); label = frequency → recovered as float weight, consumed by all metrics + penwidth scaling + Markov calcs in metrics/graph.py; penwidth=1 (overwritten by change_edge_color); fontsize=18.
- Fragile DOT-format couplings a refactor would remove: the label="..." regex and -> parsing in to_networkx, plus dot.body line mutations in utils.py and visualizer.py:2710-2715.

I'm not calling ExitPlanMode since no code implementation was requested — if you later want me to actually do the networkx-only refactor (the recommended direction), say the word and I'll plan the implementation.