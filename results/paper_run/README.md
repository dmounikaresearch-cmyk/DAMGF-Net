# Results of the run reported in the paper

These files were produced by `src/damgf_experiments.py` on Google Colab (Tesla T4) and are the
source of Tables 10-14 and Sections 5.6-5.9 of the paper.

| File | Content |
|---|---|
| `RESULTS_FOR_PAPER.md` | all tables in readable form |
| `grid.json` | every variant x pairing x seed: test metrics, confusion matrix, training history, test-time controls, adjacency diagnostics |
| `loso.json` | leave-one-source-out results |
| `efficiency.json` | parameters, GFLOPs, latency and memory |
| `text_pool_sizes.json` | size of each keyword post pool |
| `run_log.txt` | console log of the run |
| `fig_examples_per_class.png` | example images per class and source (Fig. 2) |
