# Source audit

The paper's Method, Experimental Setup, Results, and Appendix were used to select the implementation knobs in this release: entropy-weighted visual-token scoring, full-token refresh, residual merge, learned FFN gates, global channel materialization, Grad×Act head importance, five-fold intersection-and-fill, and Full Token/Random/RVT matched paths.

The public files were copied to a new local work directory from the matching read-only LAVIS 1.0.1 implementation and then cleaned there. The remote source audit confirmed the selected `blip2_opt.py`, Q-Former, OPT, runner, training, and evaluation files and their RVT symbols. The audited checkout did not expose usable Git commit metadata, so the package pins the upstream package version rather than inventing a commit hash. Historical checkpoints, logs, data, machine-specific paths, and orchestration outputs were deliberately excluded. The package therefore contains the algorithm and configuration templates, not a frozen experiment archive.

Users should still fill in local `model.pretrained` and COCO paths. The CPU release checker does not validate a full LAVIS import because that requires the upstream checkout and its optional dependencies. The remote original directory was not modified.
