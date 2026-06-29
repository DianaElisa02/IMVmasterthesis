# Current preferred analysis

The canonical empirical specification is the tercile/grouped DiD using the two co-primary hybrid exposure measures:

- `exposure_cov_hybrid`
- `exposure_exp_hybrid`

The preferred adjusted control set is defined centrally in `src/control_specs.py`:

- `head_age_group`
- `head_sex`
- `head_education_group`
- `n_adults_group`
- `n_children_group`

Run the full current pipeline with:

```bash
python run_current_analysis.py
```

Canonical thesis figures are written to `output/thesis_figures/` and canonical current regression outputs are written by `run_binned_did.py`, `run_event_study.py`, `run_late_post_did.py`, and `run_preferred_placebo.py`.

Files under older output paths that refer to `exposure_composite_hybrid`, `event_study_primary`, or the former control set are legacy results and should not be used in the thesis unless they are explicitly being discussed as historical sensitivity checks.
