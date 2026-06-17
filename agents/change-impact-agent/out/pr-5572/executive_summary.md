# Change Impact Executive Summary

**Range:** `7b7b238e` → `c70f211c`
**Severity:** MEDIUM (blast radius 40/100)

## What changed

### Changed files (git diff)
- `build_tools/github_actions/fetch_test_configurations.py`

### Content insights (CI / packaging)
- GHA wrapper timeout for `miopen`: 60 → 120 minutes (per-test limits still from test_categories.yaml)

### GHA wrapper timeout changes
- `miopen`: 60 → 120 minutes

## Blast radius

- CI test matrix or GitHub Actions configuration changed
- GHA wrapper timeout for `miopen`: 60 → 120 minutes (per-test limits still from test_categories.yaml)

**Affected build stages:** none
**Rollout:** Canary gfx family + component-specific test labels

## CI recommendations (assistant — apply labels manually)

- **test_type:** `quick` — CI or packaging manifest change only (no submodule bump)
- **Suggested PR labels:** test:miopen, test_filter:quick
- **Suggested test suites:** miopen
- _TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless test:* labels are set. These labels narrow scope to recommended suites. Recommendations are test_matrix jobs + test_filter depth — not individual ctest cases._

## Topology warnings

_Deterministic gaps between changed files and BUILD_TOPOLOGY.toml — review at PR time._

- No topology gaps detected for this change range.
