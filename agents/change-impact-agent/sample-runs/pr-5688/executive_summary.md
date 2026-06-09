# Change Impact Executive Summary

**Range:** `31b738c3` → `e5eed222`
**Severity:** MEDIUM-HIGH (blast radius 70/100)

## What changed

### Changed files (git diff)
- `build_tools/github_actions/fetch_test_configurations.py`
- `ml-libs/artifact-hipdnn.toml`

### Content insights (CI / packaging)
- Test matrix jobs removed or commented out: hipdnn_python_bindings
- Artifact exclude patterns added: share/hipdnn/python/**

## Blast radius

- ML library / hipDNN path change detected
- Test matrix jobs removed or commented out: hipdnn_python_bindings
- Artifact exclude patterns added: share/hipdnn/python/**

**Affected build stages:** math-libs
**Rollout:** Canary gfx family + component-specific test labels

## CI recommendations (assistant — apply labels manually)

- **test_type:** `quick` — CI config change with test jobs disabled — quick sanity per test_filtering.md
- **Suggested PR labels:** test:hipdnn, test_filter:quick
- **Suggested test suites:** hipdnn
- **Disabled CI jobs (do not label):** hipdnn_python_bindings
- _TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless test:* labels are set. These labels narrow scope to recommended suites. Recommendations are test_matrix jobs + test_filter depth — not individual ctest cases. Do not label disabled jobs: hipdnn_python_bindings._
