# Change Impact Executive Summary

**Range:** `9100ec68` → `adbd8440`
**Severity:** MEDIUM-HIGH (blast radius 70/100)

## What changed

### Changed files (git diff)
- `rocm-libraries`

### Manifest / submodule changes
- `rocm-libraries` (changed, superrepo)
- `rocm-libraries/rpp` (added, component)
- `rocm-libraries/hipdnn` (changed, component)
- `rocm-libraries/hipsolver` (changed, component)
- `rocm-libraries/rocsparse` (changed, component)
- `rocm-libraries/ctest` (changed, component)
- `rocm-libraries/composablekernel` (changed, component)
- `rocm-libraries/hiprand` (changed, component)
- `rocm-libraries/stinkytofu` (changed, component)
- `rocm-libraries/tensile` (changed, component)
- `rocm-libraries/hipsparselt` (changed, component)
- `rocm-libraries/miopen` (changed, component)
- `rocm-libraries/rocsolver` (changed, component)
- `rocm-libraries/hipblaslt` (changed, component)

### Changed superrepo components
- `composablekernel`: 0 file(s), 7 commit(s) in range
- `ctest`: 0 file(s), 1 commit(s) in range
- `hipblaslt`: 0 file(s), 13 commit(s) in range
- `hipdnn`: 0 file(s), 5 commit(s) in range
- `hiprand`: 0 file(s), 1 commit(s) in range
- `hipsolver`: 0 file(s), 3 commit(s) in range
- `hipsparselt`: 0 file(s), 2 commit(s) in range
- `miopen`: 0 file(s), 5 commit(s) in range
- `rocsolver`: 0 file(s), 1 commit(s) in range
- `rocsparse`: 0 file(s), 1 commit(s) in range
- `rpp`: 0 file(s), 1 commit(s) in range
- `stinkytofu`: 0 file(s), 4 commit(s) in range
- `tensile`: 0 file(s), 2 commit(s) in range

## Blast radius

- ML library / hipDNN path change detected

**Affected build stages:** math-libs
**Rollout:** Canary gfx family + component-specific test labels

## CI recommendations (assistant — apply labels manually)

- **test_type:** `standard` — Submodule or superrepo component change (matches TheRock CI policy)
- **Suggested PR labels:** test:hipblaslt, test:hipdnn, test:hiprand, test:hipsolver, test:hipsparselt, test:miopen, test:rocsolver, test:rocsparse, test_filter:standard
- **Suggested test suites:** hipblaslt, hipdnn, hiprand, hipsolver, hipsparselt, miopen, rocsolver, rocsparse
- _TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless test:* labels are set. These labels narrow scope to recommended suites. Recommendations are test_matrix jobs + test_filter depth — not individual ctest cases._

## Topology warnings

_Deterministic gaps between changed files and BUILD_TOPOLOGY.toml — review at PR time._

- No topology gaps detected for this change range.
