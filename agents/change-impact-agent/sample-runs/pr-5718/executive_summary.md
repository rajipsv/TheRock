# Change Impact Executive Summary

**Range:** `9100ec68` → `adbd8440`
**Severity:** MEDIUM (blast radius 45/100)

## What changed

### Changed files (git diff)
- `rocm-libraries`

### Manifest / submodule changes
- `rocm-libraries` (changed, superrepo)

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

- Changes detected with moderate default scoring

**Affected build stages:** math-libs
**Rollout:** Canary gfx family + component-specific test labels

## CI recommendations (assistant — apply labels manually)

- **test_type:** `standard` — Submodule or superrepo component change (matches TheRock CI policy)
- **Suite inference:** from changed superrepo components (file paths or per-directory commits)
- **Suggested PR labels:** test:hipblaslt, test:hipdnn, test:hiprand, test:hipsolver, test:hipsparselt, test:miopen, test:rocsolver, test:rocsparse, test_filter:standard
- **Suggested test suites:** hipblaslt, hipdnn, hiprand, hipsolver, hipsparselt, miopen, rocsolver, rocsparse
- _TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless test:* labels are set. These labels narrow scope to recommended suites. Recommendations are test_matrix jobs + test_filter depth — not individual ctest cases. test:* labels inferred from changed superrepo components (file compare or commit allocation). Use --full-manifest for manifest drill-down._
