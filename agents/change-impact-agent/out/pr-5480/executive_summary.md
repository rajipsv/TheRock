# Change Impact Executive Summary

**Range:** `42cdfac6` → `bdaf2213`
**Severity:** MEDIUM (blast radius 45/100)

## What changed

### Changed files (git diff)
- `third-party/openmpi/CMakeLists.txt`

## Blast radius

- Changes detected with moderate default scoring

**Affected build stages:** none
**Rollout:** Canary gfx family + test_filter:standard (no test:* inferred — set GITHUB_TOKEN and use --full-manifest)

## CI recommendations (assistant — apply labels manually)

- **test_type:** `quick` — Third-party packaging/config change — quick sanity per test_filtering.md
- **Suggested PR labels:** test_filter:quick
- **Suggested test suites:** 
- _TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless test:* labels are set. These labels narrow scope to recommended suites. Recommendations are test_matrix jobs + test_filter depth — not individual ctest cases._

## Topology warnings

_Deterministic gaps between changed files and BUILD_TOPOLOGY.toml — review at PR time._

- No topology gaps detected for this change range.
