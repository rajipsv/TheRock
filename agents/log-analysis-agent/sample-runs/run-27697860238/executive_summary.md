# Log Analysis Executive Summary

**Log:** `C:\Users\Rajeswari\.gemini\antigravity\scratch\TheRock\TheRock\agents\log-analysis-agent\sample-runs\run-27697860238\job-81925995968.log`
**GitHub run:** [27697860238](https://github.com/ROCm/TheRock/actions/runs/27697860238) (ROCm/TheRock, job: Linux::release / Build Multi-Arch Stages / compiler-runtime / Stage - Compiler Runtime)
**Mode:** tool_only
**Preset:** therock_multi_arch
**Errors found:** 4

## Summary
Tool-only qualification pass found 4 highlighted error lines. Stats: path=C:\Users\Rajeswari\.gemini\antigravity\scratch\TheRock\TheRock\agents\log-analysis-agent\sample-runs\run-27697860238\job-81925995968.log; lines=3318; bytes=503589; github_error=0; keyword_hits: ERROR=4, EXCEPTION=0, CRITICAL=0, FATAL=1, WARNING=36

## Top errors
- **Line 767** (HIGH): 2026-06-17T14:53:49.5878371Z         ===================          Compoments check [38;2;6;161;60m15[0m Passed, [38;2
  - Recommendation: Verify rocm-smi, matching driver/stack, GPU passthrough, reinstall ROCm packages
- **Line 1999** (HIGH): 2026-06-17T15:06:39.2651104Z [amd-comgr build-test 2]  REASON: ERROR
  - Recommendation: Verify rocm-smi, matching driver/stack, GPU passthrough, reinstall ROCm packages
- **Line 2012** (HIGH): 2026-06-17T15:06:39.3077212Z CMake Error at /__w/TheRock/TheRock/build/compiler/amd-comgr/prefix/build-test-runner.cmake
  - Recommendation: Verify rocm-smi, matching driver/stack, GPU passthrough, reinstall ROCm packages
- **Line 767** (CRITICAL): 2026-06-17T14:53:49.5878371Z         ===================          Compoments check [38;2;6;161;60m15[0m Passed, [38;2
  - Recommendation: Verify rocm-smi, matching driver/stack, GPU passthrough, reinstall ROCm packages

## Knowledge base matches
- `2026-06-17T14:53:49.5878371Z         ===================          Compoments che` → ROCm HIP API failure (GPU/Driver, score=11.5)
- `2026-06-17T15:06:39.2651104Z [amd-comgr build-test 2]  REASON: ERROR` → ROCm HIP API failure (GPU/Driver, score=11.5)
- `2026-06-17T15:06:39.3077212Z CMake Error at /__w/TheRock/TheRock/build/compiler/` → ROCm HIP API failure (GPU/Driver, score=11.5)
