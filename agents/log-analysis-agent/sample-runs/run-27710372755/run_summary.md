# Log Analysis Run Summary

**Run:** [27710372755](https://github.com/ROCm/TheRock/actions/runs/27710372755)
**Repo:** ROCm/TheRock
**Jobs analyzed:** 8 of 8 failed/cancelled (71 total)
**Total errors:** 88

## Linux::release / Build Multi-Arch Stages / comm-libs / Stage - Comm Libs
- Job `81975007413` | conclusion: `cancelled` | errors: 4
  - L765: 2026-06-17T18:56:41.2935176Z         ===================          Compoments check [38;2;6;161;60m15[0m Passed, [38;2
  - L1534: 2026-06-17T20:36:48.3343455Z ##[error]The operation was canceled.
  - L765: 2026-06-17T18:56:41.2935176Z         ===================          Compoments check [38;2;6;161;60m15[0m Passed, [38;2

## Linux::release / Build Multi-Arch Stages / math-libs (gfx94X-dcgpu, gfx942, linux-gfx942-1gpu-core42-ossci-rocm, false) / Stage - Math Libs (gfx94X-dcgpu)
- Job `81975007686` | conclusion: `cancelled` | errors: 4
  - L761: 2026-06-17T18:57:34.9106303Z         ===================          Compoments check [38;2;6;161;60m15[0m Passed, [38;2
  - L1952: 2026-06-17T20:36:48.5684534Z ##[error]The operation was canceled.
  - L761: 2026-06-17T18:57:34.9106303Z         ===================          Compoments check [38;2;6;161;60m15[0m Passed, [38;2

## Windows::release / Test gfx110X-all / Test rocblas / Test rocblas (shard 1/1) (gfx110X-all)
- Job `81992436676` | conclusion: `cancelled` | errors: 4
  - L1781: 2026-06-17T20:17:41.1397989Z         ===================          Compoments check [38;2;6;161;60m12[0m Passed, [38;2
  - L3741: 2026-06-17T20:36:56.0158814Z ##[error]The operation was canceled.
  - L1781: 2026-06-17T20:17:41.1397989Z         ===================          Compoments check [38;2;6;161;60m12[0m Passed, [38;2

## Windows::release / Test gfx110X-all / Test rocsparse / Test rocsparse (shard 1/1) (gfx110X-all)
- Job `81992436725` | conclusion: `failure` | errors: 50
  - L1769: 2026-06-17T20:17:42.3517629Z         ===================          Compoments check [38;2;6;161;60m12[0m Passed, [38;2
  - L26537: 2026-06-17T20:19:36.4705507Z 1: unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
  - L26551: 2026-06-17T20:19:37.4793080Z 1: C:/home/runner/_work/TheRock/TheRock/rocm-libraries/projects/rocsparse/clients/tests/tes

## Windows::release / Test gfx110X-all / Test libhipcxx_hipcc / Test libhipcxx_hipcc (shard 1/1) (gfx110X-all)
- Job `81992436788` | conclusion: `failure` | errors: 7
  - L1721: 2026-06-17T20:18:32.4668035Z         ===================          Compoments check [38;2;6;161;60m12[0m Passed, [38;2
  - L1997: 2026-06-17T20:18:32.9512404Z Error executing offload-arch: Command '['./build/lib/llvm/bin/offload-arch.exe']' returned 
  - L2063: 2026-06-17T20:18:41.8515155Z CMake Error at C:/Program Files/CMake/share/cmake-3.31/Modules/CMakeTestHIPCompiler.cmake:7

## Windows::release / Test gfx110X-all / Test hipsparse / Test hipsparse (shard 1/1) (gfx110X-all)
- Job `81992436901` | conclusion: `failure` | errors: 11
  - L1769: 2026-06-17T20:17:42.6506429Z         ===================          Compoments check [38;2;6;161;60m12[0m Passed, [38;2
  - L92545: 2026-06-17T20:18:59.7142035Z 1: C:/home/runner/_work/TheRock/TheRock/rocm-libraries/projects/hipsparse/clients/common/ar
  - L92559: 2026-06-17T20:18:59.7146799Z 1: C:/home/runner/_work/TheRock/TheRock/rocm-libraries/projects/hipsparse/clients/include\t

## Windows::release / Build PyTorch (fat + split) / Build PyTorch | multi-arch-release | torch release/2.10 | py3.12
- Job `81995075999` | conclusion: `cancelled` | errors: 6
  - L1519: 2026-06-17T20:32:01.8778318Z -a----          4/8/2025  12:57 PM          23451 urllib.error.html
  - L12889: 2026-06-17T20:32:06.8394907Z -a----          4/8/2025  12:56 PM           2489 error.py
  - L12915: 2026-06-17T20:32:06.8489420Z -a----         6/17/2026   8:31 PM           3678 error.cpython-312.pyc

## CI Summary
- Job `81996346805` | conclusion: `failure` | errors: 2
  - L181: 2026-06-17T20:38:13.2811474Z ##[error]Process completed with exit code 1.
  - L181: 2026-06-17T20:38:13.2811474Z ##[error]Process completed with exit code 1.
