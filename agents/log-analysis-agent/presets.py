# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""TheRock CI workflow presets — extra grep patterns and workflow name mapping."""

from __future__ import annotations

from dataclasses import dataclass

PRESET_NAMES = (
    "therock_multi_arch",
    "therock_install",
    "therock_pytorch",
    "therock_unit_tests",
    "custom",
)

CATEGORY_PRESETS = (
    "therock_multi_arch",
    "therock_pytorch",
    "therock_install",
    "therock_unit_tests",
)


@dataclass(frozen=True)
class WorkflowPreset:
    name: str
    label: str
    hint: str
    extra_patterns: tuple[str, ...]


WORKFLOW_PRESETS: dict[str, WorkflowPreset] = {
    "therock_multi_arch": WorkflowPreset(
        name="therock_multi_arch",
        label="TheRock — Multi-Arch CI",
        hint="Paste log from ROCm/TheRock Multi-Arch CI failed job",
        extra_patterns=(
            "therock",
            "ninja",
            "cmake",
            "gfx",
            "hipErrorOutOfMemory",
            "rocsparse",
            "ctest",
            "FAILED",
        ),
    ),
    "therock_install": WorkflowPreset(
        name="therock_install",
        label="TheRock — Native Package Install",
        hint="Install test logs (ubuntu2404, dpkg/rpm)",
        extra_patterns=("apt", "dpkg", "install test", "therock"),
    ),
    "therock_pytorch": WorkflowPreset(
        name="therock_pytorch",
        label="TheRock — PyTorch Wheels",
        hint="PyTorch full suite / GPU wheel test logs",
        extra_patterns=("pytorch", "rocm", "gfx94", "hip"),
    ),
    "therock_unit_tests": WorkflowPreset(
        name="therock_unit_tests",
        label="TheRock — Unit Tests / ctest",
        hint="Unit Tests or ctest shard output",
        extra_patterns=("ctest", "assertion", "not ok"),
    ),
    "custom": WorkflowPreset(
        name="custom",
        label="Custom / Other CI",
        hint="Any CI or application log",
        extra_patterns=(),
    ),
}

# Workflows monitored by fork reactive trigger (must match workflow `name:` in YAML).
MONITORED_WORKFLOW_NAMES: tuple[str, ...] = (
    "Multi-Arch CI",
    "Multi-Arch CI - Windows",
    "Multi-Arch CI ASAN",
    "Unit Tests",
    "Test Native Linux Packages Install",
    "Test PyTorch Wheels",
    "Test PyTorch Wheels (Full Suite)",
)


def get_preset(name: str | None) -> WorkflowPreset:
    key = (name or "custom").strip().lower()
    if key not in WORKFLOW_PRESETS:
        valid = ", ".join(PRESET_NAMES)
        raise ValueError(f"Unknown preset '{name}'. Choose one of: {valid}")
    return WORKFLOW_PRESETS[key]


def workflow_name_to_preset(name: str, path: str | None = None) -> str:
    """Classify a GitHub workflow run into a preset (ported from ARVIL workflow-map.ts)."""
    n = f"{name} {path or ''}".lower()

    if "pytorch" in n or "wheel" in n:
        return "therock_pytorch"
    if (
        "multi-arch" in n
        or "multi_arch" in n
        or "multiarch" in n
        or ("asan" in n and "pytorch" not in n)
    ):
        return "therock_multi_arch"
    if (
        "unit test" in n
        or "unit_test" in n
        or "ctest" in n
        or "test_component" in n
        or "component.yml" in n
    ):
        return "therock_unit_tests"
    if (
        "install" in n
        or "native linux" in n
        or "native_linux" in n
        or "package install" in n
        or "dpkg" in n
        or "rpm" in n
    ):
        return "therock_install"
    if "test_artifacts" in n or "test artifacts" in n:
        if "pytorch" in n or "wheel" in n or "torch" in n:
            return "therock_pytorch"
        if (
            "multi-arch" in n
            or "multi_arch" in n
            or "asan" in n
            or "gfx" in n
        ):
            return "therock_multi_arch"
    return "custom"


def preset_matches_workflow_name(
    preset: str,
    workflow_name: str,
    workflow_path: str | None = None,
) -> bool:
    if preset == "custom":
        return True
    return workflow_name_to_preset(workflow_name, workflow_path) == preset
