"""Tests for the public Lange domain boundaries."""

import importlib

import pytest


def test_mesh_domain_exports_worker_and_plugins() -> None:
    """Expose mesh runtime types from the mesh domain root."""
    from lange.mesh import MeshPlugin, MeshRelayPlugin, MeshWorker

    assert MeshWorker.__name__ == "MeshWorker"
    assert MeshPlugin.__name__ == "MeshPlugin"
    assert MeshRelayPlugin.__name__ == "MeshRelayPlugin"


def test_ai_domain_exports_mesh_ai_plugin() -> None:
    """Expose the AI mesh plugin without importing optional backends."""
    from lange.ai import MeshAiPlugin

    assert MeshAiPlugin.__name__ == "MeshAiPlugin"


def test_domain_contracts_are_canonical() -> None:
    """Load contracts through their owning domain packages."""
    from lange.ai.contracts import AiModelConfig
    from lange.mesh.contracts import MeshMessage, MeshRelayRequest

    assert AiModelConfig.__name__ == "AiModelConfig"
    assert MeshMessage.__name__ == "MeshMessage"
    assert MeshRelayRequest.__name__ == "MeshRelayRequest"


@pytest.mark.parametrize("module_name", ["lange.contracts", "lange.mesh.ai"])
def test_legacy_domain_imports_are_removed(module_name: str) -> None:
    """Reject legacy contract and nested-AI import paths."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
