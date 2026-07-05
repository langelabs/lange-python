"""Shared audit tool type definitions."""

from __future__ import annotations

from typing import Literal

AuditTool = Literal["pnpm", "uv"]

PNPM_AUDIT_TOOL: AuditTool = "pnpm"
UV_AUDIT_TOOL: AuditTool = "uv"
