"""Shared vault note validator: single source of truth for status vocabularies.

Both the prime-os-eval scorer and the vault_validate_hook (the write-time gate)
import from here. Do not duplicate these constants or helpers.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Vendor-deal status vocabulary + tag map
# ---------------------------------------------------------------------------

VALID_DEAL_STATUS = {
    # LIFECYCLE only. Priority lives in the `prio` field (P1/P2/P3), never here.
    # Clean split completed 2026-10-04 (vault 5bf0f09 / vault-ops c64fc09):
    # p1/p2/p3 removed from deal_status.
    "prospect", "loi", "negotiating", "signed", "provisioning",
    "online", "ramping", "lost", "closed",
}

VALID_PRIO = {"P1", "P2", "P3"}

STATUS_TAG_MAP = {
    "prospect": "status/prospect",
    "loi": "status/loi",
    "negotiating": "status/negotiating",
    "signed": "status/signed",
    "provisioning": "status/provisioning",
    "online": "status/online",
    "ramping": "status/ramping",
    "lost": "status/lost",
    "closed": "status/closed",
}

# ---------------------------------------------------------------------------
# Client-deal status vocabulary (current set, not the pending 5-stage funnel)
# ---------------------------------------------------------------------------

# Flip to True ONLY after confirming the archived pipeline_restructure_spec
# still stands, then run the data migration (customer->signed, reclassify
# unknown/expired, fold requested->prospect, drop completed). Single switch.
USE_FIVE_STAGE_FUNNEL = False

CLIENT_DEAL_STATUS_CURRENT = {
    "cold", "prospect", "resource_alignment", "requested",
    "negotiating", "signed", "customer",
    # Terminal lifecycle state (2026-10-04 offboarding directive, GROW-277):
    # offboarded deals are ENDED — excluded from live forward demand.
    # 'cold' remains a valid forward-demand signal (weight 0.1), so terminal
    # rows must not map to it. Mirrors CLUSTER_STATUS 'Offboarded' and the
    # existing status/offboarded tag already in cluster-note use.
    "offboarded",
}
CLIENT_DEAL_STATUS_FUNNEL = {
    "cold", "prospect", "resource_alignment", "negotiating", "signed",
}
CLIENT_DEAL_STATUS = (
    CLIENT_DEAL_STATUS_FUNNEL if USE_FIVE_STAGE_FUNNEL
    else CLIENT_DEAL_STATUS_CURRENT
)

CLIENT_DEBT_STATUS = {"unknown", "expired", "completed"}

CLIENT_DEAL_PIPELINES = {"integrated-infrastructure", "applied-research"}

# ---------------------------------------------------------------------------
# Cluster + hosted-training status vocabularies
# ---------------------------------------------------------------------------

CLUSTER_STATUS = {"LIVE", "Offboarded", "TBD", "Provisioned"}
HT_STATUS = {
    "target", "engaged", "email_sent", "qualified", "legal",
    "scoped", "proposal_sent", "procurement", "poc_live",
    "active_delivery", "live_lab_customer", "expansion", "on_hold",
}

# ---------------------------------------------------------------------------
# Entity folders the write-time gate watches
# ---------------------------------------------------------------------------

ENTITY_DIRS = (
    "CRM/vendor deals/", "CRM/client-deals/", "CRM/clusters/",
    "CRM/vendors/", "CRM/Accounts/", "CRM/contacts/",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_status(note_type: str | None, fm: dict, tags=None) -> list[str]:
    """Validate status for client-deal, cluster, and hosted-training namespaces.
    Vendor-deal is handled separately (the existing VALID_DEAL_STATUS path)."""
    fm = fm or {}
    v: list[str] = []
    if "ht_status" in fm:
        if fm.get("ht_status") not in HT_STATUS:
            v.append("unmapped_ht_status")
    elif "cluster_status" in fm:
        if fm.get("cluster_status") not in CLUSTER_STATUS:
            v.append("unmapped_cluster_status")
    elif note_type == "client-deal":
        ds = fm.get("deal_status")
        if ds is not None:
            if ds in CLIENT_DEBT_STATUS:
                v.append("deprecated_client_status")
            elif ds not in CLIENT_DEAL_STATUS:
                v.append("unmapped_client_status")
    return v


_FM_RE = re.compile(r"^﻿?---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(text: str | None) -> tuple[dict, list[str]]:
    """Minimal YAML-frontmatter parse. Returns (fm_dict, tags_list)."""
    fm: dict = {}
    tags: list[str] = []
    m = _FM_RE.match(text or "")
    if not m:
        return fm, tags
    lines = m.group(1).splitlines()
    i = 0
    while i < len(lines):
        mk = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", lines[i])
        if mk:
            key, val = mk.group(1), mk.group(2).strip()
            if key == "tags":
                if val.startswith("["):
                    inner = val[1 : val.rfind("]")] if "]" in val else val[1:]
                    tags = [
                        t.strip().strip("'\"")
                        for t in inner.split(",")
                        if t.strip()
                    ]
                else:
                    j = i + 1
                    while j < len(lines) and re.match(r"^\s*-\s+", lines[j]):
                        tags.append(
                            re.sub(r"^\s*-\s+", "", lines[j]).strip().strip("'\"")
                        )
                        j += 1
                    i = j - 1
            else:
                fm[key] = val.strip("'\"") if val != "" else None
        i += 1
    if tags:
        fm["tags"] = tags
    return fm, tags


def validate_markdown(text: str) -> list[str]:
    """Validate a vault note's raw markdown. Returns violation codes (empty = clean)."""
    fm, tags = parse_frontmatter(text)
    violations: list[str] = []
    note_type = fm.get("note_type")
    if not note_type:
        if fm.get("type") == "dashboard":
            return []
        violations.append("missing_note_type")
    if note_type == "vendor-deal":
        ds = fm.get("deal_status")
        if ds is not None:
            if ds not in VALID_DEAL_STATUS:
                violations.append("unmapped_deal_status")
            elif STATUS_TAG_MAP.get(ds) and STATUS_TAG_MAP[ds] not in tags:
                violations.append("missing_status_tag")
        # Clean split (2026-10-04): priority lives in `prio` (P1/P2/P3).
        prio = fm.get("prio")
        if prio is not None and str(prio) not in VALID_PRIO:
            violations.append("invalid_prio")
    violations.extend(_validate_status(note_type, fm, tags))
    return violations


def is_entity_path(path: str) -> bool:
    """True if the path targets one of the 7 entity folders."""
    return bool(path) and any(d in path for d in ENTITY_DIRS)
