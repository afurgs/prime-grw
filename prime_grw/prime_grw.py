"""prime-os-eval: Evaluates model ability to operate the AF-47 vault ecosystem.

Five task types: match scoring, schema validation, diff classification,
transcript extraction, and ops brief generation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

import verifiers.v1 as vf

logger = logging.getLogger(__name__)

# v0.2.11: typed task type strings prevent typos and enable IDE completion
TaskType = Literal["match", "validation", "diff", "extraction", "brief"]

# v0.5.2 sanitization (af-47 security finding, hub-public gate): no local
# vault path ships in the artifact. Vault mode is reachable only through the
# AF_VAULT_ROOT env var or the explicit vault_root config override; default
# operation is fully synthetic.
DEFAULT_VAULT_ROOT = None

# ---------------------------------------------------------------------------
# Synthetic data (fictional companies, realistic GPU distributions)
# ---------------------------------------------------------------------------

_SYNTH_VENDOR_ARCHETYPES = [
    {"name_prefix": "Aurora", "regions": ["US-East", "US-West"], "gpus": ["B200", "H200"], "price_posture": "mid"},
    {"name_prefix": "Nova", "regions": ["EU-North", "Finland"], "gpus": ["B300", "GB200"], "price_posture": "premium"},
    {"name_prefix": "Titan", "regions": ["US-West", "US-East"], "gpus": ["H100", "B200"], "price_posture": "budget"},
    {"name_prefix": "Polaris", "regions": ["India", "US-East"], "gpus": ["B200", "H100"], "price_posture": "budget"},
    {"name_prefix": "Meridian", "regions": ["EU-West", "EU-North"], "gpus": ["GB200", "NVL72"], "price_posture": "premium"},
    {"name_prefix": "Apex", "regions": ["US-East", "Iceland"], "gpus": ["H200", "B300"], "price_posture": "mid"},
    {"name_prefix": "Helios", "regions": ["US-West", "India"], "gpus": ["B200", "H100", "B300"], "price_posture": "mid"},
    {"name_prefix": "Zenith", "regions": ["Finland", "Norway"], "gpus": ["B300", "GB200"], "price_posture": "premium"},
    {"name_prefix": "Stratos", "regions": ["US-East", "US-West"], "gpus": ["H100", "H200", "B200"], "price_posture": "budget"},
    {"name_prefix": "Forge", "regions": ["EU-West", "US-East"], "gpus": ["B200", "B300", "GB200"], "price_posture": "mid"},
]

_SYNTH_CLIENT_ARCHETYPES = [
    {"name_prefix": "Nexara", "workload": "training", "scale": "large", "urgency": "high"},
    {"name_prefix": "Vortex", "workload": "training", "scale": "medium", "urgency": "medium"},
    {"name_prefix": "DeepFrame", "workload": "inference", "scale": "small", "urgency": "high"},
    {"name_prefix": "Cortex", "workload": "training", "scale": "large", "urgency": "low"},
    {"name_prefix": "Synapse", "workload": "fine-tuning", "scale": "medium", "urgency": "medium"},
    {"name_prefix": "Aether", "workload": "training", "scale": "xlarge", "urgency": "high"},
    {"name_prefix": "Quantum", "workload": "research", "scale": "small", "urgency": "low"},
    {"name_prefix": "Prism", "workload": "inference", "scale": "medium", "urgency": "medium"},
    {"name_prefix": "Atlas", "workload": "training", "scale": "large", "urgency": "medium"},
    {"name_prefix": "Helix", "workload": "fine-tuning", "scale": "small", "urgency": "low"},
    {"name_prefix": "Orbital", "workload": "training", "scale": "xlarge", "urgency": "high"},
    {"name_prefix": "Flux", "workload": "inference", "scale": "large", "urgency": "medium"},
    {"name_prefix": "Nebula", "workload": "training", "scale": "medium", "urgency": "high"},
    {"name_prefix": "Cipher", "workload": "research", "scale": "small", "urgency": "low"},
    {"name_prefix": "Lambda", "workload": "training", "scale": "large", "urgency": "medium"},
]

_SYNTH_PRICING = {
    "H100": {"budget": (1.00, 1.40), "mid": (1.30, 1.80), "premium": (1.70, 2.30)},
    "H200": {"budget": (1.60, 2.10), "mid": (2.00, 2.60), "premium": (2.50, 3.00)},
    "B200": {"budget": (2.00, 2.60), "mid": (2.40, 3.00), "premium": (2.90, 3.40)},
    "B300": {"budget": (2.80, 3.40), "mid": (3.20, 3.80), "premium": (3.60, 4.20)},
    "GB200": {"budget": (3.20, 3.80), "mid": (3.60, 4.20), "premium": (4.00, 4.80)},
    "NVL72": {"budget": (3.50, 4.10), "mid": (3.90, 4.50), "premium": (4.30, 5.20)},
}
_SYNTH_SELL_MARKUP = {"budget": 0.20, "mid": 0.40, "premium": 0.60}
_SYNTH_SCALE = {"small": [8, 16, 32, 64], "medium": [64, 128, 256], "large": [256, 512, 1024], "xlarge": [1024, 2048, 4096]}
_SYNTH_STATUSES_CLIENT = ["customer", "prospect", "prospect", "negotiating", "cold", "resource_alignment", "requested", "unknown", "expired"]
# Clean-split vocab (2026-10-04): lifecycle statuses only; "p1" -> "prospect".
_SYNTH_STATUSES_VENDOR = ["prospect", "prospect", "prospect", "signed", "signed", "negotiating", "lost"]
_SYNTH_TERMS = [3, 6, 6, 12, 12, 12, 24, 36]
_SYNTH_LOCATIONS = ["US-East", "US-West", "EU-North", "EU-West", "India", "Finland", "Iceland", "Norway"]


def _generate_synthetic_data(seed: int = 42) -> dict[str, list[dict]]:
    import random
    rng = random.Random(seed)
    data: dict[str, list[dict]] = {
        "vendor-deals": [], "client-deals": [], "clusters": [],
        "vendors": [], "clients": [], "hosted-training-deals": [],
        "matches": [], "sync_diffs": [], "validation_failures": [],
        "trajectories": {},
    }

    vendor_instances = []
    for i, arch in enumerate(_SYNTH_VENDOR_ARCHETYPES):
        for j, region in enumerate(arch["regions"]):
            suffix = f"-{region.lower().replace('-', '')}" if j > 0 else ""
            vname = f"{arch['name_prefix']} {region.split('-')[-1]}{suffix}"
            vslug = f"{arch['name_prefix'].lower()}-{region.lower().replace('-', '')}"
            vendor_instances.append((vname, vslug, region, arch["gpus"], arch["price_posture"]))
            data["vendors"].append({
                "path": f"vendors/{vslug}.md",
                "frontmatter": {
                    "note_type": "vendor", "name": vname, "slug": vslug,
                    "tags": ["type/vendor", "status/active"] + [f"gpu/{g.lower()}" for g in arch["gpus"]],
                    "hq_location": region, "contract_count": rng.randint(1, 5),
                    "active_cluster_count": rng.randint(0, 8), "last_synced": "2026-06-01",
                    "broken_link": False,
                },
                "body": f"\n# {vname}\n\n## Active Clusters\n",
                "wikilinks": [],
                "raw": f"---\nnote_type: vendor\nname: {vname}\nslug: {vslug}\n---\n# {vname}\n",
            })

    client_instances = []
    for arch in _SYNTH_CLIENT_ARCHETYPES:
        cstatus = rng.choice(_SYNTH_STATUSES_CLIENT)
        cslug = arch["name_prefix"].lower()
        cname = f"{arch['name_prefix']} AI"
        client_instances.append((cname, cslug, cstatus, arch["scale"], arch["urgency"]))
        data["clients"].append({
            "path": f"clients/{cslug}.md",
            "frontmatter": {
                "note_type": "client", "name": cname, "slug": cslug,
                "tags": ["type/client", f"status/{cstatus}"],
                "open_deal_count": rng.randint(1, 4),
                "active_cluster_count": rng.randint(0, 3), "last_synced": "2026-06-01",
                "broken_link": False,
            },
            "body": f"\n# {cname}\n\n## Active Clusters\n",
            "wikilinks": [],
            "raw": f"---\nnote_type: client\nname: {cname}\nslug: {cslug}\n---\n# {cname}\n",
        })

    for vi, (vname, vslug, vloc, vgpus, posture) in enumerate(vendor_instances):
        n_deals = rng.randint(2, 5)
        for j in range(n_deals):
            gpu = rng.choice(vgpus)
            count = rng.choice([64, 128, 256, 512, 1024, 2048])
            status = rng.choice(_SYNTH_STATUSES_VENDOR)
            pr = _SYNTH_PRICING.get(gpu, {}).get(posture, (2.5, 3.5))
            price = round(rng.uniform(pr[0], pr[1]), 2)
            avail_month = rng.randint(6, 10)
            avail_day = rng.randint(1, 28)
            earmark = rng.choice(client_instances)[1] if rng.random() < 0.25 else None
            deal_slug = f"{vslug}-{gpu.lower()}-{count}x-d{j+1}"
            data["vendor-deals"].append({
                "path": f"vendor-deals/{deal_slug}.md",
                "frontmatter": {
                    "note_type": "vendor-deal", "name": f"{vname} {gpu} {count}x",
                    "slug": deal_slug, "provider": f"[[vendors/{vslug}]]",
                    "gpu_type": gpu, "gpu_count": count,
                    "price_per_gpu_hour": price,
                    "availability_start": f"2026-{avail_month:02d}-{avail_day:02d}",
                    "contract_end_date": f"2027-{avail_month:02d}-{avail_day:02d}",
                    "term": rng.choice(["1 year", "2 year", "6 months"]),
                    "location": vloc, "deal_status": status,
                    "earmarked_for": earmark,
                    "tags": ["type/vendor-deal", f"gpu/{gpu.lower()}", f"vendor/{vslug}", f"status/{status}"],
                    "last_synced": "2026-06-01", "broken_link": False,
                },
                "body": f"\n# {vname} {gpu} {count}x\n\n**Vendor:** [[vendors/{vslug}]]\n",
                "wikilinks": [f"vendors/{vslug}"],
                "raw": f"---\nnote_type: vendor-deal\nslug: {deal_slug}\ndeal_status: {status}\ngpu_type: {gpu}\ngpu_count: {count}\nprice_per_gpu_hour: {price}\nlocation: {vloc}\navailability_start: 2026-{avail_month:02d}-{avail_day:02d}\n---\n",
            })

    for ci, (cname, cslug, cstatus, scale, urgency) in enumerate(client_instances):
        n_deals = rng.randint(2, 4)
        for j in range(n_deals):
            gpu = rng.choice(["H100", "H200", "B200", "B200", "B300", "GB200"])
            count = rng.choice(_SYNTH_SCALE.get(scale, [128]))
            posture = rng.choice(["mid", "premium"])
            pr = _SYNTH_PRICING.get(gpu, {}).get(posture, (2.5, 3.5))
            markup = _SYNTH_SELL_MARKUP.get(posture, 0.4)
            sell = round(rng.uniform(pr[0] + markup, pr[1] + markup + 0.3), 2)
            term = rng.choice(_SYNTH_TERMS)
            loc = rng.choice(_SYNTH_LOCATIONS) if rng.random() > 0.3 else None
            deal_slug = f"{cslug}-{gpu.lower()}-{count}x-d{j+1}"
            data["client-deals"].append({
                "path": f"client-deals/{deal_slug}.md",
                "frontmatter": {
                    "note_type": "client-deal", "name": f"{cname} {gpu} {count}x",
                    "slug": deal_slug, "account": f"[[clients/{cslug}]]",
                    "gpu_type": gpu, "gpu_count": count,
                    "deal_status": cstatus, "sell_rate": sell,
                    "contract_term_months": term, "location": loc,
                    "tags": ["type/client-deal", f"gpu/{gpu.lower()}", f"status/{cstatus}"],
                    "last_synced": "2026-06-01", "broken_link": False,
                },
                "body": f"\n# {cname} {gpu} {count}x\n\n**Client:** [[clients/{cslug}]]\n",
                "wikilinks": [f"clients/{cslug}"],
                "raw": f"---\nnote_type: client-deal\nslug: {deal_slug}\ndeal_status: {cstatus}\ngpu_type: {gpu}\ngpu_count: {count}\nsell_rate: {sell}\ncontract_term_months: {term}\nlocation: {loc}\n---\n",
            })

    cluster_statuses = ["LIVE"] * 6 + ["Offboarded"] * 2 + ["TBD"]
    for i in range(50):
        v = rng.choice(vendor_instances)
        c = rng.choice(client_instances)
        gpu = rng.choice(v[3])
        count = rng.choice([16, 32, 64, 128, 256])
        cstatus = rng.choice(cluster_statuses)
        cslug = f"{v[1]}-{c[1]}-cl{i+1}"
        posture = v[4]
        pr = _SYNTH_PRICING.get(gpu, {}).get(posture, (2.5, 3.5))
        data["clusters"].append({
            "path": f"clusters/{cslug}.md",
            "frontmatter": {
                "note_type": "cluster", "name": f"{v[0]} - {c[0]} Cluster {i+1}",
                "slug": cslug, "cluster_status": cstatus,
                "gpu_type": gpu, "gpu_count": count,
                "provider": f"[[vendors/{v[1]}]]", "account": f"[[clients/{c[1]}]]",
                "contract_start_date": "2026-01-15",
                "contract_end_date": f"2027-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                "client_rate": round(rng.uniform(pr[0] + 0.3, pr[1] + 0.5), 2),
                "provider_rate": round(rng.uniform(pr[0], pr[1]), 2),
                "tags": ["type/cluster", f"status/{cstatus.lower()}", f"gpu/{gpu.lower()}"],
                "last_synced": "2026-06-01", "broken_link": False,
            },
            "body": f"\n# {v[0]} - {c[0]} Cluster {i+1}\n\n**Provider:** [[vendors/{v[1]}]]\n**Client:** [[clients/{c[1]}]]\n",
            "wikilinks": [f"vendors/{v[1]}", f"clients/{c[1]}"],
            "raw": f"---\nnote_type: cluster\nslug: {cslug}\ncluster_status: {cstatus}\ngpu_type: {gpu}\ngpu_count: {count}\n---\n",
        })

    ht_statuses = ["prospect", "qualification", "negotiating", "poc"]
    for ci, (cname, cslug, _, _, _) in enumerate(client_instances[:8]):
        data["hosted-training-deals"].append({
            "path": f"hosted-training-deals/{cslug}-ht.md",
            "frontmatter": {
                "note_type": "hosted-training-deal", "name": f"{cname} Hosted Training",
                "slug": f"{cslug}-ht", "account": f"[[clients/{cslug}]]",
                "ht_status": rng.choice(ht_statuses),
                "primary_product": rng.sample(["Custom Dev", "Post-Training Stack", "Fine-Tuning", "Inference"], 2),
                "targeted_land_monthly": rng.choice([50000, 100000, 250000, 500000, None]),
                "tags": ["type/hosted-training-deal", f"status/{rng.choice(ht_statuses)}", f"client/{cslug}"],
                "last_synced": "2026-06-01", "broken_link": False,
            },
            "body": f"\n# {cname} Hosted Training\n\n**Client:** [[clients/{cslug}]]\n",
            "wikilinks": [f"clients/{cslug}"],
            "raw": f"---\nnote_type: hosted-training-deal\nslug: {cslug}-ht\n---\n",
        })

    diff_templates = [
        ("approve", "none", "| description | Old notes | 6/1: {cn} expanded cluster. Rate ${rate:.2f}/hr confirmed. | UPDATE |"),
        ("approve", "none", "| deal_notes_secondary | empty | 6/1: Renewal discussion. Upgrade path confirmed. | UPDATE |"),
        ("approve", "none", "| amount | {old_amt} | {amt_up20} | UPDATE |"),
        ("approve", "none", "| description | Q1 summary | 6/2: {cn} technical review passed. | UPDATE |"),
        ("reject", "stale_source", "| description | {cn} prospect | {cn} prospect | UPDATE |"),
        ("reject", "stale_source", "| description | Active deal | | UPDATE |"),
        ("reject", "wrong_stage", "| dealstage | closedwon | qualifiedtobuy | UPDATE |"),
        ("reject", "missing_field", "| amount | {old_amt} | | UPDATE |"),
        ("reject", "scope_violation", "| hubspot_owner_id | 12345 | 67890 | UPDATE |"),
        ("reject", "formatting", "| description | Clean notes | 6/1: <script>alert(1)</script> | UPDATE |"),
        ("defer", "none", "| dealstage | qualifiedtobuy | decisionmakerboughtin | UPDATE |\n| amount | {old_amt} | {new_amt} | UPDATE |"),
        ("defer", "none", "| amount | {old_amt} | {new_amt} | UPDATE |"),
        ("approve", "none", "| description | Intro call notes | 6/3: {cn} confirmed budget. Rate ${rate:.2f}/hr quoted. | UPDATE |"),
    ]
    data["sync_diffs"] = []
    for batch in range(4):
        diff_body = f"# Sync Diff: Synthetic Batch {batch+1}\n\n"
        diff_clients_sample = rng.sample(client_instances, min(len(client_instances), 15))
        for i, (cn, cs, _, _, _) in enumerate(diff_clients_sample):
            tmpl = rng.choice(diff_templates)
            decision, reason_code, field_tmpl = tmpl
            rate = rng.uniform(2, 4)
            old_amt = rng.randint(100, 500) * 1000
            new_amt = rng.randint(300, 900) * 1000
            fields = field_tmpl.format(cn=cn, rate=rate, old_amt=old_amt, new_amt=new_amt,
                                       amt_up20=int(old_amt * 1.2))
            hs_id = rng.randint(10000000, 99999999)
            diff_body += f"### {cn} ({hs_id})\n| Field | HubSpot | Notion | Action |\n|---|---|---|---|\n{fields}\n\n"
        data["sync_diffs"].append({
            "path": f"staging/sync_diff_synthetic_{batch+1}.md",
            "frontmatter": {},
            "body": diff_body,
            "wikilinks": [],
            "raw": "",
        })
    return data
MAPPING_MASTER_REL = "agents/staging/obsidian_mapping_master.md"

GPU_FAMILIES = {
    "H100": "hopper",
    "H200": "hopper",
    "B200": "blackwell",
    "B300": "blackwell",
    "GB200": "blackwell",
    "GB300": "blackwell",
    "NVL72": "blackwell",
    "A100": "ampere",
}

GEOGRAPHY_REGIONS = {
    "US-East": "americas",
    "US-West": "americas",
    "US": "americas",
    "USA": "americas",
    "North Carolina, USA": "americas",
    "N. Carolina, USA": "americas",
    "Texas, USA": "americas",
    "India": "asia",
    "EU-North": "europe",
    "EU-West": "europe",
    "Finland": "europe",
    "Iceland": "europe",
    "Norway": "europe",
}

# ---------------------------------------------------------------------------
# Vault parsing (ported from VLM-1)
# ---------------------------------------------------------------------------

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_vault_note(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = text.split("---", 2)
    frontmatter: dict[str, Any] = {}
    body = text
    if len(parts) >= 3 and parts[0].strip() == "":
        try:
            import yaml
            frontmatter = yaml.safe_load(parts[1]) or {}
        except Exception:
            frontmatter = {}
        body = parts[2]
    wikilinks = _WIKILINK_RE.findall(body)
    return {
        "path": str(path),
        "frontmatter": frontmatter,
        "body": body,
        "wikilinks": wikilinks,
        "raw": text,
    }


def load_vault_folder(folder: Path) -> list[dict[str, Any]]:
    if not folder.exists():
        return []
    notes = []
    for p in sorted(folder.glob("*.md")):
        if p.name.startswith((".", "_")):
            continue
        try:
            notes.append(parse_vault_note(p))
        except Exception:
            continue
    return notes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vault_data(vault_root: Path) -> dict[str, list[dict]]:
    domain_paths = {
        "vendor-deals": Path("CRM") / "vendor deals",
        "client-deals": Path("CRM") / "client-deals",
        "clusters": Path("CRM") / "clusters",
        "vendors": Path("CRM") / "vendors",
        "clients": Path("CRM") / "Accounts",
        "hosted-training-deals": Path("CRM") / "hosted-training-deals",
    }
    data: dict[str, list[dict]] = {}
    for key, rel in domain_paths.items():
        data[key] = load_vault_folder(vault_root / rel)
    # Match notes
    matches_dir = vault_root / "_runtime" / "matches"
    staging_matches = vault_root / "_runtime" / "staging" / "matches"
    data["matches"] = load_vault_folder(matches_dir) + load_vault_folder(staging_matches)
    # Trajectories
    traj_dir = vault_root / "_reference" / "builds" / "trajectories"
    data["trajectories"] = {}
    if traj_dir.exists():
        for agent_dir in traj_dir.iterdir():
            if agent_dir.is_dir():
                jsonl_files = []
                for f in sorted(agent_dir.glob("*.jsonl")):
                    try:
                        for line in f.read_text().strip().splitlines():
                            jsonl_files.append(json.loads(line))
                    except Exception:
                        continue
                data["trajectories"][agent_dir.name] = jsonl_files
    # Sync diffs
    staging = vault_root / "_runtime" / "staging"
    data["sync_diffs"] = []
    if staging.exists():
        for f in sorted(staging.glob("sync_diff*.md")):
            try:
                data["sync_diffs"].append(parse_vault_note(f))
            except Exception:
                continue
    # Validation failures
    data["validation_failures"] = []
    if staging.exists():
        for f in sorted(staging.glob("validation_failures*.md")):
            try:
                data["validation_failures"].append(parse_vault_note(f))
            except Exception:
                continue
    return data


# ---------------------------------------------------------------------------
# Programmatic scoring (ground truth for match scoring)
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return datetime.fromisoformat(str(v)).date()
    except (ValueError, TypeError):
        return None


def _gpu_type_score(supply_gpu: str | None, demand_gpu: str | None) -> int:
    if not supply_gpu or not demand_gpu:
        return 0
    s, d = supply_gpu.upper(), demand_gpu.upper()
    if s == d:
        return 30
    sf = GPU_FAMILIES.get(s, "")
    df = GPU_FAMILIES.get(d, "")
    if sf and sf == df:
        return 15
    return 0


def _scale_score(supply_count: int | None, demand_count: int | None) -> int:
    if not supply_count or not demand_count:
        return 0
    ratio = supply_count / demand_count
    if ratio >= 1.0:
        return 25
    if ratio >= 0.75:
        return 15
    if ratio >= 0.50:
        return 5
    return 0


# v0.5 determinism fix (af-47 review round, 2026-10-08): match GT was computed
# against date.today(), so rubric dims — and therefore probe means and golden
# fixtures — DRIFTED BY CALENDAR DAY (const50_dims read 0.126 through
# 2026-10-07 and 0.274 on 2026-10-08 as synthetic availability dates crossed
# the <=14/<=28-day thresholds). All match scoring now anchors on a pinned
# reference date aligned with the synthetic vault's last_synced stamp; the
# question's DERIVED block uses the same anchor, so questions and GT stay
# mutually consistent. Same bug family as the PYTHONHASHSEED set-order fix.
MATCH_REFERENCE_DATE = date(2026, 6, 1)


def _timing_score(availability_start: date | None, today: date | None = None) -> int:
    if not availability_start:
        return 5
    today = today or MATCH_REFERENCE_DATE
    days_until = (availability_start - today).days
    if days_until <= 0:
        return 15
    if days_until <= 14:
        return 10
    if days_until <= 28:
        return 5
    return 0


def _term_score(
    supply_term: str | None,
    supply_end: date | None,
    demand_term_months: int | None,
) -> int:
    supply_months = None
    if supply_term:
        m = re.search(r"(\d+)", str(supply_term))
        if m:
            val = int(m.group(1))
            if "year" in str(supply_term).lower():
                supply_months = val * 12
            else:
                supply_months = val
    if not supply_months and supply_end:
        today = MATCH_REFERENCE_DATE
        supply_months = max(1, (supply_end.year - today.year) * 12 + supply_end.month - today.month)
    if not supply_months or not demand_term_months:
        return 5
    ratio = supply_months / demand_term_months
    if ratio >= 1.0:
        return 15
    if ratio >= 0.80:
        return 10
    if ratio >= 0.60:
        return 5
    return 0


def _geography_score(supply_loc: str | None, demand_loc: str | None) -> int:
    if not supply_loc or not demand_loc:
        return 3
    if supply_loc.strip().lower() == demand_loc.strip().lower():
        return 10
    sr = GEOGRAPHY_REGIONS.get(supply_loc.strip(), supply_loc.strip().lower())
    dr = GEOGRAPHY_REGIONS.get(demand_loc.strip(), demand_loc.strip().lower())
    if sr == dr:
        return 5
    return 0


def _price_score(
    supply_price: float | None,
    demand_rate: float | None,
) -> int:
    if supply_price is None or demand_rate is None:
        return 0
    if supply_price <= 0 or demand_rate <= 0:
        return 0
    margin = demand_rate - supply_price
    if margin > 0.5:
        return 5
    if margin > 0:
        return 3
    return 0


def compute_match_score(supply_fm: dict, demand_fm: dict) -> dict:
    dims = {
        "gpu_type": _gpu_type_score(supply_fm.get("gpu_type"), demand_fm.get("gpu_type")),
        "scale": _scale_score(_safe_int(supply_fm.get("gpu_count")), _safe_int(demand_fm.get("gpu_count"))),
        "timing": _timing_score(_parse_date(supply_fm.get("availability_start"))),
        "term": _term_score(
            supply_fm.get("term"),
            _parse_date(supply_fm.get("contract_end_date")),
            _safe_int(demand_fm.get("contract_term_months")),
        ),
        "geography": _geography_score(supply_fm.get("location"), demand_fm.get("location")),
        "price": _price_score(
            _safe_float(supply_fm.get("price_per_gpu_hour")),
            _safe_float(demand_fm.get("sell_rate")),
        ),
    }
    total = sum(dims.values())
    if total >= 60:
        rec = "earmark"
    elif total >= 40:
        rec = "monitor"
    else:
        rec = "pass"
    return {"match_score": total, "dimensions": dims, "recommendation": rec}


# ---------------------------------------------------------------------------
# Schema validation (OS Section 10 rules)
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "vendor": ["name", "slug"],
    "client": ["name", "slug"],
    "cluster": ["name", "slug"],
    "vendor-deal": ["name", "slug", "provider"],
    "client-deal": ["name", "slug"],
    "hosted-training-deal": ["name", "slug", "account"],
    "match": ["slug"],
    "signal": ["slug"],
}

from vault_validate import (
    VALID_DEAL_STATUS,
    VALID_PRIO,
    STATUS_TAG_MAP,
    CLIENT_DEAL_STATUS,
    CLIENT_DEBT_STATUS,
    CLUSTER_STATUS,
    HT_STATUS,
    _validate_status,
)


def validate_note(note: dict) -> list[dict]:
    fm = note.get("frontmatter", {})
    body = note.get("body", "")
    wikilinks = note.get("wikilinks", [])
    violations = []

    note_type = fm.get("note_type", "")
    if not note_type:
        violations.append({"rule": "missing_note_type", "field": "note_type", "message": "note_type field missing"})
        return violations

    required = REQUIRED_FIELDS.get(note_type, [])
    for field in required:
        val = fm.get(field)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            violations.append({"rule": "missing_required_field", "field": field, "message": f"Required field '{field}' is empty"})

    slug = fm.get("slug", "")
    if slug and slug != slug.lower():
        violations.append({"rule": "slug_not_lowercase", "field": "slug", "message": f"Slug '{slug}' must be lowercase"})
    if slug and " " in slug:
        violations.append({"rule": "slug_has_spaces", "field": "slug", "message": f"Slug '{slug}' must use hyphens not spaces"})

    tags = fm.get("tags", [])
    if isinstance(tags, list):
        has_type_tag = any(t.startswith("type/") for t in tags if isinstance(t, str))
        has_status_tag = any(t.startswith("status/") for t in tags if isinstance(t, str))
        if not has_type_tag:
            violations.append({"rule": "missing_type_tag", "field": "tags", "message": "Missing type/ tag"})
        if not has_status_tag:
            violations.append({"rule": "missing_status_tag", "field": "tags", "message": "Missing status/ tag"})

    if note_type == "vendor-deal":
        deal_status = fm.get("deal_status")
        if deal_status and deal_status not in VALID_DEAL_STATUS:
            violations.append({
                "rule": "unmapped_deal_status",
                "field": "deal_status",
                "message": f"deal_status '{deal_status}' not in mapping table",
            })
        prio = fm.get("prio")
        if prio is not None and str(prio) not in VALID_PRIO:
            violations.append({
                "rule": "invalid_prio",
                "field": "prio",
                "message": f"prio '{prio}' not in P1/P2/P3",
            })
        if deal_status and deal_status in STATUS_TAG_MAP:
            expected_tag = STATUS_TAG_MAP[deal_status]
            if isinstance(tags, list) and expected_tag not in tags:
                violations.append({
                    "rule": "missing_status_tag_for_deal_status",
                    "field": "tags",
                    "message": f"deal_status='{deal_status}' requires tag '{expected_tag}'",
                })

    provider_link = fm.get("provider", "")
    if note_type in ("vendor-deal", "cluster") and isinstance(provider_link, str) and "[[" in provider_link:
        target = _WIKILINK_RE.search(provider_link)
        # Rule 7 is enforced on what the evaluator can see: the rendered body.
        # The note's `wikilinks` metadata is derived state the agent never sees.
        if target and target.group(1) not in (body or ""):
            violations.append({
                "rule": "frontmatter_wikilink_not_in_body",
                "field": "provider",
                "message": f"Provider wikilink {target.group(1)} not found in body",
            })

    for code in _validate_status(note_type, fm, tags):
        violations.append({"rule": code, "field": "deal_status", "message": code})

    return violations


# ---------------------------------------------------------------------------
# Diff parsing (from sync diff markdown)
# ---------------------------------------------------------------------------

def parse_sync_diff(note: dict) -> list[dict]:
    body = note.get("body", "") or note.get("raw", "")
    records = []
    current_deal = None
    for line in body.splitlines():
        if line.startswith("### "):
            deal_name = line.lstrip("# ").strip()
            match = re.match(r"(.+?)\s*\((\d+)\)", deal_name)
            if match:
                current_deal = {"deal_name": match.group(1).strip(), "hubspot_id": match.group(2), "changes": []}
            else:
                current_deal = {"deal_name": deal_name, "hubspot_id": None, "changes": []}
            records.append(current_deal)
        elif current_deal and line.startswith("| ") and "---" not in line and "Field" not in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 4:
                current_deal["changes"].append({
                    "field": parts[0],
                    "hubspot_value": parts[1],
                    "notion_value": parts[2],
                    "action": parts[3],
                })
    return records


# ---------------------------------------------------------------------------
# Task row builders
# ---------------------------------------------------------------------------

MATCH_SYSTEM_PROMPT = """Score this GPU supply-demand match. Respond with ONLY a JSON object, no explanation.

Scoring rules (normalized to 100):
- gpu_type (0-30): exact=30, same family=15, mismatch=0. Families: hopper(H100,H200), blackwell(B200,B300,GB200,GB300,NVL72), ampere(A100)
- scale (0-25): supply/demand ratio >=1.0->25, >=0.75->15, >=0.5->5, else 0
- timing (0-15): days until available <=0->15, <=14->10, <=28->5, else 0
- term (0-15): supply months / demand months >=1.0->15, >=0.8->10, >=0.6->5, else 0
- geography (0-10): exact location->10, same continent->5, else 0
- price (0-5): margin (sell-cost) >$0.50->5, >$0->3, else 0

Total >=60 -> "earmark", >=40 -> "monitor", else "pass"

Output: {"match_score":<int>,"dimensions":{"gpu_type":<int>,"scale":<int>,"timing":<int>,"term":<int>,"geography":<int>,"price":<int>},"recommendation":"earmark"|"monitor"|"pass"}"""

# v0.2.10: the status vocabularies below are rendered from vault_validate.py
# (single source of truth). Before 0.2.10 the prompt stated only the
# vendor-deal vocabulary while the GT labeler (validate_note -> vault_validate)
# also enforced client-deal/cluster/hosted-training status rules: 131/375
# validation rows (35%) were unanswerable from the stated rules — a flawless
# rule-follower replay scored 0.574 (0.864 once the vocabularies are stated).
_VENDOR_STATUS_VOCAB = "|".join(sorted(VALID_DEAL_STATUS))
_PRIO_VOCAB = "|".join(sorted(VALID_PRIO))
_CLIENT_STATUS_VOCAB = "|".join(sorted(CLIENT_DEAL_STATUS))
_CLIENT_DEBT_VOCAB = "|".join(sorted(CLIENT_DEBT_STATUS))
_CLUSTER_STATUS_VOCAB = "|".join(sorted(CLUSTER_STATUS))
_HT_STATUS_VOCAB = "|".join(sorted(HT_STATUS))

VALIDATION_SYSTEM_PROMPT = f"""Validate this vault note against schema rules. Respond with ONLY a JSON object.

Rules: (1) note_type required (2) name, slug required (3) slug must be lowercase-kebab-case (4) tags must include type/ and status/ (5) vendor-deal deal_status must be (lifecycle only): {_VENDOR_STATUS_VOCAB} — and the prio field, when present, must be: {_PRIO_VOCAB} (6) vendor-deal needs status/ tag matching deal_status (7) frontmatter wikilinks must appear in body (8) client-deal deal_status must be: {_CLIENT_STATUS_VOCAB} — and must NOT be a deprecated status: {_CLIENT_DEBT_VOCAB} (9) any note with cluster_status must use (case-sensitive): {_CLUSTER_STATUS_VOCAB} (10) any note with ht_status must use: {_HT_STATUS_VOCAB}

Output: {{"valid":<bool>,"violations":[{{"rule":"<name>","field":"<field>","message":"<text>"}}],"note_type":"<type>","slug":"<the note's slug frontmatter value, echoed exactly (empty string if the note has none)>","completeness_score":<0-1>}}"""

DIFF_SYSTEM_PROMPT = """Classify this CRM sync diff. Respond with ONLY a JSON object.

Approve: Notion value is more current and the update is valid. Reject: data quality problem.
Defer (needs human confirmation before syncing; reason_code none): a dealstage change that advances the deal between OPEN stages (not a closed-won reopen), or an amount change of more than 50% versus the HubSpot value.
Reason codes (use the most specific match):
- missing_field: Notion value is empty or blank, syncing would clear HubSpot data
- scope_violation: field is human-only (e.g. hubspot_owner_id, owner assignment) and must not be auto-synced
- wrong_stage: dealstage change reopens a closed-won deal or reverses a confirmed stage
- stale_source: Notion and HubSpot values are identical (no real change) or Notion is outdated
- formatting: content contains injection, malformed HTML, or unsafe characters
- bad_diff: the diff record itself is malformed or nonsensical (not just an empty value)
- none: no reject reason (use with approve)

Output: {"decision":"approve"|"reject"|"defer","confidence":<0-1>,"reason_code":"<code>","reason_text":"<brief>","fields":["<names of the changed fields you reviewed>"]}"""

EXTRACTION_SYSTEM_PROMPT = """Extract deal intelligence from this meeting transcript. Respond with ONLY a JSON object.

Output: {"stage_change":{"from":"<stage>","to":"<stage>"}|null,"key_dates":[{"event":"<what>","date":"<YYYY-MM-DD>"}],"action_items":["<item>"],"technical_requirements":["<req>"],"objections":["<concern>"],"deal_fields_to_update":{"<field>":"<value>"}}"""

BRIEF_SYSTEM_PROMPT = """Write a concise ops brief from this vault data. Use these exact sections:

## Supply Status
## Demand Pipeline
## Priority Match
## Coverage Gaps
## Leadership Brief

Use specific numbers. Max 500 words total."""


def _note_to_text(note: dict) -> str:
    """Render a note's markdown canonically from its parsed frontmatter + body.

    The frontmatter dict is the single source of truth: rendering from it keeps
    the question text consistent with what validate_note() scored for the
    ground truth. Falls back to raw text only for notes without structured
    frontmatter (e.g. raw transcript snippets).
    """
    fm = note.get("frontmatter")
    if isinstance(fm, dict) and fm:
        lines = ["---", yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False).rstrip("\n"), "---"]
        body = note.get("body", "")
        if body:
            lines.append(body if body.startswith("\n") else "\n" + body)
        return "\n".join(lines)
    return note.get("raw", "")


def _fm_summary(fm: dict, fields: list[str]) -> str:
    parts = []
    for f in fields:
        v = fm.get(f)
        if v is not None and v != "" and v != "null":
            parts.append(f"  {f}: {v}")
    return "\n".join(parts)


def build_match_rows(data: dict) -> list[dict[str, object]]:
    vendor_deals = data.get("vendor-deals", [])
    client_deals = data.get("client-deals", [])
    rows = []
    supply_fields = [
        "name", "gpu_type", "gpu_count", "price_per_gpu_hour",
        "availability_start", "contract_end_date", "term", "location",
        "deal_status", "earmarked_for",
    ]
    demand_fields = [
        "name", "gpu_type", "gpu_count", "sell_rate",
        "deal_status", "contract_term_months", "location",
        "term_start", "term_end",
    ]
    for vd in vendor_deals:
        vfm = vd.get("frontmatter", {})
        vd_gpu = (vfm.get("gpu_type") or "").upper()
        for cd in client_deals:
            cfm = cd.get("frontmatter", {})
            cd_gpu = (cfm.get("gpu_type") or "").upper()
            if not vd_gpu or not cd_gpu:
                continue
            if _gpu_type_score(vd_gpu, cd_gpu) == 0:
                continue
            ground_truth = compute_match_score(vfm, cfm)
            avail = _parse_date(vfm.get("availability_start"))
            days_until = (avail - MATCH_REFERENCE_DATE).days if avail else None
            supply_price = _safe_float(vfm.get("price_per_gpu_hour"))
            demand_rate = _safe_float(cfm.get("sell_rate"))
            margin = round(demand_rate - supply_price, 2) if supply_price and demand_rate else None
            scale_ratio = round((_safe_int(vfm.get("gpu_count")) or 0) / max(_safe_int(cfm.get("gpu_count")) or 1, 1), 2)
            derived = (
                f"\n\nDERIVED (pre-computed):"
                f"\n  days_until_available: {days_until}"
                f"\n  scale_ratio: {scale_ratio}"
                f"\n  margin_per_gpu_hr: ${margin}" if margin is not None else ""
            )
            question = (
                f"{MATCH_SYSTEM_PROMPT}\n\n"
                f"SUPPLY: {_fm_summary(vfm, supply_fields)}\n\n"
                f"DEMAND: {_fm_summary(cfm, demand_fields)}"
                f"{derived}"
            )
            rows.append({
                "question": question,
                "answer": json.dumps(ground_truth),
                "info": json.dumps({
                    "task_type": "match",
                    "supply_slug": vfm.get("slug", ""),
                    "demand_slug": cfm.get("slug", ""),
                    "ground_truth": ground_truth,
                }),
            })
    return rows


def _generate_invalid_notes(rng, count: int) -> list[dict]:
    """Generate synthetic notes with seeded ground-truth violations.
    Uses vault_validate to label them so eval and gate share one vocabulary."""
    from vault_validate import (
        validate_markdown, VALID_DEAL_STATUS, STATUS_TAG_MAP,
        CLIENT_DEAL_STATUS, CLIENT_DEBT_STATUS, CLUSTER_STATUS, HT_STATUS,
    )
    invalid_notes = []
    bad_vendor_statuses = ["active", "pending", "open", "draft", "in_review"]
    bad_client_statuses = ["qualification", "active", "pipeline", "won", "signed_won"]
    bad_cluster_statuses = ["active", "running", "deployed", "online", "live"]
    bad_ht_statuses = ["customer", "active", "pipeline", "signed", "cold"]
    templates = [
        ("invalid_prio", "vendor-deal", lambda i, rng: {
            "raw": f"---\nnote_type: vendor-deal\nname: Bad Prio {i}\nslug: bad-prio-{i}\nprovider: \"[[vendors/test-vendor]]\"\ndeal_status: prospect\nprio: {rng.choice(['P0', 'P4', 'high', 'urgent', '1'])}\ntags:\n  - type/vendor-deal\n  - status/prospect\n---\n# Bad Prio {i}\n\n**Vendor:** [[vendors/test-vendor]]\n",
        }),
        ("unmapped_deal_status", "vendor-deal", lambda i, rng: {
            "raw": f"---\nnote_type: vendor-deal\nname: Bad Vendor {i}\nslug: bad-vendor-{i}\nprovider: \"[[vendors/test-vendor]]\"\ndeal_status: {rng.choice(bad_vendor_statuses)}\ntags:\n  - type/vendor-deal\n  - status/{rng.choice(bad_vendor_statuses)}\n---\n# Bad Vendor {i}\n\n**Vendor:** [[vendors/test-vendor]]\n",
        }),
        ("missing_status_tag", "vendor-deal", lambda i, rng: {
            "raw": f"---\nnote_type: vendor-deal\nname: Missing Tag {i}\nslug: missing-tag-{i}\nprovider: \"[[vendors/test-vendor]]\"\ndeal_status: {rng.choice(sorted(VALID_DEAL_STATUS - {'lost', 'closed'}))}\ntags:\n  - type/vendor-deal\n---\n# Missing Tag {i}\n\n**Vendor:** [[vendors/test-vendor]]\n",
        }),
        ("unmapped_client_status", "client-deal", lambda i, rng: {
            "raw": f"---\nnote_type: client-deal\nname: Bad Client {i}\nslug: bad-client-{i}\naccount: \"[[clients/test-client]]\"\ndeal_status: {rng.choice(bad_client_statuses)}\ntags:\n  - type/client-deal\n  - status/{rng.choice(bad_client_statuses)}\n---\n# Bad Client {i}\n\n**Client:** [[clients/test-client]]\n",
        }),
        ("deprecated_client_status", "client-deal", lambda i, rng: {
            "raw": f"---\nnote_type: client-deal\nname: Debt Client {i}\nslug: debt-client-{i}\naccount: \"[[clients/test-client]]\"\ndeal_status: {rng.choice(sorted(CLIENT_DEBT_STATUS))}\ntags:\n  - type/client-deal\n  - status/{rng.choice(sorted(CLIENT_DEBT_STATUS))}\n---\n# Debt Client {i}\n\n**Client:** [[clients/test-client]]\n",
        }),
        ("unmapped_cluster_status", "cluster", lambda i, rng: {
            "raw": f"---\nnote_type: cluster\nname: Bad Cluster {i}\nslug: bad-cluster-{i}\ncluster_status: {rng.choice(bad_cluster_statuses)}\ngpu_type: H100\ngpu_count: {rng.choice([64, 128, 256])}\nprovider: \"[[vendors/test-vendor]]\"\naccount: \"[[clients/test-client]]\"\ntags:\n  - type/cluster\n  - status/{rng.choice(bad_cluster_statuses)}\n---\n# Bad Cluster {i}\n\n**Provider:** [[vendors/test-vendor]]\n**Client:** [[clients/test-client]]\n",
        }),
        ("unmapped_ht_status", "hosted-training-deal", lambda i, rng: {
            "raw": f"---\nnote_type: hosted-training-deal\nname: Bad HT {i}\nslug: bad-ht-{i}\naccount: \"[[clients/test-client]]\"\nht_status: {rng.choice(bad_ht_statuses)}\ntags:\n  - type/hosted-training-deal\n  - status/{rng.choice(bad_ht_statuses)}\n---\n# Bad HT {i}\n\n**Client:** [[clients/test-client]]\n",
        }),
    ]
    per_code = max(1, count // len(templates))
    idx = 0
    for expected_code, note_type, builder in templates:
        for j in range(per_code):
            note_data = builder(idx, rng)
            raw = note_data["raw"]
            violations = validate_markdown(raw)
            violation_rules = [v for v in violations if v != "missing_note_type"]
            if not any(expected_code in v for v in violation_rules):
                continue
            parts = raw.split("---", 2)
            fm_dict = {}
            if len(parts) >= 3:
                import yaml
                try:
                    fm_dict = yaml.safe_load(parts[1]) or {}
                except Exception:
                    fm_dict = {}
            body = parts[2] if len(parts) >= 3 else ""
            wikilinks = _WIKILINK_RE.findall(body)
            note = {
                "path": f"synthetic-invalid/{expected_code}-{idx}.md",
                "frontmatter": fm_dict,
                "body": body,
                "wikilinks": wikilinks,
                "raw": raw,
            }
            invalid_notes.append(note)
            idx += 1
    rng.shuffle(invalid_notes)
    return invalid_notes[:count]


def build_validation_rows(data: dict) -> list[dict[str, object]]:
    import random
    rows_clean = []
    rows_invalid = []
    all_notes = []
    for domain in ["vendor-deals", "client-deals", "clusters", "vendors", "clients", "hosted-training-deals"]:
        all_notes.extend(data.get(domain, []))
    for note in all_notes:
        violations = validate_note(note)
        is_valid = len(violations) == 0
        fm = note.get("frontmatter", {})
        total_fields = len([v for v in fm.values() if v is not None and v != ""])
        max_fields = max(len(fm), 1)
        completeness = total_fields / max_fields
        ground_truth = {
            "valid": is_valid,
            "violations": violations,
            "note_type": fm.get("note_type", "unknown"),
            "slug": str(fm.get("slug", "") or ""),
            "completeness_score": round(completeness, 2),
        }
        question = (
            f"{VALIDATION_SYSTEM_PROMPT}\n\n"
            f"Validate this vault note:\n\n```markdown\n{_note_to_text(note)}\n```"
        )
        row = {
            "question": question,
            "answer": json.dumps(ground_truth),
            "info": json.dumps({
                "task_type": "validation",
                "note_path": note.get("path", ""),
                "ground_truth": ground_truth,
            }),
        }
        if is_valid:
            rows_clean.append(row)
        else:
            rows_invalid.append(row)
    rng = random.Random(42)
    target_invalid = max(len(rows_clean), len(rows_invalid) + 1)
    needed = target_invalid - len(rows_invalid)
    if needed > 0:
        synth_invalid = _generate_invalid_notes(rng, needed)
        for note in synth_invalid:
            violations = validate_note(note)
            fm = note.get("frontmatter", {})
            total_fields = len([v for v in fm.values() if v is not None and v != ""])
            max_fields = max(len(fm), 1)
            completeness = total_fields / max_fields
            ground_truth = {
                "valid": False,
                "violations": violations,
                "note_type": fm.get("note_type", "unknown"),
                "slug": str(fm.get("slug", "") or ""),
                "completeness_score": round(completeness, 2),
            }
            question = (
                f"{VALIDATION_SYSTEM_PROMPT}\n\n"
                f"Validate this vault note:\n\n```markdown\n{_note_to_text(note)}\n```"
            )
            rows_invalid.append({
                "question": question,
                "answer": json.dumps(ground_truth),
                "info": json.dumps({
                    "task_type": "validation",
                    "note_path": note.get("path", ""),
                    "ground_truth": ground_truth,
                }),
            })
    rows = rows_clean + rows_invalid
    rng.shuffle(rows)
    return rows


def _default_diff_label() -> dict:
    return {"decision": "approve", "confidence": 0.8, "reason_code": "none",
            "reason_text": "Notion value is more current (default label)"}


def _label_diff_by_content(changes: list[dict]) -> dict:
    """Heuristic labeler for synthetic diff rows based on field patterns."""
    for c in changes:
        field = c.get("field", "").lower()
        notion_val = c.get("notion_value", "")
        if field == "hubspot_owner_id":
            return {"decision": "reject", "confidence": 0.95, "reason_code": "scope_violation",
                    "reason_text": "Owner assignment is human-only, not synced"}
        if not notion_val or notion_val.strip() == "":
            return {"decision": "reject", "confidence": 0.85, "reason_code": "missing_field",
                    "reason_text": "Notion value is empty, would clear HubSpot data"}
        if "<script" in notion_val.lower():
            return {"decision": "reject", "confidence": 0.95, "reason_code": "formatting",
                    "reason_text": "Content contains script injection"}
        if field == "dealstage" and "closedwon" in c.get("hubspot_value", "").lower():
            return {"decision": "reject", "confidence": 0.9, "reason_code": "wrong_stage",
                    "reason_text": "Reopening a closed-won deal is likely a mapping error"}
        if notion_val == c.get("hubspot_value", ""):
            return {"decision": "reject", "confidence": 0.7, "reason_code": "stale_source",
                    "reason_text": "Notion and HubSpot values are identical, no update needed"}
    # v0.3.0 defer class (was GT-unreachable: the prompt offered "defer" but no
    # synthetic row was ever labeled defer). Deterministic, prompt-stated rules:
    # open-stage advance or >50% amount change needs human confirmation.
    for c in changes:
        field = c.get("field", "").lower()
        hs_val = str(c.get("hubspot_value", "") or "").lower()
        no_val = str(c.get("notion_value", "") or "").lower()
        if field == "dealstage" and no_val and "closedwon" not in hs_val and no_val != hs_val:
            return {"decision": "defer", "confidence": 0.7, "reason_code": "none",
                    "reason_text": "Stage advance between open stages; confirm with the deal owner"}
        if field == "amount":
            try:
                old_amt, new_amt = float(hs_val), float(no_val)
                if old_amt > 0 and abs(new_amt - old_amt) / old_amt > 0.5:
                    return {"decision": "defer", "confidence": 0.7, "reason_code": "none",
                            "reason_text": "Amount change exceeds 50%; confirm before syncing"}
            except ValueError:
                pass
    return _default_diff_label()


def build_diff_rows(data: dict) -> list[dict[str, object]]:
    rows = []
    traj = data.get("trajectories", {})
    sync_trajs = traj.get("sync_agent", [])
    traj_decisions = {}
    for t in sync_trajs:
        reward = t.get("reward_signal", {})
        decision = reward.get("decision", "")
        if "approve" in decision.lower():
            for diff in t.get("diffs", []):
                traj_decisions[diff.get("deal_name", "")] = "approve"
        for rej in reward.get("rejected_details", []):
            traj_decisions[rej.get("record_id", "")] = "reject"
    for diff_note in data.get("sync_diffs", []):
        records = parse_sync_diff(diff_note)
        for rec in records:
            if not rec.get("changes"):
                continue
            changes_text = "\n".join(
                f"  {c['field']}: {c['hubspot_value']} → {c['notion_value']} ({c['action']})"
                for c in rec["changes"]
            )
            deal_name = rec["deal_name"]
            if deal_name in traj_decisions:
                decision = traj_decisions[deal_name]
                ground_truth = {
                    "decision": decision,
                    "confidence": 0.95,
                    "reason_code": "none" if decision == "approve" else "stale_source",
                    "reason_text": "Operator-verified decision from trajectory data",
                }
            else:
                ground_truth = _label_diff_by_content(rec["changes"])
            ground_truth["changed_fields"] = [c["field"] for c in rec["changes"] if c.get("field")]
            question = (
                f"{DIFF_SYSTEM_PROMPT}\n\n"
                f"Review this sync diff:\n\n"
                f"Deal: {deal_name}\n"
                f"HubSpot ID: {rec.get('hubspot_id', 'unknown')}\n"
                f"Changes:\n{changes_text}"
            )
            rows.append({
                "question": question,
                "answer": json.dumps(ground_truth),
                "info": json.dumps({
                    "task_type": "diff",
                    "deal_name": deal_name,
                    "ground_truth": ground_truth,
                }),
            })
    # v0.5.1: SCORER_V03_DESIGN §4.3.3 class rebalance (port of the dropped
    # v0.4 part-4). Base-rate exploits scale with class skew (p_class x
    # blind-reachable mass must stay under the 0.23 suite margin), so the
    # builder enforces the ~40/35/25 approve/reject/defer target AND caps any
    # single reject reason at 35% of rejects (a skewed reason prior re-opens
    # modal-reason guessing). Deterministic (seeded); question text unchanged.
    import random as _diff_rng
    _rng = _diff_rng.Random(13)
    by_class: dict[str, list[dict]] = {"approve": [], "reject": [], "defer": []}
    for row in rows:
        by_class[json.loads(row["info"])["ground_truth"]["decision"]].append(row)
    rejects = by_class["reject"]
    by_reason: dict[str, list[dict]] = {}
    for row in rejects:
        by_reason.setdefault(json.loads(row["info"])["ground_truth"]["reason_code"], []).append(row)
    if by_reason:
        cap = max(2, int(0.35 * len(rejects)))
        flat: list[dict] = []
        for grp in by_reason.values():
            _rng.shuffle(grp)
            flat.extend(grp[:cap])
        by_class["reject"] = flat
    target = {"approve": 0.40, "reject": 0.35, "defer": 0.25}
    n_max = min(int(len(by_class[c]) / f) for c, f in target.items() if f > 0)
    balanced: list[dict] = []
    for c, f in target.items():
        grp = by_class[c]
        _rng.shuffle(grp)
        balanced.extend(grp[: max(1, round(n_max * f))])
    _rng.shuffle(balanced)
    return balanced if balanced else rows


def build_extraction_rows(data: dict) -> list[dict[str, object]]:
    rows = []
    traj = data.get("trajectories", {})
    granola_trajs = traj.get("granola_notion_sync", [])
    for t in granola_trajs:
        outputs = t.get("outputs", {})
        inputs_data = t.get("inputs", {})
        transcript_summary = inputs_data.get("transcript_summary", "")
        if not transcript_summary:
            transcript_summary = json.dumps(inputs_data, indent=2)
        ground_truth = {
            "stage_change": outputs.get("stage_change"),
            "key_dates": outputs.get("key_dates", []),
            "action_items": outputs.get("action_items", []),
            "technical_requirements": outputs.get("technical_requirements", []),
            "objections": outputs.get("objections", []),
            "deal_fields_to_update": outputs.get("deal_fields_to_update", {}),
        }
        question = (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"Extract deal intelligence from this meeting transcript:\n\n"
            f"TRANSCRIPT:\n{transcript_summary}\n\n"
            f"CONTEXT: GPU compute infrastructure sales meeting."
        )
        rows.append({
            "question": question,
            "answer": json.dumps(ground_truth),
            "info": json.dumps({
                "task_type": "extraction",
                "ground_truth": ground_truth,
            }),
        })
    for i in range(len(_SYNTHETIC_TRANSCRIPTS)):
        row = _synthetic_extraction_row_at(i)
        # v0.3.0 generation invariant: every extraction row must have >= 2
        # non-empty GT fields so "predict nothing" never has a profitable row
        # (an all-empty-GT row paid the 0.10 empty cap for free).
        gt = json.loads(row["answer"])
        if sum(1 for v in gt.values() if v) < 2:
            continue
        rows.append(row)
    return rows


_SYNTHETIC_TRANSCRIPTS = [
    {
        "transcript": (
            "Meeting with Skylark - Q2 Expansion Discussion\n\n"
            "Avery: We can offer 256x B200 on the Skyforge cluster, available June 15.\n"
            "Skylark CTO: We need at least 128x, ideally 256x. B200 works for our "
            "post-training workloads. Main concern is the SLA - we need 99.9% uptime.\n"
            "Avery: Our standard CSA covers 99.5% with credits. We can discuss premium SLA.\n"
            "Skylark CTO: Pricing needs to be under $3.00/GPU/hr for a 12-month term.\n"
            "Avery: Let me work up a proposal by Friday. We'll include the premium SLA option.\n"
            "Skylark CTO: Also need to confirm NVLink fabric between nodes.\n"
            "Action items: Avery sends pricing proposal by Friday. Schedule technical "
            "deep-dive with infra team next week."
        ),
        "ground_truth": {
            "stage_change": {"from": "prospect", "to": "negotiation"},
            "key_dates": [{"event": "Pricing proposal due", "date": "2026-05-30"}],
            "action_items": ["Send pricing proposal by Friday", "Schedule technical deep-dive with infra team", "Confirm NVLink fabric between nodes", "Include premium SLA option in the proposal"],
            "technical_requirements": ["256x B200", "NVLink fabric", "99.9% uptime SLA", "12-month term"],
            "objections": ["Pricing must be under $3.00/GPU/hr", "Need premium SLA above standard 99.5%"],
            "deal_fields_to_update": {"gpu_type": "B200", "gpu_count": 256, "deal_status": "negotiation"},
        },
    },
    {
        "transcript": (
            "Renewal call with Vectorforge Labs\n\n"
            "Vectorforge PM: Our current 64x H100 cluster expires July 15. We want to renew "
            "but need to upgrade to B200 for our next training run.\n"
            "Avery: We can transition you to 128x B200 on NorthPeak infrastructure. Pricing would "
            "be $2.75/GPU/hr for a 6-month term.\n"
            "Vectorforge PM: That works if we can start by August 1. Can you guarantee the "
            "transition with zero downtime?\n"
            "Avery: We'll run both clusters in parallel for a week. I'll draft the CSA "
            "amendment today.\n"
            "Vectorforge PM: One more thing - we need 100TB NVMe storage attached.\n"
            "Avery: Noted. I'll confirm storage availability with infra by tomorrow."
        ),
        "ground_truth": {
            "stage_change": {"from": "customer", "to": "negotiation"},
            "key_dates": [{"event": "Current cluster expires", "date": "2026-07-15"}, {"event": "Target start", "date": "2026-08-01"}],
            "action_items": ["Draft CSA amendment today", "Confirm storage availability with infra by tomorrow"],
            "technical_requirements": ["128x B200", "100TB NVMe storage", "Zero-downtime transition", "6-month term"],
            "objections": ["Must start by August 1"],
            "deal_fields_to_update": {"gpu_type": "B200", "gpu_count": 128, "deal_status": "negotiation"},
        },
    },
    {
        "transcript": (
            "Discovery call with Quantex Trading\n\n"
            "Quantex VP: We're exploring two tracks. Track 1 is straightforward compute resale - "
            "we need 1,024 to 1,344 B300 GPUs for 3-4 years. Track 2 is a financial JV "
            "around GPU futures and credit intermediation.\n"
            "Avery: For Track 1, we have Cassiopeia B300 capacity coming online. Pricing "
            "floor is $3.75/GPU/hr.\n"
            "Quantex VP: Our target range is $3.00 to $3.50. There's a gap.\n"
            "Avery: Understood. Let me model a few structures - longer term could close "
            "the gap. On Track 2, I'll connect you with our CEO.\n"
            "Quantex VP: We also want to discuss GPU futures as a hedge product.\n"
            "Next steps: Avery sends term sheet options by end of week. CEO intro for Track 2."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "Term sheet options due", "date": "2026-05-30"}],
            "action_items": ["Send term sheet options by end of week", "Arrange CEO intro for Track 2 financial JV", "Model longer-term structures to close the pricing gap"],
            "technical_requirements": ["1024-1344x B300", "3-4 year term"],
            "objections": ["Pricing gap: target $3.00-3.50 vs floor $3.75"],
            "deal_fields_to_update": {"gpu_type": "B300", "gpu_count": 1024},
        },
    },
    {
        "transcript": (
            "Post-POC review with Cerebrix\n\n"
            "Cerebrix CTO: The POC on your post-training stack went well. We saw 40% "
            "speedup on our custom dev workloads compared to our current provider.\n"
            "Avery: Great to hear. Ready to discuss production terms?\n"
            "Cerebrix CTO: Yes, but we need a few things. First, dedicated support "
            "engineer during onboarding. Second, we want to start with 64x H200 and "
            "scale to 256x within 3 months.\n"
            "Avery: We can do a ramp schedule in the CSA. $2.50/GPU/hr for H200.\n"
            "Cerebrix CTO: That's within budget. Our board meets June 10 - we need "
            "the proposal before then.\n"
            "Avery: I'll have it to you by June 5."
        ),
        "ground_truth": {
            "stage_change": {"from": "poc", "to": "negotiation"},
            "key_dates": [{"event": "Board meeting", "date": "2026-06-10"}, {"event": "Proposal due", "date": "2026-06-05"}],
            "action_items": ["Send proposal by June 5", "Include ramp schedule 64x→256x in CSA"],
            "technical_requirements": ["64x H200 initial", "Scale to 256x in 3 months", "Dedicated support engineer"],
            "objections": [],
            "deal_fields_to_update": {"gpu_type": "H200", "gpu_count": 64, "deal_status": "negotiation"},
        },
    },
    {
        "transcript": (
            "Cold outreach follow-up with Kitsune\n\n"
            "Kitsune CEO: Thanks for reaching out. We're training a large language model "
            "and need 128x H100 urgently - can you deliver by June 1?\n"
            "Avery: We have freed capacity at NorthPeak Baldur. 128x H100 at $1.30/GPU/hr, "
            "12-month auto-renew.\n"
            "Kitsune CEO: $1.30 is great. We were quoted $2.30 elsewhere. What's the catch?\n"
            "Avery: No catch - this is recently freed capacity we're allocating quickly. "
            "I'll send the CSA today.\n"
            "Kitsune CEO: Send it. We'll review and sign by May 28 if terms are clean."
        ),
        "ground_truth": {
            "stage_change": {"from": "prospect", "to": "negotiation"},
            "key_dates": [{"event": "CSA review deadline", "date": "2026-05-28"}, {"event": "Target start", "date": "2026-06-01"}],
            "action_items": ["Send CSA today"],
            "technical_requirements": ["128x H100", "12-month auto-renew term"],
            "objections": [],
            "deal_fields_to_update": {"gpu_type": "H100", "gpu_count": 128, "deal_status": "negotiation"},
        },
    },
    {
        "transcript": (
            "Quarterly review with NordGrid supply team\n\n"
            "NordGrid PM: We have 512x B200 coming online at our Oslo facility in August. "
            "Pricing is $2.80/GPU/hr for a 2-year commitment.\n"
            "Avery: What's the networking setup?\n"
            "NordGrid PM: InfiniBand 400G, 8-node NVLink domains. Full liquid cooling.\n"
            "Avery: We have demand that could absorb this. Can you hold a soft reservation "
            "until we confirm the customer?\n"
            "NordGrid PM: We can hold for 30 days. After that it goes to market.\n"
            "Avery: Deal. I'll confirm by June 20."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "Capacity online", "date": "2026-08-01"}, {"event": "Reservation hold expires", "date": "2026-06-20"}],
            "action_items": ["Confirm customer match by June 20"],
            "technical_requirements": ["512x B200", "InfiniBand 400G", "8-node NVLink", "Liquid cooling", "2-year commitment"],
            "objections": ["30-day hold limit"],
            "deal_fields_to_update": {"gpu_type": "B200", "gpu_count": 512},
        },
    },
    {
        "transcript": (
            "Internal sync - Avery and Sam on Gantry situation\n\n"
            "Avery: Gantry is Day 4 without signing the CSA for 256x H100 at NorthPeak Baldur. "
            "I think we should treat this as freed capacity.\n"
            "Sam: Agreed. What's the plan for reallocation?\n"
            "Avery: Earmark 128x for Kitsune at $1.30/hr - they're ready to sign. "
            "The remaining 128x plus Quasar's 64x gives us 192x in the freed pool.\n"
            "Sam: What about the pricing delta? NorthPeak rate is $1.10 for Gantry.\n"
            "Avery: Kitsune at $1.30 is a $0.20/hr upgrade. 128 GPUs × $0.20 × 720 hours = "
            "$18K/month margin improvement.\n"
            "Sam: Do it. Get the Kitsune CSA out today."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [],
            "action_items": ["Send Kitsune CSA today", "Reallocate 128x H100 from Gantry to Kitsune"],
            "technical_requirements": ["128x H100"],
            "objections": [],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "Technical deep-dive with Polymesh AI\n\n"
            "Polymesh CTO: We're doing 3D generation inference and need low-latency GPU access. "
            "Ideally B200 with sub-millisecond inter-node communication.\n"
            "Avery: We have Corventus B200 capacity in US-West, 576 GPUs. Pricing at "
            "$2.60/GPU/hr.\n"
            "Polymesh CTO: We only need 64x initially but want to reserve option to scale "
            "to 256x. Can you do a 6-month term with scale-up rights?\n"
            "Avery: Yes, we can structure that. I'll add a scale-up clause to the CSA.\n"
            "Polymesh CTO: What about dedicated networking? We can't share fabric with "
            "other tenants.\n"
            "Avery: Corventus offers dedicated InfiniBand partitions. I'll confirm the "
            "isolation guarantees."
        ),
        "ground_truth": {
            "stage_change": {"from": "prospect", "to": "qualification"},
            "key_dates": [],
            "action_items": ["Add scale-up clause to CSA", "Confirm network isolation guarantees with Corventus"],
            "technical_requirements": ["64x B200 initial", "Scale to 256x option", "Sub-ms inter-node latency", "Dedicated InfiniBand partition", "6-month term with scale-up rights"],
            "objections": ["Cannot share network fabric with other tenants"],
            "deal_fields_to_update": {"gpu_type": "B200", "gpu_count": 64, "deal_status": "qualification"},
        },
    },
    {
        "transcript": (
            "Churn prevention call with Arcanum AI\n\n"
            "Arcanum VP: We're evaluating whether to renew the 512x H200 cluster. "
            "Our workloads shifted toward inference, so we need fewer GPUs but "
            "with lower latency.\n"
            "Avery: We can restructure to 128x B200, which gives better per-GPU "
            "inference throughput at $2.80/GPU/hr.\n"
            "Arcanum VP: The price point works. Our concern is the transition period. "
            "We need zero gaps in service.\n"
            "Avery: We'll overlap both clusters for two weeks. I'll draft the amendment "
            "by Wednesday.\n"
            "Arcanum VP: Also, our compliance team needs the SOC 2 report for the new DC."
        ),
        "ground_truth": {
            "stage_change": {"from": "customer", "to": "negotiation"},
            "key_dates": [{"event": "Amendment draft due", "date": "2026-06-11"}],
            "action_items": ["Draft CSA amendment by Wednesday", "Provide SOC 2 report for new DC"],
            "technical_requirements": ["128x B200", "Zero-gap transition", "Two-week cluster overlap"],
            "objections": ["Transition period: zero gaps required", "SOC 2 compliance required"],
            "deal_fields_to_update": {"gpu_type": "B200", "gpu_count": 128},
        },
    },
    {
        "transcript": (
            "Internal pipeline review: Avery and Jordan\n\n"
            "Jordan: What's the biggest deal at risk this week?\n"
            "Avery: Baseline Intelligence. 512x H100 at NorthPeak Baldur, $1.40/hr, "
            "auto-renew September 18. They asked about upgrading to B200 but "
            "the pricing delta is $1.10/hr.\n"
            "Jordan: Can we offer a blended rate? Keep them on H100 for inference "
            "and add a B200 partition for training.\n"
            "Avery: Good idea. I'll model a 256x H100 + 128x B200 hybrid. The "
            "blended rate would be about $1.85/hr.\n"
            "Jordan: Get them a proposal before the auto-renew triggers.\n"
            "Avery: On it. Also, Paper Kites wants to double from 256x to "
            "512x H100 at Sif. Same rate. Easy expansion.\n"
            "Jordan: Approve it. Any supply blockers?\n"
            "Avery: None, we have 1024x free at Sif through January."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "Baseline Intelligence auto-renew", "date": "2026-09-18"}],
            "action_items": ["Model hybrid 256x H100 + 128x B200 proposal for Baseline Intelligence", "Send proposal before auto-renew triggers", "Process Paper Kites expansion to 512x"],
            "technical_requirements": ["256x H100 + 128x B200 hybrid", "512x H100 expansion at Sif"],
            "objections": ["Pricing delta $1.10/hr for B200 upgrade"],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "First call with Subspace Labs\n\n"
            "Subspace CTO: We're building a multimodal foundation model and need "
            "serious compute. 2048x B300 GPUs, minimum 12-month commitment.\n"
            "Avery: That's a significant allocation. We have B300 capacity "
            "coming through our Cassiopeia partnership. Pricing starts at "
            "$3.60/GPU/hr for that scale.\n"
            "Subspace CTO: Our budget is $3.20. Can you sharpen the pencil?\n"
            "Avery: At 2048x for 12 months, we can explore volume pricing. "
            "I need to check with the vendor on a volume discount.\n"
            "Subspace CTO: We also need data sovereignty in the EU. No US routing.\n"
            "Avery: Cassiopeia has EU-North capacity. Let me confirm the exact "
            "location and compliance posture. I'll have answers by Friday."
        ),
        "ground_truth": {
            "stage_change": {"from": "cold", "to": "prospect"},
            "key_dates": [{"event": "Answers due to Subspace", "date": "2026-06-13"}],
            "action_items": ["Check vendor volume discount for 2048x B300", "Confirm EU-North location and compliance posture", "Respond by Friday"],
            "technical_requirements": ["2048x B300", "12-month term", "EU data sovereignty", "No US routing"],
            "objections": ["Budget target $3.20 vs offered $3.60"],
            "deal_fields_to_update": {"gpu_type": "B300", "gpu_count": 2048, "deal_status": "prospect"},
        },
    },
    {
        "transcript": (
            "Renewal close: Noeton/KJDR NVL72 cluster\n\n"
            "KJDR: The GB300 NVL72 cluster has been solid. We want to extend "
            "through March 2027.\n"
            "Avery: Great. Same terms, $4.20/GPU/hr, 72 GPUs?\n"
            "KJDR: Yes, but we want to add a second NVL72 rack in Q4 if the "
            "research goes well. Can you hold capacity?\n"
            "Avery: I can soft-reserve a second rack until October 15. After "
            "that it goes to market.\n"
            "KJDR: Fair. Send the renewal CSA today and we'll countersign "
            "this week."
        ),
        "ground_truth": {
            "stage_change": {"from": "customer", "to": "signed"},
            "key_dates": [{"event": "Renewal end", "date": "2027-03-01"}, {"event": "Second rack hold expires", "date": "2026-10-15"}],
            "action_items": ["Send renewal CSA today", "Soft-reserve second NVL72 rack until October 15"],
            "technical_requirements": ["72x GB300 NVL72", "Second NVL72 rack option"],
            "objections": [],
            "deal_fields_to_update": {"gpu_type": "GB300", "gpu_count": 72, "deal_status": "signed"},
        },
    },
    {
        "transcript": (
            "Lost deal debrief: Pyron AI\n\n"
            "Avery: Pyron went with Vertex Cloud. They cited 20% lower pricing "
            "and same-day provisioning.\n"
            "Sam: What was our offer?\n"
            "Avery: 16x B200 at $2.80/hr, 6-month term. Lambda offered $2.20 "
            "with no term commitment.\n"
            "Sam: Margin was already thin at $2.80. Nothing to do there.\n"
            "Avery: Agreed. I'll mark the deal as lost and move on. Their single-node "
            "H200 POC cluster should be offboarded too.\n"
            "Sam: Do it. Any lessons?\n"
            "Avery: For inference-only workloads under 32 GPUs, we can't compete "
            "with hyperscaler spot pricing. Our value is the 100+ GPU training "
            "segment with InfiniBand."
        ),
        "ground_truth": {
            "stage_change": {"from": "prospect", "to": "lost"},
            "key_dates": [],
            "action_items": ["Mark deal as lost", "Offboard the single-node H200 POC cluster"],
            "technical_requirements": ["16x B200"],
            "objections": ["20% lower pricing from competitor", "No term commitment from competitor"],
            "deal_fields_to_update": {"deal_status": "lost"},
        },
    },
    {
        "transcript": (
            "Technical scoping: Hosted Training for Cortex AI\n\n"
            "Cortex ML Lead: We want to evaluate your post-training stack for "
            "RLHF alignment work. Our current setup takes 3 days per iteration.\n"
            "Avery: Our hosted RL infrastructure can cut that to under 8 hours "
            "with 256x H200 and our optimized training framework.\n"
            "Cortex ML Lead: Impressive. What's the pricing model?\n"
            "Avery: $50K/month for a dedicated partition with our engineering "
            "support during the first 30 days.\n"
            "Cortex ML Lead: We'd want a POC first. Can you do 2 weeks at "
            "a reduced rate?\n"
            "Avery: Yes, $15K for a 2-week POC. If you convert, the POC cost "
            "applies against the first month.\n"
            "Cortex ML Lead: Send the POC agreement. Our VP needs to sign off "
            "by June 20."
        ),
        "ground_truth": {
            "stage_change": {"from": "prospect", "to": "poc"},
            "key_dates": [{"event": "VP sign-off deadline", "date": "2026-06-20"}],
            "action_items": ["Send POC agreement"],
            "technical_requirements": ["256x H200", "Post-training stack", "RLHF alignment", "30-day engineering support"],
            "objections": ["Wants POC before commitment"],
            "deal_fields_to_update": {"deal_status": "poc"},
        },
    },
    {
        "transcript": (
            "Capacity planning: Avery and supply team\n\n"
            "Avery: We have 3 deals closing this month that need B200 capacity: "
            "Subspace (2048x), Simile renewal (80x), and the Cortex HT POC (256x). "
            "Total demand is 2,384 GPUs.\n"
            "Supply Lead: Cassiopeia has 1,024x B200 coming online July 1 in "
            "Finland. Corventus has 576x in US-West, available now. That's "
            "1,600x total, short by 784.\n"
            "Avery: Can we pull forward the Ionstream B200 supply? Their 512x "
            "was earmarked for Q3.\n"
            "Supply Lead: Ionstream is pending the LOI. If they sign this week, "
            "we can allocate 512x by July 15.\n"
            "Avery: That gets us to 2,112. Still short 272. I'll check with "
            "E2E for spot capacity.\n"
            "Supply Lead: The E2E India cluster has 256x H200 free. Not B200, "
            "but it could bridge Subspace's ramp while Ionstream provisions.\n"
            "Avery: Good fallback. Let me run that by Subspace."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "Cassiopeia B200 online", "date": "2026-07-01"}, {"event": "Ionstream allocation if LOI signed", "date": "2026-07-15"}],
            "action_items": ["Follow up on Ionstream LOI this week", "Check E2E for spot B200 capacity", "Propose H200 bridge to Subspace"],
            "technical_requirements": ["2384x B200 total demand", "1024x Cassiopeia Finland", "576x Corventus US-West", "512x Ionstream pending"],
            "objections": ["784 GPU shortfall vs demand"],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "Contract amendment: Paper Kites expansion\n\n"
            "FA CTO: We're doubling our training cluster from 256x to 512x H100 "
            "at Sif. Same $1.25/GPU/hr rate.\n"
            "Avery: Confirmed, we have the capacity. I'll draft the amendment "
            "extending your allocation.\n"
            "FA CTO: Need it live by July 1. And add 100TB of shared NFS storage.\n"
            "Avery: Storage is $0.05/GB/month. I'll include it in the amendment.\n"
            "FA CTO: Perfect. Send it by EOD and we'll sign this week."
        ),
        "ground_truth": {
            "stage_change": {"from": "customer", "to": "signed"},
            "key_dates": [{"event": "Expansion go-live", "date": "2026-07-01"}],
            "action_items": ["Draft CSA amendment by EOD", "Include 100TB NFS storage"],
            "technical_requirements": ["512x H100", "100TB NFS storage"],
            "objections": [],
            "deal_fields_to_update": {"gpu_count": 512, "deal_status": "signed"},
        },
    },
    {
        "transcript": (
            "Vendor negotiation: NorthPeak Baldur pricing review\n\n"
            "NorthPeak Rep: We're adjusting H100 pricing from $1.10/GPU/hr to "
            "$1.25/GPU/hr for new contracts starting August.\n"
            "Avery: That's a 14% increase. Our margins on the existing clusters "
            "are already tight. Can we grandfather the current rate for renewals?\n"
            "NorthPeak Rep: We can hold $1.10 for existing customers renewing within "
            "30 days of contract end. New allocations are at the new rate.\n"
            "Avery: That works. I have 3 renewals coming up in the next 60 days. "
            "I'll prioritize those.\n"
            "NorthPeak Rep: Understood. We're also adding B300 capacity in Q4. "
            "Early commitment gets a 10% discount."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "NorthPeak pricing change effective", "date": "2026-08-01"}],
            "action_items": ["Prioritize 3 NorthPeak renewals before pricing change", "Evaluate B300 early commitment discount"],
            "technical_requirements": ["B300 capacity Q4"],
            "objections": ["14% H100 price increase"],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "Hosted training POC kickoff: Synapse AI\n\n"
            "Synapse ML Lead: We're starting the fine-tuning POC today. "
            "Our team needs SSH access to the 128x H200 partition.\n"
            "Avery: Access credentials were sent to your DevOps lead this morning. "
            "Our engineer will be on standby for the first 48 hours.\n"
            "Synapse ML Lead: Good. We're running LoRA fine-tuning on a "
            "7B model first, then scaling to 70B. Timeline is 2 weeks.\n"
            "Avery: After the POC, the production pricing is $45K/month "
            "for the same partition. What's your evaluation criteria?\n"
            "Synapse ML Lead: We need to beat our current provider's "
            "throughput by at least 20%. If we do, we commit for 6 months."
        ),
        "ground_truth": {
            "stage_change": {"from": "qualification", "to": "poc"},
            "key_dates": [{"event": "POC completion", "date": "2026-06-21"}],
            "action_items": ["Monitor POC performance vs 20% throughput target", "Keep engineer on standby for the first 48 hours of the POC"],
            "technical_requirements": ["128x H200", "SSH access", "LoRA fine-tuning", "7B then 70B scaling"],
            "objections": ["Must beat current provider throughput by 20%"],
            "deal_fields_to_update": {"gpu_type": "H200", "gpu_count": 128, "deal_status": "poc"},
        },
    },
    {
        "transcript": (
            "Cluster incident review: E2E India outage\n\n"
            "Avery: The hosted RL cluster in Delhi had a 4-hour network outage "
            "yesterday. What happened?\n"
            "E2E Support: A BGP misconfiguration during a firmware update. "
            "We've rolled back and added pre-update validation.\n"
            "Avery: This is the second incident in 3 months. Our SLA guarantees "
            "99.5% uptime. This month we're at 99.2%.\n"
            "E2E Support: We'll issue a credit for the affected hours. "
            "We're also adding redundant uplinks to prevent recurrence.\n"
            "Avery: I need a written RCA by Friday and the credit applied "
            "before the next invoice."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "RCA due", "date": "2026-06-13"}],
            "action_items": ["Get written RCA from E2E by Friday", "Confirm credit applied before next invoice"],
            "technical_requirements": ["Redundant uplinks"],
            "objections": ["Second incident in 3 months", "SLA at 99.2% vs 99.5% guaranteed"],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "New demand signal: Helix Research\n\n"
            "Helix CEO: We just closed our Series B. We're scaling from 32x "
            "to 256x H200 for our foundation model training.\n"
            "Avery: Congratulations. When do you need the capacity?\n"
            "Helix CEO: August 1 at the latest. We're evaluating two other "
            "providers, so pricing needs to be competitive.\n"
            "Avery: Our best rate for 256x H200 on a 12-month term is "
            "$2.40/GPU/hr. That includes dedicated InfiniBand.\n"
            "Helix CEO: We were quoted $2.20 elsewhere but without IB. "
            "Can you match $2.30 with IB included?\n"
            "Avery: I can do $2.30 for a 12-month commitment with IB. "
            "I'll send the term sheet today."
        ),
        "ground_truth": {
            "stage_change": {"from": "cold", "to": "negotiation"},
            "key_dates": [{"event": "Capacity needed by", "date": "2026-08-01"}],
            "action_items": ["Send term sheet today at $2.30/GPU/hr"],
            "technical_requirements": ["256x H200", "Dedicated InfiniBand", "12-month term"],
            "objections": ["Competing quote at $2.20 without IB"],
            "deal_fields_to_update": {"gpu_type": "H200", "gpu_count": 256, "deal_status": "negotiation"},
        },
    },
    {
        "transcript": (
            "Quarterly business review: Sam and Avery\n\n"
            "Sam: Give me the pipeline summary.\n"
            "Avery: Demand side: 47 active customers, 44 prospects, 6 in "
            "resource alignment, 3 signed this quarter. Supply side: 17 "
            "signed vendor deals, 15 at P1 stage, 1 negotiating. "
            "31 live clusters, 50K total GPUs.\n"
            "Sam: What's the biggest risk?\n"
            "Avery: Concentration. 60% of our GPU hours come from 4 clusters. "
            "If NorthPeak raises pricing or we lose Ricursive, the margin hit is "
            "significant.\n"
            "Sam: Mitigation?\n"
            "Avery: Diversifying to Corventus and Cassiopeia for B200. Both "
            "have competitive pricing and we have signed deals. I'm also "
            "pushing the Ionstream LOI for B300 backup.\n"
            "Sam: Good. What's the one thing you'd close this week?\n"
            "Avery: Baseline Intelligence hybrid proposal. 256x H100 plus "
            "128x B200, blended rate $1.85. They auto-renew September 18 "
            "and I want the upgrade locked before that triggers."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "Baseline Intelligence auto-renew", "date": "2026-09-18"}],
            "action_items": ["Lock Baseline Intelligence hybrid proposal before auto-renew", "Push Ionstream LOI for B300 backup"],
            "technical_requirements": ["256x H100 + 128x B200 hybrid"],
            "objections": ["60% GPU-hour concentration in 4 clusters"],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "Offboarding call: Doubleworld B200 cluster\n\n"
            "Doubleworld PM: We're not renewing the B200 cluster. Our project "
            "shifted to inference and we're moving to a managed service.\n"
            "Avery: Understood. The cluster expires March 16. Do you need any "
            "data migration support?\n"
            "Doubleworld PM: Yes, we have about 50TB to move to our S3 bucket. "
            "Can your team assist?\n"
            "Avery: We'll coordinate with Skyforge to stage the migration. "
            "I'll schedule the data transfer for the last week of the contract.\n"
            "Doubleworld PM: Thanks. No hard feelings, the cluster performed well."
        ),
        "ground_truth": {
            "stage_change": {"from": "customer", "to": "lost"},
            "key_dates": [{"event": "Cluster expiry", "date": "2026-03-16"}],
            "action_items": ["Coordinate 50TB data migration with Skyforge", "Schedule data transfer for last week of contract"],
            "technical_requirements": ["50TB S3 migration"],
            "objections": [],
            "deal_fields_to_update": {"deal_status": "lost"},
        },
    },
    {
        "transcript": (
            "Multi-cluster deal: Orbital Labs\n\n"
            "Orbital CTO: We need two separate clusters. Cluster A: 1024x H100 "
            "for pre-training, 24-month term. Cluster B: 256x B200 for "
            "post-training, 6-month term with renewal option.\n"
            "Avery: Cluster A at NorthPeak Baldur, $1.15/GPU/hr. Cluster B at "
            "Corventus US-West, $2.60/GPU/hr.\n"
            "Orbital CTO: Can we get a volume discount for committing to both?\n"
            "Avery: I can do a 5% bundle discount: H100 at $1.09, B200 at $2.47. "
            "Combined monthly value is about $1.9M.\n"
            "Orbital CTO: Run it by our finance team. We need the CSA by June 20 "
            "to hit our board deadline.\n"
            "Avery: Will do. I'll structure it as two linked CSAs with a "
            "master services agreement."
        ),
        "ground_truth": {
            "stage_change": {"from": "resource_alignment", "to": "negotiation"},
            "key_dates": [{"event": "CSA needed for board", "date": "2026-06-20"}],
            "action_items": ["Structure two linked CSAs with MSA", "Send to Orbital finance for approval"],
            "technical_requirements": ["1024x H100 24-month", "256x B200 6-month with renewal", "Bundle discount"],
            "objections": ["Wants volume discount for multi-cluster commitment"],
            "deal_fields_to_update": {"gpu_type": "H100", "gpu_count": 1024},
        },
    },
    {
        "transcript": (
            "Competitive intelligence: Vertex Cloud\n\n"
            "Avery: I lost the Pyron deal to Lambda. Their spot pricing is "
            "20% below ours for small inference workloads.\n"
            "Sam: Is that a pattern?\n"
            "Avery: Yes. Under 32 GPUs for inference, hyperscaler spot wins on "
            "price every time. We can't match it without going negative on margin.\n"
            "Sam: So we don't compete there. What's our defensible segment?\n"
            "Avery: 100+ GPU training with InfiniBand. Lambda can't do that at "
            "scale. Our win rate above 128 GPUs is over 80%.\n"
            "Sam: Formalize that as our ICP filter. No more sub-32 inference "
            "pursuits unless the customer has a training track.\n"
            "Avery: Agreed. I'll update the intake criteria."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [],
            "action_items": ["Update intake criteria: no sub-32 inference unless customer has training track", "Formalize ICP filter for 100+ GPU training segment"],
            "technical_requirements": [],
            "objections": [],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "Provisioning update: Cassiopeia B200\n\n"
            "Cassiopeia PM: The 16-node B200 cluster is racked and cabled. "
            "We're running burn-in tests this week.\n"
            "Avery: When can we expect it available for customer access?\n"
            "Cassiopeia PM: If burn-in passes, July 7. We had one node fail "
            "during initial power-on. Replacement arrives Thursday.\n"
            "Avery: I have the Inherent Labs deal earmarked for this cluster. "
            "Can you guarantee the July 7 date?\n"
            "Cassiopeia PM: Barring another hardware failure, yes. I'll send "
            "daily status updates starting Monday.\n"
            "Avery: Thanks. If it slips past July 10, I need to move Inherent "
            "to backup supply."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [{"event": "Cluster available for customer access", "date": "2026-07-07"}, {"event": "Fallback to backup supply if delayed", "date": "2026-07-10"}],
            "action_items": ["Monitor Cassiopeia burn-in daily starting Monday", "Prepare backup supply if cluster slips past July 10"],
            "technical_requirements": ["16-node B200 cluster"],
            "objections": ["One node failed during power-on, replacement arriving Thursday"],
            "deal_fields_to_update": {},
        },
    },
    {
        "transcript": (
            "Price benchmarking: Avery research notes\n\n"
            "Avery: Compiled current market rates from 6 providers.\n"
            "H100 IB: $1.10-1.60/hr (NorthPeak low, Denvr mid, VP high).\n"
            "H200 IB: $2.00-2.80/hr (E2E low, Skyforge mid).\n"
            "B200 IB: $2.40-3.20/hr (Corventus low, Cassiopeia mid).\n"
            "B300: $3.40-4.20/hr (Nebius only provider, limited supply).\n"
            "GB200 NVL72: $4.00-5.00/hr (GAI exclusive, premium).\n"
            "Our current sell rates sit at 30-40% markup over cost. "
            "B300 has the highest margin potential because supply is "
            "constrained and demand is growing fastest."
        ),
        "ground_truth": {
            "stage_change": None,
            "key_dates": [],
            "action_items": [],
            "technical_requirements": [],
            "objections": [],
            "deal_fields_to_update": {},
        },
    },
]


def _synthetic_extraction_row_at(idx: int) -> dict[str, object]:
    entry = _SYNTHETIC_TRANSCRIPTS[idx % len(_SYNTHETIC_TRANSCRIPTS)]
    return {
        "question": (
            f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"Extract deal intelligence from this meeting transcript:\n\n"
            f"TRANSCRIPT:\n{entry['transcript']}"
        ),
        "answer": json.dumps(entry["ground_truth"]),
        "info": json.dumps({"task_type": "extraction", "ground_truth": entry["ground_truth"]}),
    }


_NUM_TOKEN_RE = re.compile(r"\d[\d,.]*")


def _extract_number_tokens(text: str) -> set[str]:
    """Canonical standalone-number tokens in `text` (fabrication detection)."""
    out = set()
    for tok in _NUM_TOKEN_RE.findall(text or ""):
        t = tok.replace(",", "").rstrip(".")
        if not t:
            continue
        try:
            canon = str(float(t)) if "." in t else str(int(t))
        except ValueError:
            continue
        out.add(canon)
    return out


def _render_brief_fact_sheet(gt: dict, top_unearmarked: list, top_demand: list) -> str:
    """Compact verified-fact reference for the brief rubric judge (~200 tokens)."""
    lines = ["Verified facts for this brief:"]
    labels = [
        ("total_live_gpus", "Total live GPUs"),
        ("live_cluster_count", "Live clusters"),
        ("supply_deal_count", "Supply (vendor) deals"),
        ("demand_deal_count", "Demand (client) deals"),
        ("unearmarked_supply_count", "Unearmarked supply deals"),
        ("ht_deal_count", "Hosted-training deals"),
    ]
    for key, label in labels:
        val = gt.get(key)
        if val:
            lines.append(f"- {label}: {val}")
    if gt.get("nearest_cluster_expiry"):
        lines.append(f"- Nearest live-cluster contract expiry: {gt['nearest_cluster_expiry']}")
    if top_unearmarked:
        lines.append("Top unearmarked supply (earmark candidates):")
        for vd in top_unearmarked:
            fm = vd.get("frontmatter", {})
            lines.append(
                f"- {fm.get('name', '?')}: {fm.get('gpu_type', '?')} x{fm.get('gpu_count', '?')}, status={fm.get('deal_status', '?')}"
            )
    if top_demand:
        lines.append("Largest demand deals:")
        for cd in top_demand:
            fm = cd.get("frontmatter", {})
            lines.append(
                f"- {fm.get('name', '?')}: {fm.get('gpu_type', '?')} x{fm.get('gpu_count', '?')}, status={fm.get('deal_status', '?')}"
            )
    return "\n".join(lines)


def build_brief_rows(data: dict) -> list[dict[str, object]]:
    vendor_deals = data.get("vendor-deals", [])
    client_deals = data.get("client-deals", [])
    clusters = data.get("clusters", [])
    ht_deals = data.get("hosted-training-deals", [])

    live_clusters = [c for c in clusters if c.get("frontmatter", {}).get("cluster_status") == "LIVE"]
    total_live_gpus = sum(_safe_int(c["frontmatter"].get("gpu_count")) or 0 for c in live_clusters)

    supply_summary = "\n".join(
        f"- {vd['frontmatter'].get('name', 'unknown')}: "
        f"{vd['frontmatter'].get('gpu_type', '?')} x{vd['frontmatter'].get('gpu_count', '?')}, "
        f"status={vd['frontmatter'].get('deal_status', '?')}, "
        f"earmarked={vd['frontmatter'].get('earmarked_for', 'none')}"
        for vd in vendor_deals
    )
    demand_summary = "\n".join(
        f"- {cd['frontmatter'].get('name', 'unknown')}: "
        f"{cd['frontmatter'].get('gpu_type', '?')} x{cd['frontmatter'].get('gpu_count', '?')}, "
        f"status={cd['frontmatter'].get('deal_status', '?')}"
        for cd in client_deals[:30]
    )
    cluster_summary = (
        f"Total live clusters: {len(live_clusters)}, Total live GPUs: {total_live_gpus}\n"
        + "\n".join(
            f"- {c['frontmatter'].get('name', '?')}: "
            f"{c['frontmatter'].get('gpu_type', '?')} x{c['frontmatter'].get('gpu_count', '?')}, "
            f"provider={c['frontmatter'].get('provider', '?')}, "
            f"ends={c['frontmatter'].get('contract_end_date', '?')}"
            for c in sorted(
                live_clusters,
                key=lambda x: str(x.get("frontmatter", {}).get("contract_end_date") or "9999"),
            )[:15]
        )
    )
    ht_summary = "\n".join(
        f"- {h['frontmatter'].get('name', '?')}: "
        f"status={h['frontmatter'].get('ht_status', '?')}, "
        f"monthly={h['frontmatter'].get('targeted_land_monthly', '?')}"
        for h in ht_deals
    )

    user_msg = (
        f"Generate an operations brief from this vault data:\n\n"
        f"SUPPLY PIPELINE ({len(vendor_deals)} deals):\n{supply_summary}\n\n"
        f"DEMAND PIPELINE ({len(client_deals)} deals, showing top 30):\n{demand_summary}\n\n"
        f"ACTIVE CLUSTERS:\n{cluster_summary}\n\n"
        f"HOSTED TRAINING PIPELINE ({len(ht_deals)} deals):\n{ht_summary}"
    )

    unearmarked = [
        vd for vd in vendor_deals
        if not vd["frontmatter"].get("earmarked_for")
    ]
    nearest_expiry = None
    for c in live_clusters:
        end = _parse_date(c["frontmatter"].get("contract_end_date"))
        if end and (nearest_expiry is None or end < nearest_expiry):
            nearest_expiry = end

    ground_truth = {
        "total_live_gpus": total_live_gpus,
        "live_cluster_count": len(live_clusters),
        "supply_deal_count": len(vendor_deals),
        "demand_deal_count": len(client_deals),
        "unearmarked_supply_count": len(unearmarked),
        "nearest_cluster_expiry": str(nearest_expiry) if nearest_expiry else None,
        "ht_deal_count": len(ht_deals),
    }
    # v0.3.0 brief hardening inputs: every number present in the source tables
    # (fabrication detection) and a rendered fact sheet (judge reference).
    ground_truth["source_numbers"] = sorted(_extract_number_tokens(user_msg))
    top_unearmarked = sorted(
        unearmarked,
        key=lambda vd: _safe_int(vd["frontmatter"].get("gpu_count")) or 0,
        reverse=True,
    )[:3]
    top_demand = sorted(
        client_deals,
        key=lambda cd: _safe_int(cd["frontmatter"].get("gpu_count")) or 0,
        reverse=True,
    )[:3]
    ground_truth["fact_sheet"] = _render_brief_fact_sheet(ground_truth, top_unearmarked, top_demand)

    rows = [{
        "question": f"{BRIEF_SYSTEM_PROMPT}\n\n{user_msg}",
        "answer": json.dumps(ground_truth),
        "info": json.dumps({"task_type": "brief", "ground_truth": ground_truth}),
    }]
    focus_areas = [
        "Supply-focused brief: concentrate on supply pipeline status, vendor relationships, and availability windows. Include specific GPU counts and pricing.",
        "Demand-focused brief: concentrate on client pipeline, deal stages, competitive threats, and revenue at risk. Identify the highest-value deal to advance.",
        "Risk-focused brief: identify all coverage gaps, expiring contracts, unearmarked supply, and stale deals. Prioritize by dollar exposure.",
        "Leadership-focused brief: write for the CEO and Head of Growth. What decisions do they need to make this week? What intros or approvals are blocking deals?",
        "Renewal-focused brief: which contracts expire in the next 30 days? For each, state the client, GPU count, current rate, and whether a renewal conversation has started.",
        "Hosted training brief: focus on the HT pipeline. How many deals at each stage? What is the total targeted monthly revenue? Which POCs are closest to conversion?",
        "GPU-type brief: break down supply and demand by GPU type (H100, H200, B200, B300, GB200). Where are the gaps between available supply and client demand?",
        "Match-action brief: which supply-demand pairs should be earmarked today? List the top 3 uncommitted vendor deals and the client deal each should serve, with the match rationale.",
        "Weekly ops summary: what changed in the last 7 days? New deals, closed deals, clusters provisioned or offboarded, pricing changes. Written for the Monday standup.",
        "Margin analysis brief: for each active cluster, show the provider rate vs client rate and the per-GPU margin. Flag any cluster where margin is below 30%.",
    ]
    for focus in focus_areas:
        rows.append({
            "question": f"{BRIEF_SYSTEM_PROMPT}\n\nFOCUS: {focus}\n\n{user_msg}",
            "answer": json.dumps(ground_truth),
            "info": json.dumps({"task_type": "brief", "ground_truth": ground_truth}),
        })
    # v0.3.0 determinism fix: set iteration order is PYTHONHASHSEED-dependent, so
    # the same seed produced different GPU-filtered rows across processes.
    gpu_types_in_data = sorted({vd["frontmatter"].get("gpu_type", "") for vd in vendor_deals if vd["frontmatter"].get("gpu_type")})
    import random as _brief_rng
    _br = _brief_rng.Random(42)
    for gpu in gpu_types_in_data[:4]:
        filtered_vd = [vd for vd in vendor_deals if vd["frontmatter"].get("gpu_type") == gpu]
        filtered_cd = [cd for cd in client_deals if cd["frontmatter"].get("gpu_type") == gpu]
        if not filtered_vd and not filtered_cd:
            continue
        sub_supply = "\n".join(
            f"- {vd['frontmatter'].get('name','?')}: {vd['frontmatter'].get('gpu_count','?')}x, status={vd['frontmatter'].get('deal_status','?')}"
            for vd in filtered_vd
        )
        sub_demand = "\n".join(
            f"- {cd['frontmatter'].get('name','?')}: {cd['frontmatter'].get('gpu_count','?')}x, status={cd['frontmatter'].get('deal_status','?')}"
            for cd in filtered_cd[:15]
        )
        # v0.2.10: the GPU-filtered question shows ONLY the filtered supply and
        # demand deal lists — no cluster data and no earmark column — so
        # total_live_gpus, live_cluster_count, and unearmarked_supply_count are
        # not derivable from the question. Scoring them punished every model
        # (fact-recall denominators over unanswerable facts); zero them so the
        # scorer's `val > 0` guard skips them. Question text is unchanged.
        sub_gt = {
            "total_live_gpus": 0,
            "live_cluster_count": 0,
            "supply_deal_count": len(filtered_vd),
            "demand_deal_count": len(filtered_cd),
            "unearmarked_supply_count": 0,
            "nearest_cluster_expiry": None,
            "ht_deal_count": 0,
        }
        sub_q = f"{BRIEF_SYSTEM_PROMPT}\n\nFOCUS: {gpu}-only supply and demand analysis.\n\nSUPPLY ({len(filtered_vd)} {gpu} deals):\n{sub_supply}\n\nDEMAND ({len(filtered_cd)} {gpu} deals):\n{sub_demand}"
        sub_gt["source_numbers"] = sorted(_extract_number_tokens(sub_q))
        sub_gt["fact_sheet"] = _render_brief_fact_sheet(sub_gt, filtered_vd[:3], filtered_cd[:3])
        rows.append({
            "question": sub_q,
            "answer": json.dumps(sub_gt),
            "info": json.dumps({"task_type": "brief", "ground_truth": sub_gt}),
        })
    return rows


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def _repair_truncated_json(text: str) -> dict | None:
    """Best-effort recovery of a JSON object truncated mid-stream.

    Agents sometimes emit a complete-looking object whose tail is cut off.
    Track string/bracket state, close what is still open, drop a dangling
    trailing fragment, and retry json.loads.
    """
    closers = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == '"':
            in_string = True
    if not stack:
        return None
    variants = [text + '"' if in_string else text]
    if "," in text:
        cut = text[: text.rfind(",")]
        variants.append(cut + '"' if in_string else cut)
    variants.append(re.sub(r'"[^"]*"?$', "", text))
    closer_suffix = "".join(closers[c] for c in reversed(stack))
    for cand in variants:
        if not cand:
            continue
        try:
            obj = json.loads(cand + closer_suffix)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_model_json(completion: str) -> dict | None:
    if not completion:
        return None
    text = completion
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Truncated-output recovery: close what is still open and retry.
    return _repair_truncated_json(text)



def _score_match(response: str, gt: dict) -> float:
    parsed = _parse_model_json(response)
    if not parsed:
        # v0.3.0 (IC-OS-1 reward-shaping audit): the 0.05 '{'-floor paid JSON
        # spam free credit; unparseable output now pays zero.
        return 0.0
    gt_score = gt.get("match_score", 50)
    pred_score = parsed.get("match_score")
    if pred_score is None:
        return 0.1
    try:
        pred_score = float(pred_score)
    except (ValueError, TypeError):
        return 0.1
    # v0.2.0 hardening (audit fix 2): banded distance. The linear 1-|d|/100
    # reward paid constant-50 midpoint guessing 0.526 avg; bands require the
    # model to actually compute the rubric. |d|<=10 full, <=25 half, else 0.
    dist = abs(pred_score - gt_score)
    if dist <= 10:
        score_accuracy = 1.0
    elif dist <= 25:
        score_accuracy = 0.5
    else:
        score_accuracy = 0.0
    rec_score = 0.0
    gt_rec = gt.get("recommendation", "")
    pred_rec = parsed.get("recommendation", "")
    if pred_rec == gt_rec:
        rec_score = 0.2
    elif {pred_rec, gt_rec} <= {"earmark", "monitor"}:
        rec_score = 0.05
    dim_score = 0.0
    gt_dims = gt.get("dimensions", {})
    pred_dims = parsed.get("dimensions", {})
    if gt_dims and isinstance(pred_dims, dict) and pred_dims:
        # Rubric dimensions are discrete (0/5/10/15/25/30); only exact values
        # demonstrate the computation, so partial numeric closeness pays 0.
        exact = 0
        for key in gt_dims:
            try:
                if float(pred_dims.get(key, -1)) == float(gt_dims[key]):
                    exact += 1
            except (ValueError, TypeError):
                pass
        dim_score = exact / max(len(gt_dims), 1)
    # Gate: the dimensions fully determine score and recommendation, so band
    # and recommendation credit require demonstrating the computation. GT
    # scores cluster near the midpoint, so ungated band credit still paid
    # constant-50 ~0.44 avg.
    # v0.5: gate raised 0.5 -> 0.67 (V05_CEILING_MEMO QA). The 0.5 gate let
    # the consistent-modal-guess probe (const50_dims: modal dim values,
    # self-consistent score 50) through on the tier-1 window — 0.274 mean,
    # over the 0.25 ceiling (pre-existing red on main at v0.4.0). Requiring
    # 2/3 of dims exact restores the ceiling (probe 0.274 -> 0.105) without
    # touching goldens (0.95 mean unchanged, all perfect answers unaffected:
    # a computed rubric lands every dim).
    if dim_score < 0.67:
        return min(1.0, dim_score * 0.3)
    # v0.3.0: internal-consistency gate. The dimensions fully determine the
    # score, so a submitted score that disagrees with the submitted dims by
    # more than a rounding slop is band-guessing on top of copied dims; cap
    # at the failed-dims-gate level.
    try:
        pred_dim_sum = sum(float(v) for v in pred_dims.values())
    except (ValueError, TypeError):
        pred_dim_sum = None
    if pred_dim_sum is None or abs(pred_score - pred_dim_sum) > 2:
        # Stricter than the failed-dims cap: perfect copied dims with a
        # decoupled score is a pure copy attack (probe ceiling 0.25 > 0.3).
        return min(1.0, dim_score * 0.2)
    return min(1.0, score_accuracy * 0.5 + dim_score * 0.3 + rec_score)


_RULE_CATEGORIES = {
    "missing_required_field": "missing_field",
    "missing_field": "missing_field",
    "missing_note_type": "missing_field",
    "required_field": "missing_field",
    "empty_field": "missing_field",
    "missing_type_tag": "missing_tag",
    "missing_status_tag": "missing_tag",
    "missing_tag": "missing_tag",
    "no_type_tag": "missing_tag",
    "no_status_tag": "missing_tag",
    "tag_missing": "missing_tag",
    "missing_status_tag_for_deal_status": "status_mismatch",
    "unmapped_deal_status": "status_mismatch",
    "unmapped_client_status": "status_mismatch",
    "unmapped_cluster_status": "status_mismatch",
    "unmapped_ht_status": "status_mismatch",
    "deprecated_client_status": "status_mismatch",
    "status_mismatch": "status_mismatch",
    "invalid_status": "status_mismatch",
    "deal_status": "status_mismatch",
    # v0.2.11: model-produced rule name variants that the substring matcher
    # in _normalize_rule fails to catch because the model uses field names
    # ("ht_status enum") rather than internal vault_validate codes
    # ("unmapped_ht_status"). Without these, a semantically perfect validation
    # answer scores 0.0 because the rule name doesn't category-match.
    "ht_status": "status_mismatch",
    "ht_status_enum": "status_mismatch",
    "cluster_status": "status_mismatch",
    "cluster_status_enum": "status_mismatch",
    "client_status": "status_mismatch",
    "client_status_enum": "status_mismatch",
    "client_deal_status": "status_mismatch",
    "hosted_training_status": "status_mismatch",
    "invalid_ht_status": "status_mismatch",
    "invalid_cluster_status": "status_mismatch",
    "invalid_client_status": "status_mismatch",
    "invalid_deal_status": "status_mismatch",
    "unmapped_status": "status_mismatch",
    "enum_violation": "status_mismatch",
    # v0.5 (T-v05-2, 0.2.11 pattern): rule-6 phrasing variants the model
    # produces for "the status/ tag disagrees with deal_status" — without
    # these the same field-level defect rephrased normalizes into a
    # different category and union-dilutes category-F1 precision.
    "status_tag_mismatch": "status_mismatch",
    "tag_status_mismatch": "status_mismatch",
    "wrong_status_tag": "status_mismatch",
    "deal_status_tag": "status_mismatch",
    # Clean-split (2026-10-04): priority lives in `prio` (P1/P2/P3); its own category.
    "invalid_prio": "invalid_prio",
    "prio": "invalid_prio",
    "bad_prio": "invalid_prio",
    "priority": "invalid_prio",
    "slug_not_lowercase": "slug_format",
    "slug_has_spaces": "slug_format",
    "slug_format": "slug_format",
    "invalid_slug": "slug_format",
    "slug": "slug_format",
    "frontmatter_wikilink_not_in_body": "wikilink",
    "unidirectional_wikilink": "wikilink",
    "missing_wikilink": "wikilink",
    "wikilink": "wikilink",
    "broken_link": "wikilink",
}


def _normalize_rule(rule: str) -> str:
    rule_lower = rule.lower().strip().replace(" ", "_").replace("-", "_")
    if rule_lower in _RULE_CATEGORIES:
        return _RULE_CATEGORIES[rule_lower]
    for key, cat in _RULE_CATEGORIES.items():
        if key in rule_lower or rule_lower in key:
            return cat
    # v0.2.11: fallback — any rule mentioning a status field or "enum" with a
    # status-like substring is a status_mismatch. Catches model-produced names
    # like "ht_status enum", "cluster_status violation", "invalid client status".
    if any(s in rule_lower for s in ("_status", "deal_status", "ht_status",
                                     "cluster_status", "client_status", "status_enum",
                                     "status_violation", "status_invalid")):
        return "status_mismatch"
    if "enum" in rule_lower and "status" in rule_lower:
        return "status_mismatch"
    return rule_lower


def _score_validation(response: str, gt: dict) -> float:
    parsed = _parse_model_json(response)
    if not parsed:
        # v0.3.0 (IC-OS-1 reward-shaping audit): the 0.05 '{'-floor paid JSON
        # spam free credit; unparseable output now pays zero.
        return 0.0
    gt_valid = gt.get("valid", True)
    pred_valid = parsed.get("valid", True)
    # v0.2.0 hardening (audit fix 3): binary class credit capped at 0.15 (was
    # 0.3); when gt is invalid, violation-category overlap is required for ANY
    # credit; when gt is valid, substantive credit is gated on correctly
    # identifying the note_type (a blind constant JSON cannot know it).
    valid_match = 0.15 if pred_valid == gt_valid else 0.0
    gt_violations = gt.get("violations", [])
    pred_violations = parsed.get("violations", []) or []
    # v0.5 (T-v05-2): violations are scored as (category, normalized field)
    # instances — the shared basis for the category sets, the (category,
    # field) pair sets, and the field-wise pre-F1 dedupe below.
    def _violation_instances(viols):
        out = []
        for v in viols:
            if not isinstance(v, dict):
                continue
            rule = v.get("rule", "") or v.get("type", "") or v.get("violation", "")
            field = str(v.get("field", "") or "").lower().strip()
            if rule:
                out.append((_normalize_rule(str(rule)), field))
            elif field:
                msg = (v.get("message", "") or "").lower()
                if "tag" in field or "tag" in msg:
                    out.append(("missing_tag", field))
                elif "slug" in field or "slug" in msg:
                    out.append(("slug_format", field))
                elif "wikilink" in msg or "link" in msg:
                    out.append(("wikilink", field))
                elif "status" in msg:
                    out.append(("status_mismatch", field))
                else:
                    out.append(("missing_field", field))
        return out
    gt_instances = _violation_instances(gt_violations)
    pred_instances = _violation_instances(pred_violations)
    gt_categories = {c for c, _ in gt_instances}
    pred_categories = {c for c, _ in pred_instances}
    gt_note_type = str(gt.get("note_type", "")).lower().strip()
    pred_note_type = str(parsed.get("note_type", "")).lower().strip()
    note_type_ok = bool(gt_note_type) and pred_note_type == gt_note_type
    # v0.3.0 identity echo gate: any credit beyond the 0.15 binary cap
    # requires echoing the note's unique slug (a constant JSON cannot know a
    # per-row slug; note_type alone is guessable from class frequency).
    gt_slug = str(gt.get("slug", "") or "").lower().strip()
    pred_slug = str(parsed.get("slug", "") or "").lower().strip()
    slug_gate_ok = (pred_slug == gt_slug) if gt_slug else (pred_slug == "")
    has_slug_gate = "slug" in gt
    completeness_bonus = 0.0
    completeness = parsed.get("completeness_score")
    if completeness is not None:
        try:
            c = float(completeness)
            gt_c = float(gt.get("completeness_score", 0.5))
            completeness_bonus = 0.05 * max(0.0, 1.0 - abs(c - gt_c))
        except (ValueError, TypeError):
            pass
    if not gt_valid:
        # The note IS invalid: no credit of any kind without naming at least
        # one true violation category (kills the any-well-formed-JSON exploit).
        # v0.5 (T-v05-2) union-dilution fix: the same field-level defect
        # answered under two phrasings ("unmapped_deal_status_tag" ->
        # status_mismatch + "missing_status_tag" -> missing_tag) used to
        # normalize into two categories and halve category-F1 precision
        # (memo repro: 0.717 vs 0.95 for the identical single-phrasing
        # answer). Two scorer-side layers, invalid-row path only:

        # Layer 1 — field-wise pre-F1 dedupe: group predicted violations by
        # normalized field; within a group, if any candidate category matches
        # a GT category carried by that field keep the matching ones, else
        # keep one representative. Same-defect rephrasing collapses to one
        # prediction; cross-field shotgunning (distinct fields) keeps its
        # full precision penalty. Field-less violations carry independent
        # categories and are never collapsed.
        gt_cats_by_field = {}
        for gc, gf in gt_instances:
            gt_cats_by_field.setdefault(gf, set()).add(gc)
        deduped = [x for x in pred_instances if not x[1]]
        groups = {}
        for inst in pred_instances:
            if inst[1]:
                groups.setdefault(inst[1], []).append(inst)
        for fld, insts in groups.items():
            keep = [x for x in insts if x[0] in gt_cats_by_field.get(fld, ())]
            deduped.extend(keep if keep else insts[:1])
        pred_cats_d = {c for c, _ in deduped}

        # Layer 2 — conditional equivalence at match time (merge table):
        # pred missing_tag vs gt status_mismatch (and symmetric) counts as a
        # match when both violations' field is in (tags, deal_status)
        # territory — rule 4 (no status/ tag) and rule 6 (tag disagrees with
        # deal_status) legitimately overlap in model vocabulary. All other
        # pairs unchanged.
        def _cat_match(pc, pf, gc, gf):
            if pc == gc:
                return True
            if {pc, gc} == {"missing_tag", "status_mismatch"}:
                return pf in ("tags", "deal_status") and gf in ("tags", "deal_status")
            return False

        matched_pred = {
            pc for pc, pf in deduped
            if any(_cat_match(pc, pf, gc, gf) for gc, gf in gt_instances)
        }
        matched_gt = {
            gc for gc, gf in gt_instances
            if any(_cat_match(pc, pf, gc, gf) for pc, pf in deduped)
        }
        if not matched_gt:
            return 0.0
        if has_slug_gate and not slug_gate_ok:
            return valid_match
        precision = len(matched_pred) / len(pred_cats_d) if pred_cats_d else 0.0
        recall = len(matched_gt) / len(gt_categories) if gt_categories else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        score = valid_match + f1 * 0.6
        # v0.3.0 field-level credit: F1 over (category, field) pairs pushes
        # beyond category shotgunning (pair precision punishes sprayed
        # pairs). v0.5 (T-v05-2): pair matching uses the same merge table
        # (exact pair, or merge with both fields in territory); the
        # double-phrased rule-6 defect pairs both count.
        gt_pairs = {(c, f) for c, f in gt_instances if f}
        pred_pairs = {(c, f) for c, f in deduped if f}
        if gt_pairs and pred_pairs:
            def _pair_match(pp, gp):
                # Exact (category, field) pairs, or the merge table only —
                # a same-category pair on a DIFFERENT field still does not
                # match (pair precision is what punishes sprayed pairs).
                if pp == gp:
                    return True
                (pc, pf), (gc, gf) = pp, gp
                if {pc, gc} == {"missing_tag", "status_mismatch"}:
                    return pf in ("tags", "deal_status") and gf in ("tags", "deal_status")
                return False
            matched_pred_pairs = {
                pp for pp in pred_pairs if any(_pair_match(pp, gp) for gp in gt_pairs)
            }
            matched_gt_pairs = {
                gp for gp in gt_pairs if any(_pair_match(pp, gp) for pp in pred_pairs)
            }
            if matched_pred_pairs and matched_gt_pairs:
                pprec = len(matched_pred_pairs) / len(pred_pairs)
                prec_ = len(matched_gt_pairs) / len(gt_pairs)
                score += 0.1 * (2 * pprec * prec_ / (pprec + prec_))
        if note_type_ok:
            score += 0.05
        return min(1.0, score + completeness_bonus)
    # The note IS valid: full credit requires reading the note (note_type +
    # slug echo); spurious violation reports erode the cleanliness credit.
    clean = max(0.0, 1.0 - 0.5 * len(pred_categories))
    score = valid_match
    if has_slug_gate and not slug_gate_ok:
        return score
    if note_type_ok:
        score += 0.6 * clean + 0.1 + completeness_bonus
    return min(1.0, score)


def _score_diff(response: str, gt: dict) -> float:
    parsed = _parse_model_json(response)
    if not parsed:
        # v0.3.0 (IC-OS-1 reward-shaping audit): the 0.05 '{'-floor paid JSON
        # spam free credit; unparseable output now pays zero.
        return 0.0
    gt_decision = gt.get("decision", "approve")
    pred_decision = parsed.get("decision", "")
    # v0.2.0 hardening (audit fix 4): approve/defer adjacency credit cut
    # 0.3 -> 0.1; wrong-but-in-vocab credit removed; confidence and text
    # bonuses only pay on a correct decision; confidently-wrong answers are
    # penalized (calibration), floored at 0.
    # v0.2.1 evidence gate: decision credit requires referencing at least one
    # of the row's actual changed field names (a blind constant cannot). GT
    # skews reject, so ungated const-reject paid 0.42.
    changed = {str(f).lower().strip() for f in gt.get("changed_fields", []) if f}
    ev_mult = 1.0
    if changed:
        pred_fields = parsed.get("fields") or parsed.get("evidence_fields") or []
        if isinstance(pred_fields, list) and pred_fields:
            pf = {str(f).lower().strip() for f in pred_fields if f}
            overlap = pf & changed
            if not overlap:
                return 0.0
            # v0.3.0 evidence precision: the diff table shows exactly the
            # changed fields, so naming extras is field-name spraying (the
            # field vocabulary is small enough to enumerate blindly).
            ev_mult = len(overlap) / len(pf)
        else:
            mentioned = {f for f in changed if f in response.lower()}
            if not mentioned:
                return 0.0
            known_vocab = {"description", "deal_notes_secondary", "amount",
                           "dealstage", "hubspot_owner_id"}
            extra = {f for f in known_vocab - changed if f in response.lower()}
            ev_mult = len(mentioned) / max(len(mentioned) + len(extra), 1)
    correct = pred_decision == gt_decision
    # v0.5.1 blind-echo mitigation (port of the dropped v0.4 part-4 / 6464d1b,
    # af-47 echo-margin exception closure): the diff table makes changed-field
    # names question-echoable, so the evidence gate alone cannot stop
    # const-decision policies; per-class blind-reachable mass must satisfy
    # p_class x mass <= 0.23. (1) gt-approve rows pay the decision credit ONLY
    # (style bonuses on the majority-guessable class accrue 1:1 to blind
    # echo); (2) gt-reject rows COUPLE decision to reason: decision 0.3 +
    # exact reason 0.4 (family near-miss 0.15); conf/text pay only with the
    # exact reason; (3) approve/defer adjacency 0.1 -> 0.05. Ceilings:
    # approve 0.50, reject 0.85 (unchanged), defer 0.70. Within-group GRPO
    # ordering unaffected (per-class affine; V05_CEILING_MEMO §2).
    exact_reason = parsed.get("reason_code") == gt.get("reason_code")
    if gt_decision == "reject":
        decision_score = 0.3 if correct else 0.0
        reason_score = 0.0
        if correct:
            if exact_reason:
                reason_score = 0.4
            else:
                families = (
                    {"missing_field", "stale_source"},   # no-real-update evidence
                    {"formatting", "bad_diff"},          # malformed/unsafe content
                )
                pr, gr = str(parsed.get("reason_code", "")), str(gt.get("reason_code", ""))
                if any(pr in fam and gr in fam for fam in families):
                    reason_score = 0.15
        bonus_gate = correct and exact_reason
    elif gt_decision == "approve":
        decision_score = 0.5 if correct else (0.05 if pred_decision == "defer" else 0.0)
        reason_score = 0.0
        bonus_gate = False  # majority class: decision credit only
    else:  # defer
        decision_score = 0.5 if correct else (0.05 if pred_decision == "approve" else 0.0)
        reason_score = 0.05 if correct else 0.0
        bonus_gate = correct
    conf_score = 0.0
    try:
        pred_conf = float(parsed.get("confidence", 0.5))
        if bonus_gate:
            conf_score = 0.1 * min(pred_conf, 1.0)
        elif not correct and pred_conf > 0.7:
            conf_score = -0.05
    except (ValueError, TypeError):
        pass
    text_bonus = 0.05 if bonus_gate and parsed.get("reason_text") and len(str(parsed["reason_text"])) > 5 else 0.0
    return min(1.0, max(0.0, (decision_score + reason_score + conf_score + text_bonus) * ev_mult))


# --- Per-class scorer ceilings (V05_CEILING_MEMO main memo §3, v0.5) ---------
# The diff/validation ceiling asymmetry between row classes is principled —
# credit mass tracks each class's hidden verifiable-evidence capacity — and
# per-class renormalization is exactly gradient-invariant under std-normalized
# GRPO, so scoring NEVER rescales. These constants exist for the reporting
# layer only: dashboards/gates report mean(r_i / C_class(i)) alongside raw
# reward, and raw reward remains the gate currency within a scorer version.
DIFF_CEILINGS: dict[str, float] = {"approve": 0.70, "reject": 0.85, "defer": 0.70}
VALIDATION_CEILINGS: dict[str, float] = {"valid": 0.90, "invalid": 0.95}


def class_ceiling(task_type: str, gt: dict) -> float:
    """Oracle ceiling for one row's class (reporting-layer normalization only)."""
    if task_type == "diff":
        return DIFF_CEILINGS.get(str(gt.get("decision", "")), 0.85)
    if task_type == "validation":
        return VALIDATION_CEILINGS["valid" if gt.get("valid", True) else "invalid"]
    return 1.0


def _fuzzy_text_match(gt_item: str, pred_item: str, ratio_threshold: float = 0.45) -> bool:
    """Tolerant item matching for agent-extracted content.

    Agents paraphrase. Accept a match when the ground-truth item is contained in
    the prediction, the sequence similarity is high, or the ground-truth tokens
    are mostly present. Prevents punishing semantically-correct extractions
    that differ in phrasing.
    """
    import difflib

    g = str(gt_item).lower().strip()
    p = str(pred_item).lower().strip()
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    # v0.3.0 anchor requirement: when the ground-truth item carries a hard
    # token (number, GPU model, date — any digit-bearing token), a paraphrase
    # must share at least one of them before the similarity paths can match.
    # Generic prose ("confirm the requirements") stops matching number-bearing
    # GT items ("256x B200"), which is what made dump-spraying profitable.
    def _anchors(s: str) -> set[str]:
        out = set(re.findall(r"\d+(?:\.\d+)?", s))        # numbers / date parts
        out |= set(re.findall(r"[a-z]+\d+[a-z0-9]*", s))   # model tokens (b200, nvl72, h100)
        return out
    g_anchors = _anchors(g)
    p_anchors = _anchors(p)
    if g_anchors and not (g_anchors & p_anchors):
        return False
    # Echo remediation (af-47 ship condition for the GT-completeness bump):
    # GT items carrying >= 2 hard tokens must share >= 2 of them — a sentence
    # echo that happens to hit one number stops matching multi-anchor items.
    if len(g_anchors) >= 2 and len(g_anchors & p_anchors) < 2:
        return False
    # Length-ratio bound (tightened 2.5x+15 -> 2.0x+10, same condition): a
    # prediction much longer than the GT item is a sentence/paragraph echo,
    # not an extraction. Containment of the full GT string has already matched
    # above, so verbose phrasings that include the item verbatim are unaffected.
    if len(p) > 2.0 * len(g) + 10:
        return False
    if difflib.SequenceMatcher(None, g, p).ratio() >= ratio_threshold:
        return True
    gt_tokens = set(re.findall(r"[a-z0-9]+", g))
    pred_tokens = set(re.findall(r"[a-z0-9]+", p))
    if gt_tokens and len(gt_tokens & pred_tokens) / len(gt_tokens) >= 0.5:
        return True
    return False


# --- Extraction vocabulary normalization (v0.5.6) ----------------------------
#
# Ground truth uses the vault's controlled stage vocabulary ("customer",
# "prospect", "poc", ...); agents naturally emit free-form stage names
# ("Active contract", "Proposal Sent / Contract Review", "At-risk renewal").
# Map both sides onto shared canonical groups so semantically-correct stage
# reports score, with a fuzzy fallback for anything unmapped.

_STAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cold": ("cold", "outreach", "lead", "new", "uncontacted"),
    "prospect": ("prospect", "discovery", "qualification", "interest", "inquiry", "evaluating"),
    "qualification": ("qualification", "discovery", "scoping", "evaluating", "evaluation", "assess"),
    "poc": ("poc", "pilot", "proof of concept", "trial"),
    "negotiation": ("negotiation", "negotiate", "proposal", "term sheet", "contract review", "amendment", "restructure", "restructuring"),
    "resource_alignment": ("resource alignment", "alignment", "earmark", "allocation", "matching"),
    "customer": ("customer", "active", "existing", "renewal", "churn", "at-risk", "at risk", "retention", "renew"),
    "signed": ("signed", "sign", "closing", "closed won", "closed-won", "won", "verbal agreement", "committed", "countersign"),
    "lost": ("lost", "churned", "offboard", "not renewing", "competitor won", "closed lost", "closed-lost"),
}


def _stage_groups(name: object) -> set[str]:
    n = str(name).lower()
    return {
        group
        for group, keywords in _STAGE_KEYWORDS.items()
        if any(kw in n for kw in keywords)
    }


def _stage_match(gt_stage: object, pred_stage: object) -> bool:
    """Tolerant pipeline-stage comparison (controlled vocab vs free-form)."""
    g = str(gt_stage).lower().strip()
    p = str(pred_stage).lower().strip()
    if not g or not p:
        return False
    if g == p:
        return True
    if _stage_groups(g) & _stage_groups(p):
        return True
    return _fuzzy_text_match(g, p, ratio_threshold=0.5)


# deal_fields_to_update: ground truth uses canonical field names; agents
# paraphrase them ("gpu_count" vs "gpu_quantity" vs "gpu_allocation").
_FIELD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "gpu_type": ("gpu_type", "gpu_model", "product", "product_sku", "gpu", "hardware", "chip", "model", "sku"),
    "gpu_count": ("gpu_count", "gpu_quantity", "quantity", "volume", "gpu_allocation", "num_gpus", "gpu_total", "gpus", "allocation", "count"),
    "deal_status": ("deal_status", "deal_stage", "stage", "status", "pipeline_stage", "pipeline_status"),
}


def _find_pred_field(gt_key: str, pred_fields: dict) -> object | None:
    """Locate the agent's counterpart for a ground-truth deal field."""
    gk = str(gt_key).lower().strip()
    for key in pred_fields:
        if str(key).lower().strip() == gk:
            return key
    synonyms = _FIELD_SYNONYMS.get(gk, ())
    for key in pred_fields:
        kl = str(key).lower().strip()
        if any(s in kl or kl in s for s in synonyms if s):
            return key
    for key in pred_fields:
        if _fuzzy_text_match(gk, key, ratio_threshold=0.6):
            return key
    return None


def _value_match(gt_val: object, pred_val: object) -> bool:
    """Tolerant field-value comparison: containment, numeric, or fuzzy."""
    gs = str(gt_val).lower().strip()
    ps = str(pred_val).lower().strip()
    if not gs or not ps:
        return False
    if gs == ps:
        return True
    if gs in ps or ps in gs:
        return True
    # When the ground truth carries letters, the prediction must share one
    # ("B200" must not fuzz-match "H200" despite 3-of-4 character overlap).
    g_alpha = set(re.findall(r"[a-z]+", gs))
    p_alpha = set(re.findall(r"[a-z]+", ps))
    if g_alpha and not (g_alpha & p_alpha):
        return False
    # Purely-numeric ground-truth values match any pred value citing the
    # number (e.g. 256 vs "128 minimum, 256 ideal").
    if re.fullmatch(r"[\d.,]+", gs):
        g_nums = set(re.findall(r"\d+", gs.replace(",", "")))
        p_nums = set(re.findall(r"\d+", ps.replace(",", "")))
        if g_nums and g_nums & p_nums:
            return True
    return _fuzzy_text_match(gs, ps, ratio_threshold=0.6)


def _score_extraction(response: str, gt: dict) -> float:
    # v0.2.1 hardening: recall-only credit paid a generic 8-item dump 0.35 and
    # GT-empty branches paid an empty output 0.20. Now every list field scores
    # F1 (precision x recall), a field participates only if GT *or* the
    # prediction has content (fabrication counts against, silence on GT-empty
    # fields earns nothing), weights renormalize over participating fields,
    # and an all-empty response against an all-empty GT caps at 0.10.
    parsed = _parse_model_json(response)
    if not parsed:
        # v0.3.0 (IC-OS-1 reward-shaping audit): the 0.05 '{'-floor paid JSON
        # spam free credit; unparseable output now pays zero.
        return 0.0

    def _f1(gt_items: list, pred_items: list, matcher) -> float:
        if not gt_items or not pred_items:
            return 0.0
        recall_hits = sum(1 for g in gt_items if any(matcher(g, p) for p in pred_items))
        prec_hits = sum(1 for p in pred_items if any(matcher(g, p) for g in gt_items))
        recall = recall_hits / len(gt_items)
        precision = prec_hits / len(pred_items)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _as_list(v: Any) -> list:
        if isinstance(v, list):
            return v
        if v is None or v == "" or v == {}:
            return []
        return [v]

    components: list[tuple[float, float]] = []  # (weight, score)

    def add(weight: float, gt_items: list, pred_items: list, score: float) -> None:
        if gt_items or pred_items:
            components.append((weight, score))

    gt_actions = gt.get("action_items", []) or []
    pred_actions = _as_list(parsed.get("action_items"))
    add(0.25, gt_actions, pred_actions, _f1(gt_actions, pred_actions, _fuzzy_text_match))

    gt_stage = gt.get("stage_change")
    pred_stage = parsed.get("stage_change")
    if gt_stage or pred_stage:
        stage_score = 0.0
        if gt_stage and pred_stage:
            if not isinstance(gt_stage, dict):
                gt_stage = {"to": str(gt_stage)}
            if not isinstance(pred_stage, dict):
                pred_stage = {"to": str(pred_stage)}
            if _stage_match(gt_stage.get("to", ""), pred_stage.get("to", "")):
                stage_score = 1.0
            elif _stage_match(gt_stage.get("from", ""), pred_stage.get("from", "")):
                stage_score = 0.4
        components.append((0.2, stage_score))

    gt_reqs = gt.get("technical_requirements", []) or []
    pred_reqs = _as_list(parsed.get("technical_requirements"))
    add(0.2, gt_reqs, pred_reqs, _f1(gt_reqs, pred_reqs, _fuzzy_text_match))

    gt_fields = gt.get("deal_fields_to_update", {}) or {}
    pred_fields = parsed.get("deal_fields_to_update", {}) or {}
    if not isinstance(pred_fields, dict):
        pred_fields = {}
    if gt_fields or pred_fields:
        value_matched = 0
        for gk, gv in gt_fields.items():
            pk = _find_pred_field(gk, pred_fields)
            if pk is None:
                continue
            pv = pred_fields[pk]
            if "status" in str(gk).lower() or "stage" in str(gk).lower():
                ok = _stage_match(gv, pv)
            else:
                ok = _value_match(gv, pv)
            if ok:
                value_matched += 1
        if gt_fields and pred_fields:
            recall = value_matched / len(gt_fields)
            precision = value_matched / len(pred_fields)
            fields_score = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
        else:
            fields_score = 0.0
        components.append((0.2, fields_score))

    gt_objections = gt.get("objections", []) or []
    pred_objections = _as_list(parsed.get("objections"))
    add(0.1, gt_objections, pred_objections, _f1(gt_objections, pred_objections, _fuzzy_text_match))

    def _date_match(gd: Any, pd: Any) -> bool:
        if not isinstance(gd, dict):
            return False
        pd_str = str(pd)
        gdate = str(gd.get("date", ""))
        if gdate and gdate in pd_str:
            return True
        g_md = re.search(r"\d{4}-(\d{2})-(\d{2})", gdate)
        p_md = re.search(r"\d{4}-(\d{2})-(\d{2})", pd_str)
        if g_md and p_md and g_md.group(1, 2) == p_md.group(1, 2):
            return True
        if isinstance(pd, dict):
            ge = gd.get("event", "")
            pe = pd.get("event", "")
            if ge and pe and _fuzzy_text_match(ge, pe, ratio_threshold=0.4):
                return True
        return False

    gt_dates = gt.get("key_dates", []) or []
    pred_dates = _as_list(parsed.get("key_dates"))
    add(0.05, gt_dates, pred_dates, _f1(gt_dates, pred_dates, _date_match))

    if not components:
        # GT is empty on every field; an equally silent response is correct
        # but trivially so — cap at 0.10 (was additive-stacked to 0.20+).
        return 0.1
    total_w = sum(w for w, _ in components)
    return min(1.0, sum(w * s for w, s in components) / total_w)


def _brief_grounding(response: str, gt: dict) -> tuple[float, bool]:
    """Deterministic grounding score (0..0.35) + judge-gate verdict.

    v0.3.0 restructure per SCORER_V03_DESIGN §4.5: fact coverage and section
    credit are multiplied by a fabrication penalty computed against the row's
    known numbers (GT facts ∪ every number in the source tables), plus a
    number-count band. The gate controls the rubric-judge call: degenerate
    policies must fail it, so they cost $0 in judge spend and keep a hard
    reward ceiling even under a worst-case judge.

    v0.5 recalibration (T-v05-1, memo Option A): the v0.3 scorer was
    scorer-dead on real rollouts (0.00-0.14 regardless of quality) because
    the fabrication multiplier punished legitimate derived column sums and
    the absolute number band punished markdown tables. Response is
    comma-normalized; fabrication counts against known ∪ subset-sums of the
    integer source numbers; the absolute band is replaced by the 3-part
    conjunction band (run <= 8 AND density <= 0.35 AND tokens <= 0.6 x
    |known|), applied identically in the multiplier and the gate.
    """
    if not response or len(response) < 20:
        return 0.0, False
    # v0.5 (T-v05-1): comma-normalize the response before fact matching and
    # number extraction — "3,936" must match GT fact 3936, count as ONE
    # numeric token, and not split into fabricated-looking fragments (memo:
    # the comma fix alone regains fact hits on 4/15 real rollouts).
    response = re.sub(r"(?<=\d),(?=\d)", "", response)
    response_lower = response.lower()
    # Fact grounding: GT numbers must appear as standalone numbers, not as
    # digit substrings of unrelated numbers.
    num_checks = 0
    num_hits = 0
    for key in ["total_live_gpus", "unearmarked_supply_count", "supply_deal_count", "demand_deal_count", "live_cluster_count", "ht_deal_count"]:
        val = gt.get(key)
        if val and val > 0:
            num_checks += 1
            if re.search(rf"(?<![\d.]){re.escape(str(val))}(?![\d])", response):
                num_hits += 1
    nearest_expiry = gt.get("nearest_cluster_expiry")
    if nearest_expiry:
        num_checks += 1
        if nearest_expiry in response:
            num_hits += 1
    fact_score = 0.25 * (num_hits / num_checks) if num_checks else 0.0
    grounded = num_hits >= 1 or num_checks == 0
    # Fabrication penalty: numbers not present in the GT facts or anywhere in
    # the row's source tables are fabricated. Spraying the integer space
    # (1..59 + powers of two paid 0.69 on v0.2.0) now zeroes the multiplier.
    known = set(gt.get("source_numbers") or [])
    for key in ["total_live_gpus", "unearmarked_supply_count", "supply_deal_count", "demand_deal_count", "live_cluster_count", "ht_deal_count"]:
        if gt.get(key):
            known.add(str(gt[key]))
    if nearest_expiry:
        known |= _extract_number_tokens(str(nearest_expiry))
    resp_numbers = _extract_number_tokens(response)
    # v0.5 (T-v05-1): derived-sum closure — legitimate derived arithmetic
    # (column totals per GPU type) is not fabrication. Fabricated numbers
    # count against known UNION subset-sums of the integer source numbers
    # (deterministic DP bitset: <=60 integer values, 2e5 sum bound,
    # microseconds). Measured on the memo's 15 real rollouts: fab mean
    # 0.62 -> 0.14, fab-gate pass 2/15 -> 11/15. Multiplicative figures
    # (MRR = count x rate x hours) stay uncovered — accepted residual.
    if resp_numbers and known:
        sums_mask = 1
        for v in sorted({int(t) for t in known if t.isdigit()})[:60]:
            if 0 < v <= 200_000:
                sums_mask |= sums_mask << v
                sums_mask &= (1 << 200_001) - 1
        def _covered(t: str) -> bool:
            if t in known:
                return True
            try:
                f = float(t)
            except ValueError:
                return False
            return f.is_integer() and 0 <= f <= 200_000 and (sums_mask >> int(f)) & 1 == 1
        fab_rate = sum(1 for t in resp_numbers if not _covered(t)) / len(resp_numbers)
    else:
        fab_rate = 0.0 if not resp_numbers else (1.0 if not known else 0.0)
    fab_mult = 1.0 if fab_rate <= 0.2 else max(0.0, 1.0 - 2.0 * (fab_rate - 0.2))
    # v0.5 (T-v05-1): the absolute number-count band (cap 40) punished real
    # markdown-table analyses (51-146 numeric tokens on the memo's rollouts).
    # It is replaced by a 3-part conjunction band — a naive run-length-only
    # band was defeated by word-interleaved known spray (run length 1), so
    # all three independent signals must hold (memo riding-armor):
    #   (a) longest consecutive numeric-token run <= 8 (tables run 2-3;
    #       plain sprays run 60-80);
    #   (b) numeric-token density <= 0.35 of tokens (real table rollouts
    #       measure 0.10-0.30; interleaved spray ~0.4-0.5);
    #   (c) numeric tokens <= 0.6 x |known| (generalizes the v0.4.0
    #       small-table guard: citing most of the source table is table echo,
    #       which also defeats padded interleaved spray).
    words = re.findall(r"[a-z0-9']+", response_lower)
    total_words = max(len(words), 1)
    numeric_tokens = [w for w in words if re.fullmatch(r"\d[\d,.]*", w)]
    max_run = run = 0
    for w in words:
        if re.fullmatch(r"\d[\d,.]*", w):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    numeric_density = len(numeric_tokens) / total_words
    cover_cap = max(4, int(0.6 * len(known))) if known else 40
    # v0.5.2 (RB7 finding, af-47-gated): the coverage leg counts UNIQUE
    # known-member tokens (fab-side tokenization, BASE known set — closure-
    # derived numbers are fab's concern, not echo's). The 0.5.1 leg counted
    # every numeric token, so table-style analytic briefs that cite known
    # numbers repeatedly read as echo (4 substantive RB7 rollouts with fab
    # 0.02-0.04 were killed solely here; 2 restore under unique counting, the
    # other 2 cite >=60% of the table's DISTINCT values = majority-table
    # reproduction, intentionally still failed — judge-mode resolves those).
    # Full sprays cite 100% of known by construction, so every spray probe
    # still fails the band; cap 0.8 was measured and REJECTED (a 75% sub-cap
    # interleaved spray passed the judge gate 14/15 — see known_subcap_spray
    # probe, which pins the 55% variant at 0.169 / 2 gate passes under 0.6).
    unique_known_cited = len(_extract_number_tokens(response.replace(",", "")) & known)
    band_ok = max_run <= 8 and numeric_density <= 0.35 and unique_known_cited <= cover_cap
    band_mult = 1.0 if band_ok else 0.2
    section_keywords = {
        "supply": ["supply", "vendor", "provider"],
        "demand": ["demand", "client", "pipeline", "deal"],
        "priority": ["priority", "match", "earmark", "urgent"],
        "gaps": ["gap", "unearmarked", "coverage", "risk", "expir"],
        "leadership": ["leadership", "brief", "decision", "action"],
    }
    section_score = 0.0
    if grounded:
        for section, keywords in section_keywords.items():
            if any(kw in response_lower for kw in keywords):
                section_score += 0.02
    # Keyword-density backstop on the section credit (fact gating already
    # zeroes pure stuffing; threshold sits above natural domain-word density).
    kw_occurrences = sum(
        response_lower.count(kw)
        for kws in section_keywords.values()
        for kw in kws
    )
    density = kw_occurrences / total_words
    density_mult = 1.0 if density <= 0.35 else max(0.1, 1.0 - 3.0 * (density - 0.35))
    grounding = (fact_score + section_score * density_mult) * fab_mult * band_mult
    # v0.5 (T-v05-1): the gate uses the same 3 band signals so the
    # word-interleaved known-number spray fails the judge gate on every row
    # (a gate still keyed on the retired absolute cap would pass it once the
    # cap is gone).
    gate = (
        num_hits >= min(2, max(num_checks, 1))
        and fab_rate <= 0.3
        and band_ok
        and 150 <= len(response) <= 6000
    )
    return min(0.35, grounding), gate


def _score_brief(response: str, gt: dict) -> float:
    """Deterministic brief score (no-judge mode): the grounding floor, max 0.35.

    Deliberately NOT renormalized to 1.0: surface grounding cannot
    distinguish a good brief from garbage that embeds the same two facts
    (judge_sycophancy probe paid 0.51 renormalized), so the deterministic
    regime only ever pays the grounding floor. With the rubric judge enabled
    (PrimeGrwTaskConfig.judge_enabled) the task scores
    grounding + 0.65 × judge — see PrimeGrwTask.reward.
    """
    grounding, _ = _brief_grounding(response, gt)
    return grounding


# ---------------------------------------------------------------------------
# Schema drift detection
# ---------------------------------------------------------------------------

def check_schema_drift(vault_root: Path | None) -> None:
    if vault_root is None:
        return
    mapping_path = vault_root / MAPPING_MASTER_REL
    if not mapping_path.exists():
        return
    text = mapping_path.read_text(encoding="utf-8", errors="replace")
    version_match = re.search(r"version:\s*v?([\d.]+)", text)
    if version_match:
        current_version = version_match.group(1)
        pinned_version = "1.3"
        if current_version != pinned_version:
            warnings.warn(
                f"Mapping master drifted: pinned v{pinned_version}, "
                f"current v{current_version}. Reward functions may score "
                f"against stale schema.",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Dataset builder (classic pattern, compatible with all verifiers versions)
# ---------------------------------------------------------------------------

def _resolve_vault_root(vault_root_override: str | None = None) -> Path | None:
    if vault_root_override:
        p = Path(vault_root_override)
        if p.exists():
            return p
    env_root = os.environ.get("AF_VAULT_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    if DEFAULT_VAULT_ROOT is not None and DEFAULT_VAULT_ROOT.exists():
        return DEFAULT_VAULT_ROOT
    return None


def _row_hash(row: dict) -> int:
    q = row.get("question", "")
    return hash(q[:200])


def build_dataset(
    data_source: str = "synthetic",
    vault_root: str | None = None,
    task_types: list[str] | None = None,
    rows_per_task_type: int = 40,
    seed: int = 42,
    split: str = "train",
):
    import random
    from datasets import Dataset

    if task_types is None:
        task_types = ["match", "validation", "diff", "extraction", "brief"]

    if data_source == "synthetic":
        data = _generate_synthetic_data(seed)
    else:
        resolved = _resolve_vault_root(vault_root)
        data = load_vault_data(resolved) if resolved else _generate_synthetic_data(seed)

    rng = random.Random(seed)
    all_rows: list[dict[str, object]] = []
    mixture = {"match": 0.35, "validation": 0.40, "diff": 0.10, "extraction": 0.10, "brief": 0.05}
    # v0.1.0 lineage note: mixture rebalanced per Stage-5 gate FAIL (2026-10-04, eval bi0ha7yhc4w8jmdpf80vbi05):
    # trained 9B collapsed on validation (0.309, -0.490 vs baseline) and regressed on match (-0.217)
    # while extraction (+0.187) and brief (+0.040) exceeded baseline -> oversample the weak, trim the learned.
    builders = {
        "match": build_match_rows,
        "validation": build_validation_rows,
        "diff": build_diff_rows,
        "extraction": build_extraction_rows,
        "brief": build_brief_rows,
    }
    for task_type in task_types:
        builder_fn = builders.get(task_type)
        if not builder_fn:
            continue
        cap = max(5, int(rows_per_task_type * 5 * mixture.get(task_type, 0.2)))
        task_rows = builder_fn(data)
        if len(task_rows) > cap:
            task_rows = rng.sample(task_rows, cap)
        all_rows.extend(task_rows)
    rng.shuffle(all_rows)

    if not all_rows:
        all_rows = [_synthetic_extraction_row_at(0)]

    if split in ("train", "eval"):
        indexed = list(enumerate(all_rows))
        rng2 = random.Random(seed + 1)
        rng2.shuffle(indexed)
        cut = max(1, len(indexed) * 4 // 5)
        train_idx = {i for i, _ in indexed[:cut]}
        train_rows = [r for i, r in enumerate(all_rows) if i in train_idx]
        eval_rows = [r for i, r in enumerate(all_rows) if i not in train_idx]
        if split == "eval":
            return Dataset.from_list(eval_rows if eval_rows else all_rows[:5])
        return Dataset.from_list(train_rows if train_rows else all_rows)

    return Dataset.from_list(all_rows)


# ---------------------------------------------------------------------------
# v1 taskset (verifiers.v1: TaskData + Task + TasksetConfig + Taskset)
# ---------------------------------------------------------------------------

_DEFAULT_TASK_TYPES: tuple[TaskType, ...] = ("match", "validation", "diff", "extraction", "brief")
"""Immutable default task mixture; mirrors the legacy load_environment default."""




# v0.2.11 (reconciled): typed ground truth models per task type. Declarative
# documentation of each task's GT contract; enforced in tier-1 CI
# (tests/test_probes.py::test_gt_models_validate), not at build time, so a
# schema bug fails CI without crashing a hosted run. Field types corrected in
# the v0.3.0 reconciliation to match the actual builders (stage_change is a
# dict, action_items are strings, deal_fields_to_update is a dict) and
# extended with the v0.3 brief fields (source_numbers, fact_sheet).


class MatchGT(BaseModel):
    """Ground truth for match scoring tasks."""

    match_score: int = Field(ge=0, le=100)
    dimensions: dict[str, int] = Field(default_factory=dict)
    recommendation: str = Field(default="pass")


class ValidationGT(BaseModel):
    """Ground truth for schema validation tasks."""

    valid: bool
    violations: list[dict[str, Any]] = Field(default_factory=list)
    note_type: str = Field(default="unknown")
    slug: str = Field(default="")
    completeness_score: float = Field(default=0.5, ge=0.0, le=1.0)


class DiffGT(BaseModel):
    """Ground truth for diff classification tasks."""

    decision: Literal["approve", "reject", "defer"] = "approve"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason_code: str = Field(default="none")
    reason_text: str = Field(default="")
    changed_fields: list[str] = Field(default_factory=list)


class ExtractionGT(BaseModel):
    """Ground truth for transcript extraction tasks."""

    stage_change: dict[str, Any] | str | None = None
    key_dates: list[dict[str, Any]] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    technical_requirements: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    deal_fields_to_update: dict[str, Any] = Field(default_factory=dict)


class BriefGT(BaseModel):
    """Ground truth for ops brief generation tasks."""

    total_live_gpus: int = Field(default=0)
    live_cluster_count: int = Field(default=0)
    supply_deal_count: int = Field(default=0)
    demand_deal_count: int = Field(default=0)
    unearmarked_supply_count: int = Field(default=0)
    nearest_cluster_expiry: str | None = None
    ht_deal_count: int = Field(default=0)
    source_numbers: list[str] = Field(default_factory=list)
    fact_sheet: str = Field(default="")


# Map task type strings to GT model classes
GT_MODELS: dict[str, type] = {
    "match": MatchGT,
    "validation": ValidationGT,
    "diff": DiffGT,
    "extraction": ExtractionGT,
    "brief": BriefGT,
}

# Task-type system prompts, separated from task content at load time (v1
# taskset model: system instructions ride TaskData.system_prompt, not prompt).
SYSTEM_PROMPT_BY_TYPE = {
    "match": MATCH_SYSTEM_PROMPT,
    "validation": VALIDATION_SYSTEM_PROMPT,
    "diff": DIFF_SYSTEM_PROMPT,
    "extraction": EXTRACTION_SYSTEM_PROMPT,
    "brief": BRIEF_SYSTEM_PROMPT,
}


# v0.2.11: probe suite as a permanent CI release gate
# No degenerate policy should score > 0.25 on any task type.
# Callable by the Hub release pipeline before publishing a new version.


def run_probes() -> dict[str, float]:
    """Run degenerate-policy probes against all task type scorers.

    Returns a dict mapping ``f"{task_type}:{policy_name}" -> max_reward``
    across a set of representative GTs. The release gate checks that all
    values are < 0.25.

    Usage:
        from prime_grw import run_probes
        results = run_probes()
        assert all(v < 0.25 for v in results.values()), f"Probe failures: {results}"
    """
    # Representative GTs for each task type
    _probe_gts = {
        "match": [
            {"match_score": 85, "dimensions": {"gpu_type": 25, "scale": 20, "timing": 10, "term": 15, "geography": 10, "price": 5}, "recommendation": "earmark"},
            {"match_score": 80, "dimensions": {"gpu_type": 8, "scale": 7, "timing": 6, "term": 8, "geography": 9, "price": 7}, "recommendation": "earmark"},
        ],
        "validation": [
            {"valid": False, "violations": [{"rule": "unmapped_ht_status", "field": "deal_status", "message": "unmapped_ht_status"}], "note_type": "hosted-training-deal", "slug": "test-ht-01", "completeness_score": 1.0},
            {"valid": False, "violations": [{"rule": "unmapped_cluster_status", "field": "cluster_status", "message": "unmapped_cluster_status"}], "note_type": "cluster", "slug": "test-cluster-01", "completeness_score": 0.8},
            {"valid": True, "violations": [], "note_type": "vendor-deal", "slug": "aurora-us-east", "completeness_score": 1.0},
        ],
        "diff": [
            {"decision": "reject", "confidence": 0.85, "reason_code": "missing_field", "reason_text": "empty value", "changed_fields": ["description"]},
            {"decision": "approve", "confidence": 0.9, "reason_code": "none", "reason_text": "valid update", "changed_fields": ["name", "description"]},
        ],
        "extraction": [
            {"stage_change": "prospect_to_negotiating", "key_dates": [{"date": "2026-10-15", "event": "follow-up"}], "action_items": [{"item": "send proposal"}], "technical_requirements": [{"gpu": "H100", "count": 64}], "objections": [{"concern": "price"}], "deal_fields_to_update": [{"field": "deal_status", "value": "negotiating"}]},
            {"stage_change": None, "key_dates": [], "action_items": [], "technical_requirements": [], "objations": [], "deal_fields_to_update": []},
        ],
        "brief": [
            {"total_live_gpus": 0, "live_cluster_count": 0, "supply_deal_count": 3, "demand_deal_count": 2, "unearmarked_supply_count": 0, "nearest_cluster_expiry": None, "ht_deal_count": 0},
        ],
    }

    # Degenerate policies (zero-understanding strategies)
    _degenerate = {
        "match": [
            '{"match_score":50,"dimensions":{"gpu_type":5,"scale":5,"timing":5,"term":5,"geography":5,"price":5},"recommendation":"monitor"}',
        ],
        "validation": [
            '{"valid":false,"violations":[{"rule":"status_mismatch"}],"note_type":"vendor-deal","completeness_score":0.5}',
            '{"valid":false,"violations":[{"rule":"ht_status enum"}],"note_type":"vendor-deal","completeness_score":0.5}',
            '{"valid":true,"violations":[],"note_type":"vendor-deal","completeness_score":0.5}',
            '{"valid":false,"violations":[{"rule":"missing_field"},{"rule":"status_mismatch"},{"rule":"slug_format"},{"rule":"wikilink"}],"note_type":"vendor-deal","completeness_score":0.5}',
        ],
        "diff": [
            '{"decision":"approve","confidence":0.5,"reason_code":"none","reason_text":"ok","changed_fields":["status"]}',
            '{"decision":"reject","confidence":0.5,"reason_code":"missing_field","reason_text":"empty","changed_fields":["status"]}',
        ],
        "extraction": [
            '{"stage_change":null,"key_dates":[],"action_items":[],"technical_requirements":[],"objections":[],"deal_fields_to_update":[]}',
        ],
        "brief": [
            'Supply Status: H100 GPUs available.\nDemand Pipeline: Active deals.\nPriority Match: Close deal.\nCoverage Gaps: None.\nLeadership Brief: Proceed.',
        ],
    }

    results = {}
    scorers = {
        "match": _score_match,
        "validation": _score_validation,
        "diff": _score_diff,
        "extraction": _score_extraction,
        "brief": _score_brief,
    }

    for task_type, gts in _probe_gts.items():
        scorer = scorers[task_type]
        policies = _degenerate.get(task_type, [])
        for i, policy_response in enumerate(policies):
            policy_name = f"policy_{i}"
            max_reward = 0.0
            for gt in gts:
                try:
                    r = scorer(policy_response, gt)
                    max_reward = max(max_reward, r)
                except Exception:
                    pass
            results[f"{task_type}:{policy_name}"] = round(max_reward, 4)

    return results


class PrimeGrwData(vf.TaskData):
    """One row's immutable data: the prompt plus the scoring ground truth."""

    task_type: str
    gt: dict[str, Any]
    fact_sheet: str = ""
    """Rendered verified-fact reference for the brief rubric judge (empty for
    non-brief rows); wired to the judge via RubricJudgeConfig.answer_field."""


def _default_brief_judge_config() -> "vf.RubricJudgeConfig":
    """Brief rubric-judge defaults (SCORER_V03_DESIGN §4.5c).

    Kwargs are filtered by the resolved RubricJudgeConfig's model_fields so the
    hosted-training image's bundled verifiers fork (0.2.2.dev6, field set may
    differ) can still construct the config — same defensive pattern as load().
    """
    kwargs: dict[str, Any] = {
        "name": "brief_judge",
        # Different family from the glm-5.3 policy (no self-judging); hosted by
        # Prime, per-token via Prime Inference. Override:
        # --env.taskset.task.judge.model
        "model": "Qwen/Qwen3.5-122B-A10B",
        "path": Path(__file__).resolve().parent / "brief_rubric.toml",
        "answer_field": "fact_sheet",
        # The brief is single-turn: judge the final reply only, not the trace.
        "view": "last_reply",
        # Plain-text JSON parsing: hosted/non-OpenAI endpoints are flaky on
        # structured decoding (documented in verifiers' rubric judge).
        "structured_output": False,
    }
    allowed = set(vf.RubricJudgeConfig.model_fields)
    return vf.RubricJudgeConfig(**{k: v for k, v in kwargs.items() if k in allowed})


class PrimeGrwTaskConfig(vf.TaskConfig):
    """Per-task knobs: the brief rubric judge (off by default).

    judge_enabled=False keeps the env self-contained for hosted RL training on
    the image's bundled verifiers fork (no judge credentials in the sandbox,
    fork drift risk); evals enable it with
    --env.taskset.task.judge_enabled=true. With the judge off, brief scores
    are deterministic grounding renormalized to 1.0 — judged and unjudged
    brief scores are different regimes and must not be compared.
    """

    judge_enabled: bool = False
    judge: "vf.RubricJudgeConfig" = Field(default_factory=_default_brief_judge_config)


class PrimeGrwTask(vf.Task[PrimeGrwData]):
    """Dispatches trace.last_reply to the battle-tested _score_* functions."""

    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        response = trace.last_reply
        if not response:
            return 0.0
        task_type = self.data.task_type
        gt = self.data.gt
        if task_type == "match":
            return _score_match(response, gt)
        if task_type == "validation":
            return _score_validation(response, gt)
        if task_type == "diff":
            return _score_diff(response, gt)
        if task_type == "extraction":
            return _score_extraction(response, gt)
        if task_type == "brief":
            grounding, gate = _brief_grounding(response, gt)
            cfg = self.config
            if getattr(cfg, "judge_enabled", False):
                if not gate:
                    # Degenerate/ungrounded rollouts never reach the judge:
                    # hard reward floor AND zero judge spend.
                    return grounding
                judge = vf.RubricJudge(getattr(cfg, "judge"))
                judge_score = await judge.score(self.data, trace)
                return min(1.0, grounding + 0.65 * judge_score)
            return grounding
        return 0.0


class PrimeGrwConfig(vf.TasksetConfig):
    """Load-time knobs; mirror the legacy load_environment signature."""

    data_source: str = "synthetic"
    """synthetic for seeded synthetic data, anything else for the AF-47 vault."""
    vault_root: str | None = None
    """Optional vault root override (AF_VAULT_ROOT env var also applies; no default path ships)."""
    task_types: list[TaskType] = Field(default_factory=lambda: list(_DEFAULT_TASK_TYPES))
    """Task types to include; the legacy None default maps to all five."""
    rows_per_task_type: int = 40
    """Rows requested per task type before the seeded mixture cap."""
    seed: int = 42
    """Seed for synthetic data generation and row sampling."""
    split: str = "all"
    """all, train, or eval; train/eval apply the legacy 80/20 split."""
    task: PrimeGrwTaskConfig = Field(default_factory=PrimeGrwTaskConfig)
    """Per-task config (brief rubric judge); load() already passes it through."""


class PrimeGrwTaskset(vf.Taskset[PrimeGrwTask, PrimeGrwConfig]):
    def load(self) -> list[PrimeGrwTask]:
        check_schema_drift(_resolve_vault_root(self.config.vault_root))
        dataset = build_dataset(
            data_source=self.config.data_source,
            vault_root=self.config.vault_root,
            task_types=self.config.task_types,
            rows_per_task_type=self.config.rows_per_task_type,
            seed=self.config.seed,
            split=self.config.split,
        )
        tasks: list[PrimeGrwTask] = []
        for row in dataset:
            info = row["info"]
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except (json.JSONDecodeError, TypeError):
                    info = {}
            if not isinstance(info, dict):
                info = {}
            row_id = row.get("id") or row.get("row_id") or info.get("id")
            # Per the v1 taskset model: prompts must not contain system-message
            # content. Each builder embeds its task-type system prompt at the
            # head of the question; split it into TaskData.system_prompt here so
            # harnesses (and GEPA) receive it as a proper system component.
            task_type = info.get("task_type", "")
            sys_prompt = SYSTEM_PROMPT_BY_TYPE.get(task_type, "")
            question = row["question"]
            if sys_prompt and question.startswith(sys_prompt):
                question = question[len(sys_prompt):].lstrip("\n")
            else:
                sys_prompt = ""
            # Field availability differs across verifiers v1 generations (the
            # hosted-training image bundles a fork whose TaskData has no `id`
            # and forbids extras) — pass only fields the resolved model knows.
            candidate = {
                "id": str(row_id) if row_id is not None else f"idx-{len(tasks)}",
                "idx": len(tasks),
                "system_prompt": sys_prompt or None,
                "prompt": question,
                "task_type": info.get("task_type", ""),
                "gt": info.get("ground_truth", {}),
                "fact_sheet": str(info.get("ground_truth", {}).get("fact_sheet", "") or ""),
                "name": "prime-grw/" + info.get("task_type", "unknown"),
            }
            allowed = set(PrimeGrwData.model_fields)
            data = PrimeGrwData(**{k: v for k, v in candidate.items() if k in allowed})
            task_cfg = getattr(self.config, "task", None)
            tasks.append(PrimeGrwTask(data, task_cfg) if task_cfg is not None else PrimeGrwTask(data))
        return tasks


# Backward-compat aliases (pre-debut class names; imports only — new code uses PrimeGrw*)
PrimeOSEvalData = PrimeGrwData
PrimeOSEvalTask = PrimeGrwTask
PrimeOSTaskConfig = PrimeGrwTaskConfig
PrimeOSEvalConfig = PrimeGrwConfig
PrimeOSEvalTaskset = PrimeGrwTaskset

__all__ = ["PrimeGrwTaskset", "PrimeGrwConfig", "PrimeGrwTask", "PrimeGrwTaskConfig", "PrimeGrwData"]
# Backward-compat aliases stay importable but are excluded from __all__ so taskset
# discovery (which resolves __all__ entries to classes) sees exactly one Taskset.
