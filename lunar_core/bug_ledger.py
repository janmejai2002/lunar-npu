"""
Autonomous Bug Ledger Engine
Maintains docs/BUG_LEDGER.md as an external, structured defect audit trail.
Supports programmatic defect logging, status progression, and automated test linking.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


class BugLedgerEngine:
    """Engine for reading, writing, and advancing bug lifecycle in docs/BUG_LEDGER.md."""

    DEFAULT_LEDGER_PATH = Path("docs") / "BUG_LEDGER.md"

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        self.ledger_path = (ledger_path or self.DEFAULT_LEDGER_PATH).resolve()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            self._init_empty_ledger()

    def _init_empty_ledger(self) -> None:
        content = """# Lunar Project Defect & Bug Ledger

This ledger is the authoritative, structured audit trail for autonomous defect identification, reproduction, root-cause analysis, and specialist verification.

## Schema
| ID | Timestamp | Severity | Subsystem | Defect Summary | Root Cause | Assigned Specialist | Status | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

---

## Active & Historical Entries

| ID | Timestamp | Severity | Subsystem | Defect Summary | Root Cause | Assigned Specialist | Status | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

---

## Status Legend
- **`OPEN`**: Defect identified and reproduction confirmed.
- **`INVESTIGATING`**: Root cause analysis underway by designated council specialist.
- **`FIX_STAGED`**: Patch applied, awaiting automated test suite pass.
- **`VERIFIED_FIX`**: Automated regression test passing cleanly; committed to Git.
"""
        self.ledger_path.write_text(content, encoding="utf-8")

    def read_entries(self) -> List[Dict[str, str]]:
        """Parse all markdown table rows into structured dicts."""
        if not self.ledger_path.exists():
            return []

        text = self.ledger_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        entries = []

        in_entries = False
        for line in lines:
            line_str = line.strip()
            if "## Active & Historical Entries" in line_str:
                in_entries = True
                continue
            if in_entries and line_str.startswith("## "):
                break
            if in_entries and line_str.startswith("|") and not line_str.startswith("| :---") and not line_str.startswith("| ID"):
                cols = [c.strip() for c in line_str.split("|")[1:-1]]
                if len(cols) >= 9:
                    clean_id = re.sub(r"\*\*|\*", "", cols[0])
                    entries.append({
                        "id": clean_id,
                        "timestamp": cols[1],
                        "severity": cols[2],
                        "subsystem": cols[3],
                        "summary": cols[4],
                        "root_cause": cols[5],
                        "specialist": cols[6],
                        "status": re.sub(r"\*\*|\*|`", "", cols[7]),
                        "verification_test": cols[8],
                    })
        return entries

    def log_bug(
        self,
        bug_id: str,
        severity: str,
        subsystem: str,
        summary: str,
        root_cause: str,
        specialist: str,
        test: str,
        status: str = "OPEN",
    ) -> Dict[str, str]:
        """Add a new defect entry to the ledger."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = {
            "id": bug_id,
            "timestamp": now_str,
            "severity": severity,
            "subsystem": subsystem,
            "summary": summary,
            "root_cause": root_cause,
            "specialist": specialist,
            "status": status,
            "verification_test": test,
        }

        entries = self.read_entries()
        # Update existing or append
        updated = False
        for i, existing in enumerate(entries):
            if existing["id"] == bug_id:
                entries[i] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)

        self._save_entries(entries)
        return entry

    def resolve_bug(self, bug_id: str, verification_result: Optional[str] = None) -> bool:
        """Advance bug status to VERIFIED_FIX."""
        entries = self.read_entries()
        for e in entries:
            if e["id"] == bug_id:
                e["status"] = "VERIFIED_FIX"
                if verification_result:
                    e["verification_test"] = f"{e['verification_test']} (PASSED: {verification_result})"
                self._save_entries(entries)
                return True
        return False

    def _save_entries(self, entries: List[Dict[str, str]]) -> None:
        """Regenerate markdown document with updated table."""
        header = """# Lunar Project Defect & Bug Ledger

This ledger is the authoritative, structured audit trail for autonomous defect identification, reproduction, root-cause analysis, and specialist verification.

## Schema
| ID | Timestamp | Severity | Subsystem | Defect Summary | Root Cause | Assigned Specialist | Status | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

---

## Active & Historical Entries

| ID | Timestamp | Severity | Subsystem | Defect Summary | Root Cause | Assigned Specialist | Status | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        rows = []
        for e in entries:
            status_disp = f"**{e['status']}**" if "FIX" in e["status"] else f"`{e['status']}`"
            row = (
                f"| **{e['id']}** | {e['timestamp']} | {e['severity']} | {e['subsystem']} | "
                f"{e['summary']} | {e['root_cause']} | {e['specialist']} | {status_disp} | `{e['verification_test']}` |"
            )
            rows.append(row)

        footer = """
---

## Status Legend
- **`OPEN`**: Defect identified and reproduction confirmed.
- **`INVESTIGATING`**: Root cause analysis underway by designated council specialist.
- **`FIX_STAGED`**: Patch applied, awaiting automated test suite pass.
- **`VERIFIED_FIX`**: Automated regression test passing cleanly; committed to Git.
"""
        content = header + "\n".join(rows) + "\n" + footer
        self.ledger_path.write_text(content, encoding="utf-8")
