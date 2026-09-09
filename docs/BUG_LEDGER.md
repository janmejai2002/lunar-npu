# Lunar Project Defect & Bug Ledger

This ledger is the authoritative, structured audit trail for autonomous defect identification, reproduction, root-cause analysis, and specialist verification.

## Schema
| ID | Timestamp | Severity | Subsystem | Defect Summary | Root Cause | Assigned Specialist | Status | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

---

## Active & Historical Entries

| ID | Timestamp | Severity | Subsystem | Defect Summary | Root Cause | Assigned Specialist | Status | Verification Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BUG-001** | 2026-09-09 23:45 | High | Antigravity Gateway | `antigravity-tools` proxy defaulted to disabled on startup (HTTP 503) | Service requires POST `/api/proxy/start` after daemon launch | Multi-Account Quota Orchestrator | **VERIFIED_FIX** | `cli-anything-antigravity-tools proxy status` -> `running: true` |
| **BUG-002** | 2026-09-10 00:45 | Medium | CLI Harness | `UnicodeEncodeError: 'charmap'` on Windows PowerShell console output | Windows stdout default encoding cp1252 unable to serialize box drawing & UTF-8 dots | Autonomous DevOps Reliability Lead | **VERIFIED_FIX** | `antigravity_tools_cli.py` & `repl_skin.py` reconfigured with `sys.stdout.reconfigure(encoding="utf-8")` |
| **BUG-003** | 2026-09-10 00:46 | Medium | Perpetual Bridge | `bridge_8045.py` forwarded requests without `Authorization: Bearer <api_key>`, resulting in HTTP 401 | Missing automatic discovery of `sk-3e5a3fa2...` from `~/.antigravity_tools/gui_config.json` | Hardware Security & Cryptography Specialist | **VERIFIED_FIX** | `test_bridge_forward_chat_completion` PASSED in `antigravity-perpetual/tests/` |

---

## Status Legend
- **`OPEN`**: Defect identified and reproduction confirmed.
- **`INVESTIGATING`**: Root cause analysis underway by designated council specialist.
- **`FIX_STAGED`**: Patch applied, awaiting automated test suite pass.
- **`VERIFIED_FIX`**: Automated regression test passing cleanly; committed to Git.
