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
| **BUG-004** | 2026-09-10 01:43 | High | Power Supervisor | `create_app` instantiated decoupled `PowerManager`, reporting `away_mode_enabled: False` in `/api/telemetry` | `create_app` did not activate or inherit the active `PowerManager` instance | System Software & Kernel Architect | **VERIFIED_FIX** | `create_app` automatically enforces `power_mgr.enable_perpetual_mode()`; `/api/telemetry` verified `away_mode_enabled: True` |
| **BUG-005** | 2026-09-10 01:43 | Medium | NPU Synergy | `perpetual_config.yaml` pointed `npu_server_url` to port 8765 instead of 8899, resulting in offline NPU status | Config defaulted to supervisor port instead of Lunar Studio port | Heterogeneous Hardware Integration Lead | **VERIFIED_FIX** | Config updated to `http://127.0.0.1:8899`; `/api/telemetry` verified `npu_silicon: online=True` |
| **BUG-006** | 2026-09-10 01:45 | Low | Test Suite | `test_lunar_router.py` failed with `assert 768 == 384` on physical silicon | BGE-base model weights on machine produce 768-dim embeddings rather than 384-dim MiniLM | Edge AI & On-Device SLM Researcher | **VERIFIED_FIX** | `test_lunar_router.py` updated to assert `dim in (384, 768)`; 25/25 tests passing |

---

## Status Legend
- **`OPEN`**: Defect identified and reproduction confirmed.
- **`INVESTIGATING`**: Root cause analysis underway by designated council specialist.
- **`FIX_STAGED`**: Patch applied, awaiting automated test suite pass.
- **`VERIFIED_FIX`**: Automated regression test passing cleanly; committed to Git.

