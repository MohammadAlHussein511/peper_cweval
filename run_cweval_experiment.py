#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cweval_experiment.py — النسخة التنفيذية المباشرة (تعمل في WSL2 Ubuntu أو أي سحابة بأمر واحد)
تدعم كلاً من:
1. وضع السحاب الفوري بدون GPU وبدون تنزيل 57GB (عبر مفتاح NVIDIA NIM المجاني من build.nvidia.com أو OpenRouter).
2. وضع الـ GPU المحلي/السحابي (عبر llama-server وملفات GGUF).
مع حفظ واستئناف تلقائي فوري (SQLite WAL + JSON Checkpoints + Portable Snapshot) لا يبدأ من الصفر أبداً!
"""

# ==============================================================================
# الخلية 1: تهيئة البيئة، الربط الإلزامي بـ Google Drive، ومحرك الحالة الدائم (SQLite WAL)
# ==============================================================================
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

IS_KAGGLE = Path("/kaggle/working").exists()
IS_COLAB = ("google.colab" in sys.modules) or Path("/content").exists()

def resolve_persistent_workspace() -> Path:
    """
    يضمن استخدام مساحة تخزين دائمة لا تُمسح عند انقطاع الإنترنت أو إعادة تشغيل الجلسة:
    - في Google Colab: يربط Google Drive تلقائياً ويحفظ كل شيء في `/content/drive/MyDrive/cweval_study`.
    - في Kaggle: يستخدم `/kaggle/working/cweval_study`.
    - في Lightning AI / محلياً: يستخدم المجلد الدائم للمشروع.
    """
    override = os.environ.get("CWEVAL_WORK_DIR")
    if override:
        p = Path(override).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    if IS_COLAB:
        drive_root = Path("/content/drive/MyDrive")
        if not drive_root.exists():
            try:
                from google.colab import drive
                print("🔗 جاري ربط Google Drive لضمان حفظ التقدم بشكل دائم وعدم الإعادة من الصفر عند انقطاع النت...")
                drive.mount("/content/drive", force_remount=False)
            except Exception as exc:
                print(f"⚠️ تعذر ربط Google Drive تلقائياً ({exc})، سيتم استخدام /content/cweval_study المؤقت.")
                fallback = Path("/content/cweval_study")
                fallback.mkdir(parents=True, exist_ok=True)
                return fallback
        if drive_root.exists():
            persistent_dir = drive_root / "cweval_study"
            persistent_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ تم تفعيل الحفظ الدائم في Google Drive: {persistent_dir}")
            return persistent_dir

    if IS_KAGGLE:
        p = Path("/kaggle/working/cweval_study")
        p.mkdir(parents=True, exist_ok=True)
        return p

    p = Path("./cweval_study").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


BASE_WORK_DIR: Path = resolve_persistent_workspace()
TMP_MODELS_DIR: Path = Path(os.environ.get("CWEVAL_TMP_MODELS", "/tmp/cweval_gguf_models"))
LOCAL_EXEC_BIN_DIR: Path = Path("/tmp/cweval_exec_bin")  # لتشغيل الملف التنفيذي بصلاحيات Linux كاملة
PERSISTENT_BIN_DIR: Path = BASE_WORK_DIR / "bin"
CHECKPOINTS_DIR: Path = BASE_WORK_DIR / "checkpoints"
PROMPTS_DIR: Path = BASE_WORK_DIR / "sanitized_prompts"
RAW_LOGS_DIR: Path = BASE_WORK_DIR / "raw_responses"
EXPORT_DIR: Path = BASE_WORK_DIR / "cweval_export"
CWEVAL_REPO_DIR: Path = Path(os.environ.get("CWEVAL_REPO_DIR", str(BASE_WORK_DIR / "CWEval_repo_cache")))
SQLITE_DB_PATH: Path = BASE_WORK_DIR / "state_ledger.sqlite3"

for directory in [
    BASE_WORK_DIR,
    TMP_MODELS_DIR,
    LOCAL_EXEC_BIN_DIR,
    PERSISTENT_BIN_DIR,
    CHECKPOINTS_DIR,
    PROMPTS_DIR,
    RAW_LOGS_DIR,
    EXPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# الثوابت العلمية المرجعية
CWEVAL_REPO_URL = "https://github.com/Co1lin/CWEval.git"
CWEVAL_PINNED_COMMIT = "e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7"
CWEVAL_DOCKER_DIGEST = "co1lin/cweval@sha256:2d28261cc26838e20b69ce28a742b376a7c420b2f9591b32b95922e2794413d0"
GLOBAL_SEED = 42

# ==============================================================================
# ⚡ خيار التسريع الفوري عبر السحاب (Cloud API Mode - بدون الحاجة لتحميل 57GB أو بطاقة GPU!)
# ==============================================================================
# إذا وضعت مفتاح NVIDIA NIM المجاني (من https://build.nvidia.com - فوري بدون مراجعة وبدون بطاقة)
# أو مفتاح OpenRouter، سيعمل الكود عبر السحاب فوراً وينهي الـ 275 استدعاء في ~12 دقيقة حتى على جهازك العادي بدون GPU!
# أما إذا تركتهما فارغين ("")، فسيعمل تلقائياً بنظام تحميل ملفات GGUF على الـ GPU.
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
NVIDIA_NIM_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

CLOUD_API_PROVIDERS = {
    "cloudflare_workers_ai": {
        "base_url": f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1",
        "api_key": CF_API_TOKEN,
        "models": {
            "writer": "@cf/qwen/qwen3-30b-a3b-fp8",
            "func_reviewer": "@cf/qwen/qwen3.8-27b",
            "sec_reviewer": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        },
        "fallbacks": {
            "writer": ["@cf/qwen/qwen3.8-27b"],
            "func_reviewer": ["@cf/google/gemma-4-26b-a4b-it", "@cf/qwen/qwen3-30b-a3b-fp8"],
            "sec_reviewer": ["@cf/deepseek-ai/deepseek-v4.1-flash-1017"],
        },
    },
    "nvidia_nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": NVIDIA_NIM_API_KEY,
        "models": {
            "writer": "qwen/qwen3-coder-480b-a35b-instruct",
            "func_reviewer": "qwen/qwen3-235b-a22b",
            "sec_reviewer": "deepseek-ai/deepseek-r1-distill-qwen-32b",
        },
        "fallbacks": {
            "writer": ["qwen/qwen3-next-80b-a3b-instruct"],
            "func_reviewer": ["qwen/qwen3-next-80b-a3b-instruct", "qwen/qwen3.5-122b-a10b"],
            "sec_reviewer": ["deepseek-ai/deepseek-r1"],
        },
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "models": {
            "writer": "qwen/qwen3-coder-30b-a3b-instruct",
            "func_reviewer": "qwen/qwen3-32b",
            "sec_reviewer": "deepseek/deepseek-r1-distill-qwen-32b",
        },
    },
}

RUN_ID = os.environ.get("CWEVAL_RUN_ID", "run_2026_persistent_v3")

MODELS_CONFIG: dict[str, dict[str, Any]] = {
    "writer": {
        "role": "Writer & Homogeneous Self-Reviewer",
        "canonical_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "hf_repo": "ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "context_size": 4096,
    },
    "func_reviewer": {
        "role": "Heterogeneous Functional Reviewer",
        "canonical_name": "Qwen/Qwen3.8-27B",
        "hf_repo": "ggml-org/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "context_size": 4096,
    },
    "sec_reviewer": {
        "role": "Heterogeneous Security Reviewer",
        "canonical_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "context_size": 4096,
    },
}

MANUAL_AUDITED_OVERRIDES = {
    "cwe_020_0": {
        "test_nodeid": "core/py/cwe_020_0_test.py::test_get_target_for_redirect_unsafe_security",
        "source_lines": "benchmark/core/py/cwe_020_0_test.py:84-96",
        "override_category": "Security",
        "reason": "Manual Audited Override: security test function lacks standard pytest security marker in upstream commit",
    },
    "cwe_643_0": {
        "test_nodeid": "core/py/cwe_643_0_test.py::test_get_student_info_unsafe_security",
        "source_lines": "benchmark/core/py/cwe_643_0_test.py:118-131",
        "override_category": "Security",
        "reason": "Manual Audited Override: security test function lacks standard pytest security marker in upstream commit",
    },
}


class PersistentExperimentLedger:
    """
    محرك حالة حديث مزدوج الطبقة (SQLite WAL + Atomic Fsync JSON Checkpoints):
    - يحفظ كل استدعاء فور انتهائه مباشرة على القرص الدائم (مع os.fsync لضمان الكتابة الفيزيائية في Google Drive).
    - يسترجع الحالة الكاملة عند إعادة التشغيل في أجزاء من الثانية.
    - يحدد بدقة أي المراحل (Stages 1..4) اكتملت بالكامل لتخطي تحميل نماذجها (19GB) نهائياً!
    """

    def __init__(self, db_path: Path, checkpoints_dir: Path):
        self.db_path = db_path
        self.checkpoints_dir = checkpoints_dir
        self._init_db()
        self._sync_existing_json_checkpoints()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_calls (
                    call_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    role TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inference_seconds REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    syntax_valid INTEGER DEFAULT 1,
                    report_json_valid INTEGER DEFAULT 1,
                    payload_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _sync_existing_json_checkpoints(self) -> None:
        """يدمج أي ملفات JSON موجودة مسبقاً في مجلد checkpoints داخل قاعدة بيانات SQLite لضمان التوافق التام."""
        json_files = sorted(self.checkpoints_dir.glob("*.json"))
        if not json_files:
            return
        with self._connect() as conn:
            existing_ids = {
                row[0]
                for row in conn.execute("SELECT call_id FROM llm_calls WHERE status='SUCCESS'").fetchall()
            }
            imported = 0
            for jf in json_files:
                cid = jf.stem
                if cid in existing_ids:
                    continue
                try:
                    rec = json.loads(jf.read_text(encoding="utf-8"))
                    if rec.get("status") == "SUCCESS" and (
                        "extracted_code" in rec or "parsed_report" in rec
                    ):
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO llm_calls (
                                call_id, task_id, stage_name, arm, role, model_key, status,
                                inference_seconds, input_tokens, output_tokens,
                                syntax_valid, report_json_valid, payload_json, updated_at_utc
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec["call_id"],
                                rec["task_id"],
                                rec["stage_name"],
                                rec["arm"],
                                rec["role"],
                                rec["model_key"],
                                rec["status"],
                                float(rec.get("inference_seconds", 0.0)),
                                int(rec.get("input_tokens", 0)),
                                int(rec.get("output_tokens", 0)),
                                int(bool(rec.get("syntax_valid", True))),
                                int(bool(rec.get("report_json_valid", True))),
                                json.dumps(rec, ensure_ascii=False),
                                rec.get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
                            ),
                        )
                        imported += 1
                except Exception:
                    continue
            if imported > 0:
                conn.commit()
                print(f"🔄 تم استيراد وتثبيت {imported} نقطة حفظ سابقة في سجل SQLite الدائم.")

    def get_completed_call(self, call_id: str) -> Optional[dict[str, Any]]:
        """يعيد بيانات الاستدعاء المكتمل فوراً إذا كان محفوظاً في SQLite أو ملف JSON صالح."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM llm_calls WHERE call_id=? AND status='SUCCESS'",
                (call_id,),
            ).fetchone()
            if row:
                try:
                    rec = json.loads(row[0])
                    if "extracted_code" in rec or "parsed_report" in rec:
                        return rec
                except Exception:
                    pass

        ckpt_file = self.checkpoints_dir / f"{call_id}.json"
        if ckpt_file.exists():
            try:
                rec = json.loads(ckpt_file.read_text(encoding="utf-8"))
                if rec.get("status") == "SUCCESS" and (
                    "extracted_code" in rec or "parsed_report" in rec
                ):
                    self.save_completed_call(rec)
                    return rec
            except Exception:
                pass
        return None

    def save_completed_call(self, record: dict[str, Any]) -> None:
        """يكتب الاستدعاء المكتمل بشكل ذري (Atomic Write + Fsync) في كل من JSON وSQLite."""
        call_id = record["call_id"]
        payload_str = json.dumps(record, indent=2, ensure_ascii=False)

        # 1. كتابة ذرية لملف JSON مع fsync لإجبار Google Drive على المزامنة الفورية
        ckpt_file = self.checkpoints_dir / f"{call_id}.json"
        tmp_file = self.checkpoints_dir / f"{call_id}.json.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(payload_str)
            f.flush()
            os.fsync(f.fileno())
        tmp_file.replace(ckpt_file)

        # 2. كتابة فورية في قاعدة بيانات SQLite WAL
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_calls (
                    call_id, task_id, stage_name, arm, role, model_key, status,
                    inference_seconds, input_tokens, output_tokens,
                    syntax_valid, report_json_valid, payload_json, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    record["task_id"],
                    record["stage_name"],
                    record["arm"],
                    record["role"],
                    record["model_key"],
                    record["status"],
                    float(record.get("inference_seconds", 0.0)),
                    int(record.get("input_tokens", 0)),
                    int(record.get("output_tokens", 0)),
                    int(bool(record.get("syntax_valid", True))),
                    int(bool(record.get("report_json_valid", True))),
                    payload_str,
                    record.get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
                ),
            )
            conn.commit()

    def get_expected_call_ids_for_stage(self, stage_num: int, task_ids: list[str]) -> list[str]:
        expected: list[str] = []
        for tid in task_ids:
            if stage_num == 1:
                expected.append(f"{tid}__A1_draft")
                for s_idx in range(4):
                    expected.append(f"{tid}__A2_sample_{s_idx}")
                expected.append(f"{tid}__B_func_review")
                expected.append(f"{tid}__B_sec_review")
            elif stage_num == 2:
                expected.append(f"{tid}__C_func_review")
            elif stage_num == 3:
                expected.append(f"{tid}__C_sec_review")
            elif stage_num == 4:
                expected.append(f"{tid}__B_merged")
                expected.append(f"{tid}__C_merged")
        return expected

    def get_stage_progress(self, stage_num: int, task_ids: list[str]) -> tuple[int, int]:
        expected = self.get_expected_call_ids_for_stage(stage_num, task_ids)
        done = sum(1 for cid in expected if self.get_completed_call(cid) is not None)
        return done, len(expected)

    def print_resume_dashboard(self, task_ids: list[str]) -> dict[int, tuple[int, int]]:
        print("\n" + "═" * 78)
        print("📊 لوحة تتبع حالة الإنجاز والاستئناف التلقائي (Persistent Resume Dashboard)")
        print("═" * 78)
        stage_titles = {
            1: "Stage 1 (Qwen3-Coder-30B : A1 + A2x4 + B Self-Reviews)",
            2: "Stage 2 (Qwen3.8-27B     : Arm C Functional Reviews)",
            3: "Stage 3 (DeepSeek-R1-32B : Arm C Security Audits)",
            4: "Stage 4 (Qwen3-Coder-30B : Final Merge for Arms B & C)",
        }
        summary: dict[int, tuple[int, int]] = {}
        total_done = 0
        total_expected = 0
        for s_num in (1, 2, 3, 4):
            done, exp = self.get_stage_progress(s_num, task_ids)
            summary[s_num] = (done, exp)
            total_done += done
            total_expected += exp
            pct = (done / exp * 100.0) if exp else 100.0
            badge = "✅ مكتملة (سيتم تخطي تنزيل النموذج)" if done == exp else ("⏳ قيد الاستئناف" if done > 0 else "⬜ لم تبدأ بعد")
            print(f"  • {stage_titles[s_num]:<54}: {done:3d}/{exp:3d} ({pct:5.1f}%) | {badge}")
        overall_pct = (total_done / total_expected * 100.0) if total_expected else 0.0
        print("─" * 78)
        print(f"  📌 الإجمالي المنجز والمدقق على القرص الدائم: {total_done}/{total_expected} استدعاء ({overall_pct:.1f}%)")
        print("═" * 78 + "\n")
        return summary


def get_system_ram_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 * 1024), 2)
    except Exception:
        pass
    return 12.7


def collect_environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "run_id": RUN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "is_kaggle": IS_KAGGLE,
        "is_colab": IS_COLAB,
        "persistent_work_dir": str(BASE_WORK_DIR),
        "system_ram_gb": get_system_ram_gb(),
        "global_seed": GLOBAL_SEED,
        "cweval_pinned_commit": CWEVAL_PINNED_COMMIT,
        "cweval_docker_digest": CWEVAL_DOCKER_DIGEST,
        "gpus": [],
        "disk_working_free_gb": round(shutil.disk_usage(BASE_WORK_DIR).free / (1024**3), 2),
        "disk_tmp_free_gb": round(shutil.disk_usage("/tmp").free / (1024**3), 2),
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                meta["gpus"].append({
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total": parts[2],
                    "compute_cap": parts[3],
                    "driver_version": parts[4],
                })
    except Exception as e:
        meta["gpu_query_error"] = str(e)
    return meta


LEDGER = PersistentExperimentLedger(SQLITE_DB_PATH, CHECKPOINTS_DIR)
env_meta = collect_environment_metadata()
(BASE_WORK_DIR / "environment_manifest.json").write_text(
    json.dumps(env_meta, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"📂 مسار التخزين الدائم الفعال: {BASE_WORK_DIR}")
print(f"🖥️ البطاقات الرسومية المكتشفة: {env_meta['gpus'] if env_meta['gpus'] else 'CPU Mode / No GPU'}")



# ==============================================================================
# الخلية 2: تجهيز مستودع CWEval المرجعي (مع الاسترداد الفوري من الذاكرة الدائمة دون إعادة التحميل)
# ==============================================================================

def prepare_cweval_repository(repo_dir: Path, repo_url: str, pinned_commit: str) -> str:
    """
    يستنسخ مستودع CWEval مرة واحدة فقط ويحفظه، وعند إعادة التشغيل يتحقق من البصمة في ثانية واحدة.
    """
    if (repo_dir / ".git").exists():
        try:
            current_head = subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            ).strip()
            if current_head == pinned_commit and (repo_dir / "benchmark" / "core" / "py").exists():
                print(f"✅ مستودع CWEval جاهز ومحفوظ مسبقاً على الإصدار المثبت: {current_head}")
                return current_head
        except Exception:
            pass

    if not (repo_dir / ".git").exists():
        print(f"⬇️ جاري استنساخ {repo_url} إلى {repo_dir}...")
        subprocess.run(["git", "clone", "--quiet", repo_url, str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--quiet"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--force", pinned_commit], check=True)
    actual_head = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual_head != pinned_commit:
        raise RuntimeError(f"Commit mismatch! Expected {pinned_commit}, got {actual_head}")
    print(f"✅ تم تثبيت مستودع CWEval على الإصدار: {actual_head}")
    return actual_head


def extract_and_verify_prompt(task_file_path: Path) -> dict[str, Any]:
    """
    تقرأ ملف المهمة الأصلي من benchmark/core/py/cwe_*_task.py
    وتقتطع كل ما يأتي قبل BEGIN SOLUTION، ثم تتحقق عبر AST من عدم تسرب أي حل مرجعي.
    """
    raw_source = task_file_path.read_text(encoding="utf-8")
    raw_sha256 = hashlib.sha256(raw_source.encode("utf-8")).hexdigest()

    marker_used = None
    for marker in ["# BEGIN SOLUTION", "# ======= BEGIN SOLUTION =======", "BEGIN SOLUTION"]:
        if marker in raw_source:
            idx = raw_source.index(marker)
            prefix = raw_source[:idx]
            last_newline = prefix.rfind("\n")
            prompt_code = prefix[: last_newline + 1] if last_newline != -1 else prefix
            marker_used = marker
            break

    if marker_used is None:
        tree = ast.parse(raw_source)
        lines = raw_source.splitlines(keepends=True)
        safe_blocks = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
                seg = ast.get_source_segment(raw_source, node)
                if seg:
                    safe_blocks.append(seg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                header_end = node.body[0].lineno - 1
                if (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    header_end = node.body[0].end_lineno
                header_src = "".join(lines[node.lineno - 1 : header_end])
                safe_blocks.append(header_src.rstrip() + "\n    pass\n")
        prompt_code = "\n\n".join(safe_blocks) + "\n"
        marker_used = "AST_BODY_STRIP"

    testable_src = prompt_code
    try:
        tree_check = ast.parse(testable_src)
    except SyntaxError:
        testable_src = prompt_code.rstrip() + "\n    pass\n"
        tree_check = ast.parse(testable_src)

    leaked_functions = []
    func_names = []
    for node in ast.walk(tree_check):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.append(node.name)
            non_trivial_stmts = []
            for idx_stmt, stmt in enumerate(node.body):
                if (
                    idx_stmt == 0
                    and isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    continue
                if isinstance(stmt, ast.Pass):
                    continue
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                ):
                    continue
                non_trivial_stmts.append(type(stmt).__name__)
            if non_trivial_stmts:
                leaked_functions.append((node.name, non_trivial_stmts))

    if leaked_functions:
        raise ValueError(f"SECURITY LEAK DETECTED in {task_file_path.name}: {leaked_functions}")

    prompt_sha256 = hashlib.sha256(prompt_code.encode("utf-8")).hexdigest()
    return {
        "task_id": task_file_path.name.replace("_task.py", ""),
        "task_filename": task_file_path.name,
        "relative_path": f"benchmark/core/py/{task_file_path.name}",
        "function_names": func_names,
        "marker_used": marker_used,
        "raw_task_sha256": raw_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_code": prompt_code,
    }


def load_or_restore_25_python_tasks(repo_dir: Path) -> list[dict[str, Any]]:
    """
    إذا كان ملف البيان `tasks_25_manifest.json` محفوظاً مسبقاً في Google Drive، يستعيده فوراً في 0.01 ثانية
    دون الحاجة لإعادة استنساخ المستودع أو إعادة الفحص!
    """
    manifest_path = BASE_WORK_DIR / "tasks_25_manifest.json"
    if manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(cached_manifest) == 25 and all("prompt_code" in t for t in cached_manifest):
                print(f"⚡ تم استرداد قائمة المهام الـ 25 المنقاة فوراً من الذاكرة الدائمة: {manifest_path}")
                return cached_manifest
        except Exception:
            pass

    if not (repo_dir / "benchmark" / "core" / "py").exists():
        prepare_cweval_repository(repo_dir, CWEVAL_REPO_URL, CWEVAL_PINNED_COMMIT)

    py_dir = repo_dir / "benchmark" / "core" / "py"
    task_files = sorted(py_dir.glob("cwe_*_task.py"))
    print(f"Found {len(task_files)} Python tasks in {py_dir}")

    tasks_manifest = []
    for tf in task_files:
        info = extract_and_verify_prompt(tf)
        (PROMPTS_DIR / f"{info['task_id']}_prompt.py").write_text(info["prompt_code"], encoding="utf-8")
        tasks_manifest.append(info)

    manifest_path.write_text(json.dumps(tasks_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ تم حفظ قائمة المهام الـ 25 المدققة في {manifest_path} (0 تسريبات).")
    return tasks_manifest


TASKS_MANIFEST = []
if IS_KAGGLE or IS_COLAB or ("ipykernel" in sys.modules) or os.environ.get("RUN_LIVE_CLONE", "0") == "1" or os.environ.get("RUN_FULL_EXPERIMENT", "0") == "1" or (CWEVAL_REPO_DIR / "benchmark" / "core" / "py").exists():
    TASKS_MANIFEST = load_or_restore_25_python_tasks(CWEVAL_REPO_DIR)
    LEDGER.print_resume_dashboard([t["task_id"] for t in TASKS_MANIFEST])



# ==============================================================================
# الخلية 3: إدارة محرك الاستدلال llama.cpp مع التخزين الدائم في Google Drive والمنفذ الحر
# ==============================================================================
import requests

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def find_free_port(preferred_port: int = 18080) -> int:
    """يجد منفذاً محلياً حراً (يتفادى المنفذ 8080 المحجوز داخلياً في Google Colab)."""
    for p in [preferred_port, 18081, 18082, 18083, 19090]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


LLAMA_SERVER_PORT = find_free_port(int(os.environ.get("LLAMA_SERVER_PORT", "18080")))
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"


def detect_gpu_topology() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_count": 0,
        "total_vram_mb": 0.0,
        "per_gpu_vram_mb": [],
        "cuda_arch": "75",
        "gpu_names": [],
        "system_ram_gb": get_system_ram_gb(),
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        archs = set()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["gpu_names"].append(parts[0])
                mem_mb = float(parts[1])
                info["per_gpu_vram_mb"].append(mem_mb)
                cc = parts[2].replace(".", "").strip()
                if cc.isdigit():
                    archs.add(cc)
        info["gpu_count"] = len(info["per_gpu_vram_mb"])
        info["total_vram_mb"] = sum(info["per_gpu_vram_mb"])
        if archs:
            info["cuda_arch"] = ";".join(sorted(archs))
    except Exception:
        pass
    return info


def _stage_to_local_exec(src_bin: Path) -> Path:
    """
    ينسخ الملف التنفيذي والمكتبات المشتركة من Google Drive إلى `/tmp/cweval_exec_bin`
    لأن نظام ملفات Google Drive FUSE قد يمنع صلاحية التنفيذ المباشر (`chmod +x`) لبعض الملفات الثنائية.
    """
    LOCAL_EXEC_BIN_DIR.mkdir(parents=True, exist_ok=True)
    for item in PERSISTENT_BIN_DIR.glob("*"):
        if item.is_file():
            dst = LOCAL_EXEC_BIN_DIR / item.name
            if not dst.exists() or dst.stat().st_size != item.stat().st_size:
                shutil.copy2(item, dst)
            os.chmod(dst, 0o755)
    target = LOCAL_EXEC_BIN_DIR / src_bin.name
    os.chmod(target, 0o755)
    return target


def ensure_llama_server_binary() -> Path:
    """
    يوفر محرك llama.cpp الداعم لـ CUDA دون إعادة البناء عند انقطاع الجلسة:
    1. إذا كان محفوظاً مسبقاً في مجلد Google Drive الدائم (`PERSISTENT_BIN_DIR`)، ينسخه إلى `/tmp/cweval_exec_bin` في ثانية واحدة!
    2. إذا لم يكن موجوداً، ينزل النسخة الرسمية المترجمة مسبقاً لـ CUDA من `https://llama.app/install.sh` في 10 ثوانٍ ويحفظها في Google Drive.
    3. إذا لزم البناء من المصدر كخيار احتياطي، يبنيه مرة واحدة ويحفظه في Google Drive للأبد.
    """
    for candidate_name in ("llama-app", "llama-server"):
        persisted = PERSISTENT_BIN_DIR / candidate_name
        if persisted.exists() and persisted.stat().st_size > 1_000_000:
            exec_bin = _stage_to_local_exec(persisted)
            print(f"⚡ تم استرداد محرك الاستدلال الجاهز فوراً من الذاكرة الدائمة: {exec_bin}")
            return exec_bin

    if hf_hub_download is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "requests"], check=True)

    topo = detect_gpu_topology()
    has_gpu = topo["gpu_count"] > 0

    if not has_gpu and os.environ.get("ALLOW_CPU_FALLBACK", "0") != "1":
        raise RuntimeError(
            "\n❌ توقف وقائي: الجلسة الحالية لا تحتوي على بطاقة GPU مفعلة!\n"
            "في Google Colab: اضغط Runtime -> Change runtime type -> اختر T4 GPU ثم اضغط Save وأعد التشغيل."
        )

    print("⬇️ جاري تثبيت النسخة الرسمية المترجمة مسبقاً لـ CUDA من ggml-org (تستغرق ~10 ثوانٍ)...")
    try:
        subprocess.run(["bash", "-c", "curl -LsSf https://llama.app/install.sh | sh"], check=True)
        prebuilt_path = Path.home() / ".llama-app" / "llama"
        if prebuilt_path.exists():
            persisted_llama = PERSISTENT_BIN_DIR / "llama-app"
            shutil.copy2(prebuilt_path, persisted_llama)
            exec_bin = _stage_to_local_exec(persisted_llama)
            ver_out = subprocess.check_output([str(exec_bin), "version"], text=True, stderr=subprocess.STDOUT).strip()
            (BASE_WORK_DIR / "llama_cpp_build_info.json").write_text(
                json.dumps(
                    {
                        "source": "ggml-org/llama-install.sh (official prebuilt CUDA binary)",
                        "version_output": ver_out,
                        "cuda_enabled": has_gpu,
                        "gpu_topology": topo,
                        "binary_sha256": hashlib.sha256(exec_bin.read_bytes()).hexdigest(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"✅ تم تثبيت وحفظ محرك CUDA الجاهز بنجاح في ثوانٍ!")
            return exec_bin
    except Exception as e:
        print(f"  [Info] الانتقال إلى البناء المصدري الاحتياطي ({e})...")

    target_arch = topo["cuda_arch"]
    src_dir = Path("/tmp/llama_cpp_src")
    build_dir = Path("/tmp/llama_cpp_build")
    shutil.rmtree(src_dir, ignore_errors=True)
    shutil.rmtree(build_dir, ignore_errors=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(src_dir)],
        check=True,
    )
    cmake_cmd = [
        "cmake", "-B", str(build_dir), "-S", str(src_dir),
        f"-DGGML_CUDA={'ON' if has_gpu else 'OFF'}",
        f"-DCMAKE_CUDA_ARCHITECTURES={target_arch}",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.run(cmake_cmd, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "llama-server", "-j", str(os.cpu_count() or 4)], check=True)

    for item in (build_dir / "bin").glob("*"):
        if item.is_file():
            shutil.copy2(item, PERSISTENT_BIN_DIR / item.name)
    for so_file in build_dir.rglob("*.so*"):
        if so_file.is_file():
            target_so = PERSISTENT_BIN_DIR / so_file.name
            if not target_so.exists():
                shutil.copy2(so_file, target_so)

    return _stage_to_local_exec(PERSISTENT_BIN_DIR / "llama-server")


class LlamaServerManager:
    """
    يدير دورة حياة النموذج والخادم:
    - لا يقوم بتنزيل ملف GGUF إلا عندما تكون المرحلة الحالية بحاجة فعلية لتشغيل استدعاءات متبقية.
    - إذا كان ملف الـ GGUF موجوداً بالفعل في `/tmp/cweval_gguf_models` من تشغيل سابق في نفس الجلسة، يستخدمه فوراً دون إعادة تنزيل!
    - يحتفظ ببصمة `SHA-256` في السجل الدائم على Google Drive حتى لا يُعيد حساب البصمة لملف حجمه 19GB عند الاستئناف!
    """

    def __init__(self, bin_path: Path, port: int = LLAMA_SERVER_PORT):
        self.bin_path = bin_path
        self.port = port
        self.proc: Optional[subprocess.Popen] = None
        self.current_model_key: Optional[str] = None
        self.current_gguf_path: Optional[Path] = None
        self.model_load_duration_sec: float = 0.0
        self.server_log_path: Optional[Path] = None

    def download_and_hash_gguf(self, model_key: str) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download as _hf_download

        cfg = MODELS_CONFIG[model_key]
        registry_path = BASE_WORK_DIR / "models_sha256_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}

        expected_local = TMP_MODELS_DIR / cfg["filename"]
        if expected_local.exists() and expected_local.stat().st_size > 5_000_000_000:
            print(f"⚡ ملف النموذج موجود مسبقاً في الذاكرة المحلية: {expected_local.name}")
            local_path = expected_local
            dl_sec = 0.0
        else:
            print(f"⬇️ جاري تنزيل {cfg['hf_repo']}/{cfg['filename']} إلى {TMP_MODELS_DIR}...")
            t0 = time.time()
            local_path = Path(
                _hf_download(
                    repo_id=cfg["hf_repo"],
                    filename=cfg["filename"],
                    local_dir=str(TMP_MODELS_DIR),
                )
            )
            dl_sec = round(time.time() - t0, 2)

        # إذا كانت بصمة SHA-256 محفوظة مسبقاً في Google Drive لنفس الملف والحجم، لا نضيع دقيقتين في إعادة حسابها!
        if (
            model_key in registry
            and registry[model_key].get("filename") == cfg["filename"]
            and registry[model_key].get("file_size_bytes") == local_path.stat().st_size
            and registry[model_key].get("sha256")
        ):
            print(f"⚡ بصمة SHA-256 للنموذج [{model_key}] موثقة مسبقاً في السجل الدائم: {registry[model_key]['sha256'][:16]}...")
        else:
            print(f"🔒 جاري حساب بصمة SHA-256 لملف {local_path.name} (تُحسب مرة واحدة فقط وتحفظ في Google Drive)...")
            h = hashlib.sha256()
            with open(local_path, "rb") as f:
                while chunk := f.read(32 * 1024 * 1024):
                    h.update(chunk)
            registry[model_key] = {
                **cfg,
                "file_size_bytes": local_path.stat().st_size,
                "sha256": h.hexdigest(),
                "download_seconds": dl_sec,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

        self.current_gguf_path = local_path
        return registry[model_key]

    def start(self, model_key: str) -> dict[str, Any]:
        # إذا كان الخادم يعمل بالفعل بنفس النموذج، لا نعيد تشغيله!
        if self.current_model_key == model_key and self.proc is not None and self.proc.poll() is None:
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    print(f"⚡ الخادم للنموذج [{model_key}] يعمل وجاهز بالفعل!")
                    return {"model_load_duration_sec": self.model_load_duration_sec, "gpu_offload_verified": True}
            except Exception:
                pass

        self.stop(delete_gguf=False)
        model_info = self.download_and_hash_gguf(model_key)
        cfg = MODELS_CONFIG[model_key]
        self.server_log_path = BASE_WORK_DIR / f"llama_server_{model_key}.log"

        topo = detect_gpu_topology()
        gpu_count = topo["gpu_count"]
        total_vram_mb = topo["total_vram_mb"]

        extra_flags: list[str] = []
        if gpu_count >= 2:
            ngl_candidates = ["99"]
            extra_flags = ["--tensor-split", ",".join(["1"] * gpu_count)]
        elif gpu_count == 1 and total_vram_mb >= 21500:
            ngl_candidates = ["99"]
        elif gpu_count == 1 and total_vram_mb >= 14000:
            ngl_candidates = ["auto", "35", "30", "24"]
        else:
            ngl_candidates = ["0"]

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{LOCAL_EXEC_BIN_DIR}:{PERSISTENT_BIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        subcommand = ["serve"] if self.bin_path.name == "llama-app" else []

        for ngl in ngl_candidates:
            self.port = find_free_port(self.port)
            global LLAMA_SERVER_PORT, LLAMA_SERVER_URL
            LLAMA_SERVER_PORT = self.port
            LLAMA_SERVER_URL = f"http://127.0.0.1:{self.port}"

            cmd = [
                str(self.bin_path),
                *subcommand,
                "--model", str(self.current_gguf_path),
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--n-gpu-layers", ngl,
                *extra_flags,
                "--ctx-size", str(cfg["context_size"]),
                "--parallel", "1",
                "--threads", str(min(8, os.cpu_count() or 4)),
            ]

            t0 = time.time()
            log_file = open(self.server_log_path, "w", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

            ready = False
            exited_early = False
            for _ in range(180):
                if self.proc.poll() is not None:
                    exited_early = True
                    break
                try:
                    resp = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(2)

            if ready:
                self.model_load_duration_sec = round(time.time() - t0, 3)
                self.current_model_key = model_key
                log_content = self.server_log_path.read_text(encoding="utf-8", errors="ignore")
                offload_verified = ("offloaded" in log_content.lower()) or ("cuda" in log_content.lower())
                print(
                    f"✅ النموذج [{model_key}] جاهز على المنفذ {self.port} خلال {self.model_load_duration_sec}s | "
                    f"n_gpu_layers={ngl} | GPU Offload={offload_verified}"
                )
                return {
                    **model_info,
                    "model_load_duration_sec": self.model_load_duration_sec,
                    "n_gpu_layers": ngl,
                    "gpu_topology": topo,
                    "gpu_offload_verified": offload_verified,
                }

            if exited_early:
                print(f"  [Info] n_gpu_layers={ngl} لم يتسع في الذاكرة، جاري تجربة التقسيم الأكثر أماناً...")
                self.stop(delete_gguf=False)

        log_tail = self.server_log_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
        raise RuntimeError(f"Failed to start llama-server for {model_key}. Log tail:\n{log_tail}")

    def stop(self, delete_gguf: bool = True) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        subprocess.run(["pkill", "-f", "llama-server"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "llama-app"], stderr=subprocess.DEVNULL)
        if delete_gguf and self.current_gguf_path and self.current_gguf_path.exists():
            print(f"🧹 حذف ملف الأوزان المؤقت {self.current_gguf_path.name} لتفريغ مساحة القرص للمرحلة التالية...")
            self.current_gguf_path.unlink(missing_ok=True)
            self.current_gguf_path = None
        self.current_model_key = None



# ==============================================================================
# الخلية 4: قوالب التوجيه (Prompts)، استخراج الكود، فصل <think>، ومحلل JSON الموثوق
# ==============================================================================

def strip_thinking_blocks(raw_text: str) -> tuple[str, str]:
    """
    يفصل سلسلة التفكير الداخلية <think>...</think> (الخاصة بـ Qwen3.8 وDeepSeek-R1)
    عن الإجابة النهائية، ويعيد (thinking_text, clean_answer_text).
    """
    if not raw_text:
        return "", ""
    if "</think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        think_part = parts[0].replace("<think>", "").strip()
        answer_part = parts[1].strip()
        return think_part, answer_part
    return "", raw_text.strip()


def extract_clean_python_code(raw_text: str, prompt_code: str) -> tuple[str, bool, str]:
    """
    يستخرج كود Python النظيف من إجابة النموذج، ويتحقق من صلاحيته النحوية عبر ast.parse.
    يعيد: (final_code, syntax_valid, extraction_method)
    """
    _, answer = strip_thinking_blocks(raw_text)

    code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if code_blocks:
        sorted_blocks = sorted(code_blocks, key=lambda b: ("def " in b, len(b)), reverse=True)
        for blk in sorted_blocks:
            candidates.append((blk.strip(), "fenced_block"))
    candidates.append((answer.strip(), "raw_answer"))

    for cand, method in candidates:
        if "def " in cand:
            try:
                ast.parse(cand)
                return cand + "\n", True, method + "_full"
            except SyntaxError:
                pass
        combined = prompt_code.rstrip() + "\n" + cand + "\n"
        try:
            ast.parse(combined)
            return combined, True, method + "_merged_with_prompt"
        except SyntaxError:
            pass

    fallback_code = code_blocks[0].strip() if code_blocks else answer.strip()
    try:
        ast.parse(fallback_code)
        return fallback_code + "\n", True, "fallback_valid"
    except SyntaxError as e:
        return fallback_code + "\n", False, f"syntax_error: {e.msg}"


def parse_review_report_json(raw_text: str, review_type: str) -> tuple[dict, bool]:
    """
    يستخرج تقرير المراجعة المهيكل بصيغة JSON بعد عزل قسم <think>،
    مع آلية استرداد ذكية في حال وجود نص إضافي حول كائن JSON.
    يعيد: (parsed_report_dict, is_valid_json)
    """
    _, answer = strip_thinking_blocks(raw_text)

    json_blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    search_pool = json_blocks + [answer]

    for item in search_pool:
        item_str = item.strip()
        start_idx = item_str.find("{")
        end_idx = item_str.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate_json = item_str[start_idx : end_idx + 1]
            try:
                obj = json.loads(candidate_json)
                if isinstance(obj, dict):
                    obj.setdefault("review_type", review_type)
                    obj.setdefault("verdict", "PASS" if not obj.get("issues") else "NEEDS_REVISION")
                    obj.setdefault("issues", [])
                    return obj, True
            except Exception:
                continue

    fallback_obj = {
        "review_type": review_type,
        "verdict": "NEEDS_REVISION" if ("vulnerability" in answer.lower() or "bug" in answer.lower() or "error" in answer.lower()) else "PASS",
        "issues": [{"severity": "MEDIUM", "description": answer[:1500]}],
        "raw_parse_fallback": True,
    }
    return fallback_obj, False


def build_writer_messages(prompt_code: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python software engineer specializing in both functional correctness "
                "and secure coding practices. Complete the requested Python implementation. "
                "Return the complete, self-contained Python module (including all necessary imports and "
                "the full function implementation) inside a single ```python ... ``` block without extra commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Complete the following Python task according to its docstring specification. "
                "Ensure all edge cases and security constraints are properly handled:\n\n"
                f"```python\n{prompt_code}\n```"
            ),
        },
    ]


def build_functional_reviewer_messages(prompt_code: str, draft_code: str) -> list[dict]:
    schema_example = '{"verdict": "PASS" or "NEEDS_REVISION", "issues": [{"location": "...", "severity": "HIGH|MEDIUM|LOW", "description": "...", "fix_recommendation": "..."}]}'
    return [
        {
            "role": "system",
            "content": (
                "You are a Senior Functional Verification Reviewer. Analyze the candidate Python implementation "
                "strictly against the task specification and docstring requirements. Check for logic errors, "
                "return types, edge cases, exception handling, and missing imports. Do NOT rewrite the full code; "
                f"respond with a structured JSON report inside ```json ... ``` with keys: {schema_example}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### Task Specification:\n```python\n{prompt_code}\n```\n\n"
                f"### Candidate Implementation:\n```python\n{draft_code}\n```\n\n"
                "Provide your independent functional correctness review in JSON format."
            ),
        },
    ]


def build_security_reviewer_messages(prompt_code: str, draft_code: str) -> list[dict]:
    schema_example = '{"verdict": "PASS" or "NEEDS_REVISION", "cwe_candidates": ["CWE-..."], "issues": [{"location": "...", "severity": "CRITICAL|HIGH|MEDIUM", "description": "...", "fix_recommendation": "..."}]}'
    return [
        {
            "role": "system",
            "content": (
                "You are a Senior Application Security Auditor specializing in CWE vulnerability detection in Python. "
                "Inspect the candidate implementation for security flaws (e.g., path traversal, command/SQL/XPath injection, "
                "open redirect, unsafe deserialization, weak cryptography, regex DoS, improper input validation). "
                f"Respond with a structured JSON report inside ```json ... ``` with keys: {schema_example}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### Task Specification:\n```python\n{prompt_code}\n```\n\n"
                f"### Candidate Implementation:\n```python\n{draft_code}\n```\n\n"
                "Provide your independent security audit report in JSON format."
            ),
        },
    ]


def build_merger_messages(prompt_code: str, draft_code: str, func_report: dict, sec_report: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are the Lead Python Engineer. You wrote an initial implementation and received two independent "
                "peer-review reports: one from a Functional Reviewer and one from a Security Auditor. "
                "Synthesize both reviews, resolve any conflicts without breaking functional requirements or introducing "
                "security regressions, and output the final complete Python module inside a single ```python ... ``` block."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### Original Task Specification:\n```python\n{prompt_code}\n```\n\n"
                f"### Initial Implementation (A1 Draft):\n```python\n{draft_code}\n```\n\n"
                f"### Independent Functional Review Report:\n```json\n{json.dumps(func_report, indent=2, ensure_ascii=False)}\n```\n\n"
                f"### Independent Security Audit Report:\n```json\n{json.dumps(sec_report, indent=2, ensure_ascii=False)}\n```\n\n"
                "Produce the final revised Python module inside a single ```python ... ``` block."
            ),
        },
    ]




def get_active_cloud_api() -> Optional[dict[str, Any]]:
    if CF_ACCOUNT_ID and CF_API_TOKEN:
        return {"provider": "cloudflare_workers_ai", **CLOUD_API_PROVIDERS["cloudflare_workers_ai"]}
    if NVIDIA_NIM_API_KEY:
        return {"provider": "nvidia_nim", **CLOUD_API_PROVIDERS["nvidia_nim"]}
    if OPENROUTER_API_KEY:
        return {"provider": "openrouter", **CLOUD_API_PROVIDERS["openrouter"]}
    return None

# ==============================================================================
# الخلية 5: محرك الاستدعاء الذري المربوط بسجل SQLite WAL + JSON Checkpoints
# ==============================================================================

def query_gpu_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return round(sum(vals), 2)
    except Exception:
        return 0.0


def execute_or_resume_call(
    call_id: str,
    task_id: str,
    stage_name: str,
    arm: str,
    role: str,
    model_key: str,
    messages: list[dict[str, str]],
    temperature: float,
    seed: int,
    expect_type: str,  # "code" or "review_json"
    prompt_code: str = "",
    review_kind: str = "functional",
    max_tokens: int = 3072,
    mock_mode: bool = False,
) -> tuple[dict[str, Any], bool]:
    """
    يفحص أولاً سجل `LEDGER` الدائم:
    - إذا كان الاستدعاء منجزاً مسبقاً، يعيده في 0.001 ثانية مع `was_cached=True`.
    - إذا لم يكن منجزاً، يستدعي النموذج، ويستخرج الكود أو تقرير الـ JSON، ويحفظه ذرياً في SQLite + Google Drive JSON فوراً!
    """
    cached = LEDGER.get_completed_call(call_id)
    if cached is not None:
        return cached, True

    cfg = MODELS_CONFIG[model_key]
    t0 = time.time()
    vram_before = query_gpu_memory_mb()

    if mock_mode:
        time.sleep(0.002)
        if expect_type == "review_json":
            raw_content = (
                "<think>Analyzing code structure and security boundaries...</think>\n"
                "```json\n"
                + json.dumps(
                    {
                        "verdict": "NEEDS_REVISION",
                        "issues": [
                            {
                                "location": "main_function",
                                "severity": "HIGH",
                                "description": f"Verified by {role}",
                                "fix_recommendation": "Ensure strict input validation and safe return handling.",
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n```"
            )
        else:
            raw_content = (
                "<think>Drafting safe and functional implementation...</think>\n"
                f"```python\n{prompt_code.rstrip()}\n    return None\n```"
            )
        in_tokens = sum(len(m["content"].split()) for m in messages) * 2
        out_tokens = len(raw_content.split()) * 2
        status = "SUCCESS"
        error_msg = None
    else:
        cloud_api = get_active_cloud_api()
        if cloud_api is not None:
            target_url = f"{cloud_api['base_url']}/chat/completions"
            target_model = cloud_api["models"][model_key]
            headers = {
                "Authorization": f"Bearer {cloud_api['api_key']}",
                "Content-Type": "application/json",
            }
        else:
            target_url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
            target_model = cfg["canonical_name"]
            headers = {"Content-Type": "application/json"}

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature if temperature > 0 else 0.0,
            "top_p": 0.95 if temperature > 0 else 1.0,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        }
        status = "FAILED"
        raw_content = ""
        in_tokens = 0
        out_tokens = 0
        error_msg = None

        candidate_models = [target_model]
        if cloud_api is not None and "fallbacks" in cloud_api:
            candidate_models.extend(cloud_api["fallbacks"].get(model_key, []))
        model_idx = 0

        for attempt in range(1, 7):
            payload["model"] = candidate_models[min(model_idx, len(candidate_models) - 1)]
            try:
                if cloud_api is not None:
                    time.sleep(1.6)  # احترام حد 40 طلب/دقيقة المجاني في NVIDIA NIM لمنع أخطاء 429
                resp = requests.post(
                    target_url,
                    headers=headers,
                    json=payload,
                    timeout=600,
                )
                if resp.status_code == 404 and model_idx + 1 < len(candidate_models):
                    model_idx += 1
                    print(f"  [Fallback] Switching {model_key} to {candidate_models[model_idx]}...")
                    continue
                if resp.status_code == 429:
                    wait_s = 10 * attempt
                    print(f"  [Rate Limit 429] Waiting {wait_s}s before retrying {call_id}...")
                    time.sleep(wait_s)
                    continue
                if not resp.ok:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
                data = resp.json()
                msg_obj = data["choices"][0]["message"]
                msg_content = msg_obj.get("content") or ""
                msg_reasoning = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""
                if msg_reasoning and msg_content:
                    raw_content = f"<think>\n{msg_reasoning}\n</think>\n{msg_content}"
                elif msg_content:
                    raw_content = str(msg_content)
                elif msg_reasoning:
                    raw_content = f"<think>\n{msg_reasoning}\n</think>\n"
                else:
                    raise ValueError(f"Empty content and reasoning in API response: {str(data)[:400]}")
                usage = data.get("usage", {})
                in_tokens = int(usage.get("prompt_tokens", 0))
                out_tokens = int(usage.get("completion_tokens", 0))
                status = "SUCCESS"
                error_msg = None
                break
            except Exception as e:
                error_msg = f"Attempt {attempt} failed: {str(e)}"
                print(f"  [WARN] {call_id} -> {error_msg}")
                time.sleep(3 * attempt)

        if status != "SUCCESS":
            raise RuntimeError(f"LLM call {call_id} failed after retries: {error_msg}")

    elapsed_sec = round(time.time() - t0, 4)
    vram_after = query_gpu_memory_mb()
    thinking_text, clean_answer = strip_thinking_blocks(raw_content)

    record: dict[str, Any] = {
        "run_id": RUN_ID,
        "call_id": call_id,
        "task_id": task_id,
        "stage_name": stage_name,
        "arm": arm,
        "role": role,
        "model_key": model_key,
        "canonical_model": payload["model"] if not mock_mode else cfg["canonical_name"],
        "gguf_repo": cfg["hf_repo"],
        "gguf_filename": cfg["filename"],
        "quantization": cfg["quantization"],
        "temperature": temperature,
        "seed": seed,
        "status": status,
        "error": error_msg,
        "inference_seconds": elapsed_sec,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "gpu_vram_used_mb": max(vram_before, vram_after),
        "thinking_trace": thinking_text,
        "raw_response": raw_content,
        "clean_answer": clean_answer,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if expect_type == "code":
        ext_code, syn_valid, ext_method = extract_clean_python_code(raw_content, prompt_code)
        record["extracted_code"] = ext_code
        record["syntax_valid"] = syn_valid
        record["extraction_method"] = ext_method
    else:
        rep_json, rep_valid = parse_review_report_json(raw_content, review_kind)
        record["parsed_report"] = rep_json
        record["report_json_valid"] = rep_valid

    raw_content = raw_content or ""
    (RAW_LOGS_DIR / f"{call_id}.txt").write_text(str(raw_content), encoding="utf-8")
    LEDGER.save_completed_call(record)
    return record, False



# ==============================================================================
# الخلية 6: منسق المراحل الأربع الذكي (يتخطى المراحل والمهام المنجزة تلقائياً!)
# ==============================================================================

def run_full_multi_agent_experiment(tasks_manifest: list[dict[str, Any]], mock_mode: bool = False) -> dict[str, Any]:
    """
    ينفذ الدراسة الكاملة عبر 4 مراحل متتالية (275 استدعاءً إجمالياً لـ 25 مهمة) مع استئناف فوري:
    - قبل أي مرحلة: يفحص هل جميع استدعاءات هذه المرحلة موجودة في `LEDGER`؟
      * إذا كانت الإجابة نعم: يتخطى تنزيل النموذج وتشغيل الخادم لهذه المرحلة بالكامل في 0.01 ثانية!
      * إذا كانت الإجابة لا: يشغل النموذج ويكمل فقط الاستدعاءات المتبقية من حيث توقف بالضبط!
    """
    task_ids = [t["task_id"] for t in tasks_manifest]
    progress_summary = LEDGER.print_resume_dashboard(task_ids)

    cloud_api = get_active_cloud_api()
    use_local_gguf = (not mock_mode) and (cloud_api is None)
    if cloud_api is not None:
        print(f"🌐 [CLOUD API TURBO MODE ACTIVE] Provider: {cloud_api['provider']} (بدون تنزيل ملفات GGUF الضخمة وبدون الحاجة لـ GPU محلي!)")
    server: Optional[LlamaServerManager] = None
    if use_local_gguf:
        # إذا لم تكن جميع المراحل مكتملة، نجهز الملف التنفيذي لـ llama-server
        if any(done < exp for done, exp in progress_summary.values()):
            bin_path = ensure_llama_server_binary()
            server = LlamaServerManager(bin_path=bin_path, port=LLAMA_SERVER_PORT)

    metrics_file = BASE_WORK_DIR / "stage_load_metrics.json"
    stage_load_metrics: dict[str, Any] = (
        json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    )

    def persist_stage_metrics() -> None:
        metrics_file.write_text(json.dumps(stage_load_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ==========================================================================
    # STAGE 1: Writer Model (A1 + A2x4 + B_func_review + B_sec_review) = 175 calls
    # ==========================================================================
    s1_done, s1_exp = LEDGER.get_stage_progress(1, task_ids)
    if s1_done == s1_exp:
        print(f"⏭️ [STAGE 1/4 SKIPPED] جميع استدعاءات المرحلة الأولى ({s1_done}/{s1_exp}) مكتملة ومحفوظة مسبقاً!")
    else:
        print("\n" + "=" * 78)
        print(f"🚀 STAGE 1/4: Writer Model (Qwen3-Coder-30B-A3B) | استئناف من {s1_done}/{s1_exp}")
        print("=" * 78)
        if use_local_gguf and server is not None:
            stage_load_metrics["stage1_writer"] = server.start("writer")
        else:
            stage_load_metrics.setdefault("stage1_writer", {"model_load_duration_sec": 0.0, "gpu_offload_verified": True})
        persist_stage_metrics()

        for idx, task in enumerate(tasks_manifest, 1):
            tid = task["task_id"]
            pcode = task["prompt_code"]

            # 1. A1 Draft
            r_a1, c_a1 = execute_or_resume_call(
                call_id=f"{tid}__A1_draft",
                task_id=tid,
                stage_name="stage1_writer",
                arm="A1",
                role="Writer_Initial",
                model_key="writer",
                messages=build_writer_messages(pcode),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="code",
                prompt_code=pcode,
                mock_mode=mock_mode,
            )
            a1_code = r_a1["extracted_code"]

            # 2. A2 Four Independent Samples
            c_a2_list = []
            for s_idx in range(4):
                _, c_s = execute_or_resume_call(
                    call_id=f"{tid}__A2_sample_{s_idx}",
                    task_id=tid,
                    stage_name="stage1_writer",
                    arm="A2",
                    role=f"Writer_Sample_{s_idx}",
                    model_key="writer",
                    messages=build_writer_messages(pcode),
                    temperature=0.8,
                    seed=101 + s_idx,
                    expect_type="code",
                    prompt_code=pcode,
                    mock_mode=mock_mode,
                )
                c_a2_list.append(c_s)

            # 3. Arm B Homogeneous Functional Self-Review
            _, c_bf = execute_or_resume_call(
                call_id=f"{tid}__B_func_review",
                task_id=tid,
                stage_name="stage1_writer",
                arm="B",
                role="Homogeneous_Functional_Reviewer",
                model_key="writer",
                messages=build_functional_reviewer_messages(pcode, a1_code),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="review_json",
                review_kind="functional",
                mock_mode=mock_mode,
            )

            # 4. Arm B Homogeneous Security Self-Review
            _, c_bs = execute_or_resume_call(
                call_id=f"{tid}__B_sec_review",
                task_id=tid,
                stage_name="stage1_writer",
                arm="B",
                role="Homogeneous_Security_Auditor",
                model_key="writer",
                messages=build_security_reviewer_messages(pcode, a1_code),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="review_json",
                review_kind="security",
                mock_mode=mock_mode,
            )

            all_cached = c_a1 and all(c_a2_list) and c_bf and c_bs
            status_tag = "⚡ تم التخطي (محفوظ مسبقاً)" if all_cached else "💾 تم التوليد والحفظ الفوري"
            print(f"  [Stage 1 | Task {idx:02d}/25 | {tid}] -> {status_tag}")

        if use_local_gguf and server is not None:
            server.stop(delete_gguf=True)

    # ==========================================================================
    # STAGE 2: Heterogeneous Functional Reviewer (Qwen3.8-27B) = 25 calls
    # ==========================================================================
    s2_done, s2_exp = LEDGER.get_stage_progress(2, task_ids)
    if s2_done == s2_exp:
        print(f"⏭️ [STAGE 2/4 SKIPPED] جميع استدعاءات المرحلة الثانية ({s2_done}/{s2_exp}) مكتملة ومحفوظة مسبقاً!")
    else:
        print("\n" + "=" * 78)
        print(f"🚀 STAGE 2/4: Functional Reviewer (Qwen3.8-27B) | استئناف من {s2_done}/{s2_exp}")
        print("=" * 78)
        if use_local_gguf and server is not None:
            stage_load_metrics["stage2_func_reviewer"] = server.start("func_reviewer")
        else:
            stage_load_metrics.setdefault("stage2_func_reviewer", {"model_load_duration_sec": 0.0, "gpu_offload_verified": True})
        persist_stage_metrics()

        for idx, task in enumerate(tasks_manifest, 1):
            tid = task["task_id"]
            pcode = task["prompt_code"]
            a1_rec = LEDGER.get_completed_call(f"{tid}__A1_draft")
            assert a1_rec is not None, f"Missing A1 draft for {tid}"

            _, c_cf = execute_or_resume_call(
                call_id=f"{tid}__C_func_review",
                task_id=tid,
                stage_name="stage2_func_reviewer",
                arm="C",
                role="Heterogeneous_Functional_Reviewer",
                model_key="func_reviewer",
                messages=build_functional_reviewer_messages(pcode, a1_rec["extracted_code"]),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="review_json",
                review_kind="functional",
                mock_mode=mock_mode,
            )
            status_tag = "⚡ تم التخطي (محفوظ مسبقاً)" if c_cf else "💾 تم التوليد والحفظ الفوري"
            print(f"  [Stage 2 | Task {idx:02d}/25 | {tid}] -> {status_tag}")

        if use_local_gguf and server is not None:
            server.stop(delete_gguf=True)

    # ==========================================================================
    # STAGE 3: Heterogeneous Security Reviewer (DeepSeek-R1-Distill-Qwen-32B) = 25 calls
    # ==========================================================================
    s3_done, s3_exp = LEDGER.get_stage_progress(3, task_ids)
    if s3_done == s3_exp:
        print(f"⏭️ [STAGE 3/4 SKIPPED] جميع استدعاءات المرحلة الثالثة ({s3_done}/{s3_exp}) مكتملة ومحفوظة مسبقاً!")
    else:
        print("\n" + "=" * 78)
        print(f"🚀 STAGE 3/4: Security Reviewer (DeepSeek-R1-32B) | استئناف من {s3_done}/{s3_exp}")
        print("=" * 78)
        if use_local_gguf and server is not None:
            stage_load_metrics["stage3_sec_reviewer"] = server.start("sec_reviewer")
        else:
            stage_load_metrics.setdefault("stage3_sec_reviewer", {"model_load_duration_sec": 0.0, "gpu_offload_verified": True})
        persist_stage_metrics()

        for idx, task in enumerate(tasks_manifest, 1):
            tid = task["task_id"]
            pcode = task["prompt_code"]
            a1_rec = LEDGER.get_completed_call(f"{tid}__A1_draft")
            assert a1_rec is not None, f"Missing A1 draft for {tid}"

            _, c_cs = execute_or_resume_call(
                call_id=f"{tid}__C_sec_review",
                task_id=tid,
                stage_name="stage3_sec_reviewer",
                arm="C",
                role="Heterogeneous_Security_Auditor",
                model_key="sec_reviewer",
                messages=build_security_reviewer_messages(pcode, a1_rec["extracted_code"]),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="review_json",
                review_kind="security",
                mock_mode=mock_mode,
            )
            status_tag = "⚡ تم التخطي (محفوظ مسبقاً)" if c_cs else "💾 تم التوليد والحفظ الفوري"
            print(f"  [Stage 3 | Task {idx:02d}/25 | {tid}] -> {status_tag}")

        if use_local_gguf and server is not None:
            server.stop(delete_gguf=True)

    # ==========================================================================
    # STAGE 4: Writer Model Merger (Arm B Final + Arm C Final) = 50 calls
    # ==========================================================================
    s4_done, s4_exp = LEDGER.get_stage_progress(4, task_ids)
    if s4_done == s4_exp:
        print(f"⏭️ [STAGE 4/4 SKIPPED] جميع استدعاءات المرحلة الرابعة ({s4_done}/{s4_exp}) مكتملة ومحفوظة مسبقاً!")
    else:
        print("\n" + "=" * 78)
        print(f"🚀 STAGE 4/4: Writer Merger (Qwen3-Coder-30B-A3B) | استئناف من {s4_done}/{s4_exp}")
        print("=" * 78)
        if use_local_gguf and server is not None:
            stage_load_metrics["stage4_writer_merge"] = server.start("writer")
        else:
            stage_load_metrics.setdefault("stage4_writer_merge", {"model_load_duration_sec": 0.0, "gpu_offload_verified": True})
        persist_stage_metrics()

        for idx, task in enumerate(tasks_manifest, 1):
            tid = task["task_id"]
            pcode = task["prompt_code"]
            a1_rec = LEDGER.get_completed_call(f"{tid}__A1_draft")
            bf_rec = LEDGER.get_completed_call(f"{tid}__B_func_review")
            bs_rec = LEDGER.get_completed_call(f"{tid}__B_sec_review")
            cf_rec = LEDGER.get_completed_call(f"{tid}__C_func_review")
            cs_rec = LEDGER.get_completed_call(f"{tid}__C_sec_review")
            assert all(x is not None for x in (a1_rec, bf_rec, bs_rec, cf_rec, cs_rec)), f"Missing upstream records for {tid}"

            _, c_bm = execute_or_resume_call(
                call_id=f"{tid}__B_merged",
                task_id=tid,
                stage_name="stage4_writer_merge",
                arm="B",
                role="Writer_Merger_B",
                model_key="writer",
                messages=build_merger_messages(pcode, a1_rec["extracted_code"], bf_rec["parsed_report"], bs_rec["parsed_report"]),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="code",
                prompt_code=pcode,
                mock_mode=mock_mode,
            )

            _, c_cm = execute_or_resume_call(
                call_id=f"{tid}__C_merged",
                task_id=tid,
                stage_name="stage4_writer_merge",
                arm="C",
                role="Writer_Merger_C",
                model_key="writer",
                messages=build_merger_messages(pcode, a1_rec["extracted_code"], cf_rec["parsed_report"], cs_rec["parsed_report"]),
                temperature=0.0,
                seed=GLOBAL_SEED,
                expect_type="code",
                prompt_code=pcode,
                mock_mode=mock_mode,
            )
            status_tag = "⚡ تم التخطي (محفوظ مسبقاً)" if (c_bm and c_cm) else "💾 تم التوليد والحفظ الفوري"
            print(f"  [Stage 4 | Task {idx:02d}/25 | {tid}] -> {status_tag}")

        if use_local_gguf and server is not None:
            server.stop(delete_gguf=True)

    LEDGER.print_resume_dashboard(task_ids)
    return stage_load_metrics



# ==============================================================================
# الخلية 7: تصدير الحزمة المتوافقة مع CWEval وحفظ النسخة النهائية في Google Drive
# ==============================================================================

def export_cweval_compatible_bundle(tasks_manifest: list[dict[str, Any]]) -> Path:
    evals_root = EXPORT_DIR / "evals"
    if evals_root.exists():
        shutil.rmtree(evals_root)

    arms_dirs = {
        "A1": evals_root / "arm_A1_baseline",
        "A2": evals_root / "arm_A2_samples4",
        "A2_Index0": evals_root / "arm_A2_Index0",
        "A2_SyntaxSelected": evals_root / "arm_A2_SyntaxSelected",
        "B": evals_root / "arm_B_homogeneous",
        "C": evals_root / "arm_C_heterogeneous",
    }

    def write_cweval_candidate(arm_dir: Path, sample_idx: int, task_id: str, clean_code: str) -> None:
        target_dir = arm_dir / f"generated_{sample_idx}" / "core" / "py"
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_wrapped = f"```python\n{clean_code.strip()}\n```\n"
        (target_dir / f"{task_id}_raw.py").write_text(raw_wrapped, encoding="utf-8")
        (target_dir / f"{task_id}_task.py").write_text(clean_code.strip() + "\n", encoding="utf-8")

    task_telemetry_rows = []

    for task in tasks_manifest:
        tid = task["task_id"]
        a1 = LEDGER.get_completed_call(f"{tid}__A1_draft")
        a2_samples = [LEDGER.get_completed_call(f"{tid}__A2_sample_{i}") for i in range(4)]
        bf = LEDGER.get_completed_call(f"{tid}__B_func_review")
        bs = LEDGER.get_completed_call(f"{tid}__B_sec_review")
        bm = LEDGER.get_completed_call(f"{tid}__B_merged")
        cf = LEDGER.get_completed_call(f"{tid}__C_func_review")
        cs = LEDGER.get_completed_call(f"{tid}__C_sec_review")
        cm = LEDGER.get_completed_call(f"{tid}__C_merged")

        write_cweval_candidate(arms_dirs["A1"], 0, tid, a1["extracted_code"])

        for i, s_rec in enumerate(a2_samples):
            write_cweval_candidate(arms_dirs["A2"], i, tid, s_rec["extracted_code"])

        write_cweval_candidate(arms_dirs["A2_Index0"], 0, tid, a2_samples[0]["extracted_code"])

        syntax_selected_rec = a2_samples[0]
        syntax_selected_idx = 0
        for i, s_rec in enumerate(a2_samples):
            if s_rec.get("syntax_valid", False):
                syntax_selected_rec = s_rec
                syntax_selected_idx = i
                break
        write_cweval_candidate(arms_dirs["A2_SyntaxSelected"], 0, tid, syntax_selected_rec["extracted_code"])

        write_cweval_candidate(arms_dirs["B"], 0, tid, bm["extracted_code"])
        write_cweval_candidate(arms_dirs["C"], 0, tid, cm["extracted_code"])

        a2_in_tok = sum(x["input_tokens"] for x in a2_samples)
        a2_out_tok = sum(x["output_tokens"] for x in a2_samples)
        a2_time = round(sum(x["inference_seconds"] for x in a2_samples), 4)

        b_in_tok = a1["input_tokens"] + bf["input_tokens"] + bs["input_tokens"] + bm["input_tokens"]
        b_out_tok = a1["output_tokens"] + bf["output_tokens"] + bs["output_tokens"] + bm["output_tokens"]
        b_time = round(a1["inference_seconds"] + bf["inference_seconds"] + bs["inference_seconds"] + bm["inference_seconds"], 4)

        c_in_tok = a1["input_tokens"] + cf["input_tokens"] + cs["input_tokens"] + cm["input_tokens"]
        c_out_tok = a1["output_tokens"] + cf["output_tokens"] + cs["output_tokens"] + cm["output_tokens"]
        c_time = round(a1["inference_seconds"] + cf["inference_seconds"] + cs["inference_seconds"] + cm["inference_seconds"], 4)

        task_telemetry_rows.append({
            "task_id": tid,
            "A1_syntax_valid": int(a1["syntax_valid"]),
            "A1_time_sec": a1["inference_seconds"],
            "A1_in_tokens": a1["input_tokens"],
            "A1_out_tokens": a1["output_tokens"],
            "A2_valid_samples_count": sum(int(x["syntax_valid"]) for x in a2_samples),
            "A2_syntax_selected_index": syntax_selected_idx,
            "A2_total_time_sec": a2_time,
            "A2_total_in_tokens": a2_in_tok,
            "A2_total_out_tokens": a2_out_tok,
            "B_func_review_time_sec": bf["inference_seconds"],
            "B_sec_review_time_sec": bs["inference_seconds"],
            "B_merge_time_sec": bm["inference_seconds"],
            "B_total_time_sec": b_time,
            "B_total_in_tokens": b_in_tok,
            "B_total_out_tokens": b_out_tok,
            "B_func_json_valid": int(bf["report_json_valid"]),
            "B_sec_json_valid": int(bs["report_json_valid"]),
            "B_final_syntax_valid": int(bm["syntax_valid"]),
            "C_func_review_time_sec": cf["inference_seconds"],
            "C_sec_review_time_sec": cs["inference_seconds"],
            "C_merge_time_sec": cm["inference_seconds"],
            "C_total_time_sec": c_time,
            "C_total_in_tokens": c_in_tok,
            "C_total_out_tokens": c_out_tok,
            "C_func_json_valid": int(cf["report_json_valid"]),
            "C_sec_json_valid": int(cs["report_json_valid"]),
            "C_final_syntax_valid": int(cm["syntax_valid"]),
        })

    telemetry_csv_path = EXPORT_DIR / "generation_telemetry_25tasks.csv"
    with open(telemetry_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_telemetry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_telemetry_rows)

    for meta_file in [
        "environment_manifest.json",
        "tasks_25_manifest.json",
        "models_sha256_registry.json",
        "stage_load_metrics.json",
        "llama_cpp_build_info.json",
    ]:
        src_p = BASE_WORK_DIR / meta_file
        if src_p.exists():
            shutil.copy2(src_p, EXPORT_DIR / meta_file)

    shutil.copytree(CHECKPOINTS_DIR, EXPORT_DIR / "checkpoints", dirs_exist_ok=True)
    shutil.copytree(PROMPTS_DIR, EXPORT_DIR / "sanitized_prompts", dirs_exist_ok=True)

    # حفظ الحزمة المضغوطة داخل مجلد العمل الدائم (Google Drive) وكذلك في المجلد المحلي لسهولة التنزيل
    bundle_base = BASE_WORK_DIR / "CWEval_Candidates_Bundle"
    zip_path = Path(shutil.make_archive(str(bundle_base), "zip", root_dir=str(EXPORT_DIR)))
    if IS_COLAB and Path("/content").exists() and zip_path.parent != Path("/content"):
        shutil.copy2(zip_path, Path("/content/CWEval_Candidates_Bundle.zip"))

    bundle_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"\n🎉 === اكتملت التجربة وتصدير الحزمة بنجاح === 🎉")
    print(f"📦 المسار الدائم للحزمة (في Google Drive): {zip_path}")
    print(f"🔒 بصمة SHA256 للحزمة: {bundle_sha256}")
    return zip_path


if IS_KAGGLE or IS_COLAB or ("ipykernel" in sys.modules) or os.environ.get("RUN_FULL_EXPERIMENT", "0") == "1":
    if not TASKS_MANIFEST:
        TASKS_MANIFEST = load_or_restore_25_python_tasks(CWEVAL_REPO_DIR)
    MOCK_FLAG = os.environ.get("CWEVAL_MOCK_MODE", "0") == "1"
    run_full_multi_agent_experiment(TASKS_MANIFEST, mock_mode=MOCK_FLAG)
    export_cweval_compatible_bundle(TASKS_MANIFEST)