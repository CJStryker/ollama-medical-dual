"""
Clinic Roleplay Engine (Patient ↔ Nurse Zesty ↔ Doctor Scarlett)

Key reliability upgrades vs your original:
- Hard timeouts for every Ollama call (prevents indefinite hangs)
- Spinner + phase labels so you always know what it’s doing
- Retry + exponential backoff on timeouts/transient errors
- Automatic model failover (tries fallback models if primary is stuck)
- Doctor consult is optional (continues nurse-only if doctor model is down)
- Rolling conversation window + message size caps (prevents prompt bloat)
- Debug tooling: /ping, /health, /models, /debug, /setmodel, /settimeout, /compact
- Graceful Ctrl+C handling during generation (returns to prompt)

Before running:
- Ensure Ollama is reachable (local default or set OLLAMA_HOST env var)
- Ensure your model names are correct (PopPooB-Dr:latest / PopPooB-Nurse:latest or change below)
"""

import os
import sys
import time
import json
import threading
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from http import HTTPStatus

try:
    from flask import Flask, jsonify, request, Response
except Exception:
    Flask = None
    jsonify = None
    request = None
    Response = None



# =====================
# CONFIG
# =====================

DEFAULT_NURSE_MODEL = os.getenv("NURSE_MODEL", "PopPooB-Nurse:latest")
DEFAULT_DOCTOR_MODEL = os.getenv("DOCTOR_MODEL", "PopPooB-Dr:latest")

# Fallback models to try if the primary model stalls.
# IMPORTANT: Put models you actually have installed on your Ollama server.
FALLBACK_MODELS = [
    os.getenv("FALLBACK_MODEL_1", "llama3:8b"),
    os.getenv("FALLBACK_MODEL_2", "phi3:mini"),
]

# Hardcoded default host so the app works without local Ollama setup.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://96.242.172.92:11434").rstrip("/")
FALLBACK_OLLAMA_HOSTS = [
    os.getenv("FALLBACK_OLLAMA_HOST_1", "http://127.0.0.1:11434").rstrip("/"),
    os.getenv("FALLBACK_OLLAMA_HOST_2", "").rstrip("/"),
]
ACTIVE_OLLAMA_HOST = OLLAMA_HOST
VISIT_DB_PATH = Path(os.getenv("VISIT_DB_PATH", "visits_db.json"))

API_STATS: Dict[str, Any] = {
    "calls": 0,
    "success": 0,
    "fail": 0,
    "latency_ms_total": 0.0,
    "last_error": None,
    "last_host": None,
}


# =====================
# ROLE PROMPTS
# =====================

EMERGENCY_OVERRIDE_MESSAGE = (
    "This may be a medical emergency. Do not use this tool. "
    "Call 911 or go to the nearest emergency room immediately."
)

ZESTY_PROMPT = """
You are “Zesty”, an AI nurse performing a structured intake in a fictional clinic simulation.

GLOBAL SAFETY RULES (MANDATORY):
- This is roleplay / educational only, not real medical care.
- Never diagnose.
- Never prescribe medication or dosage.
- Never override advice from real clinicians.
- Never say “you are fine”.
- Never handle emergencies in roleplay; escalate instead.
- Use cautious language: “may”, “could”, “might”.
- Prioritize user safety over immersion.

PERSONALITY:
- Calm, slightly warm, focused.
- Efficient but not robotic.
- Supportive tone, not overly emotional.

ROLE:
- Gather structured symptom information.
- Ask relevant follow-up questions.
- Clarify missing details.
- Prepare the user for a doctor-style summary.

OUTPUT FORMAT:
1. Intake Summary:
- Restate symptoms clearly.
2. Key Details Collected:
- Onset
- Location
- Severity
- Duration
- Triggers
- Associated symptoms
3. Follow-up Questions:
- Ask 2–4 targeted questions only.
4. Gentle Guidance:
- Suggest whether symptoms might need evaluation (without diagnosing).

FOLLOW-UP MODE (after doctor response):
- Translate doctor note into simpler language.
- Suggest care-seeking next steps (not treatment).
- Keep it short and clear.

EMERGENCY OVERRIDE:
If the message suggests chest pain, breathing difficulty, stroke symptoms, seizure, suicidal thoughts, heavy bleeding, overdose, or allergic reaction, output only:
\"\"\"This may be a medical emergency. Do not use this tool. Call 911 or go to the nearest emergency room immediately.\"\"\"
"""

SCARLETT_PROMPT = """
You are “Dr. Scarlett”, an AI that generates a structured doctor-style note based on intake in educational roleplay.

GLOBAL SAFETY RULES (MANDATORY):
- This is roleplay / educational only, not real medical care.
- Never diagnose.
- Never prescribe medication or dosage.
- Never override advice from real clinicians.
- Never say “you are fine”.
- Never handle emergencies in roleplay; escalate instead.
- Use cautious language: “may”, “could”, “might”.
- Prioritize user safety over immersion.

PERSONALITY:
- Precise, analytical, composed.
- Slightly formal tone.
- Clear and confident but not authoritative.

ROLE:
- Interpret intake data.
- Organize into structured clinical-style note.
- Highlight possible concerns without diagnosing.

OUTPUT FORMAT:
1. Summary of Presentation:
- Concise restatement of issue.
2. Possible Considerations:
- Use cautious language only (“could represent…”).
- List 2–4 possibilities max.
3. Risk Signals:
- Identify concerning features.
4. Suggested Next Steps:
- Consider evaluation by a healthcare professional.
- Urgent care may be appropriate if symptoms worsen.
5. Safety Note:
- Remind that this is not a diagnosis.

EMERGENCY OVERRIDE:
If the message suggests chest pain, breathing difficulty, stroke symptoms, seizure, suicidal thoughts, heavy bleeding, overdose, or allergic reaction, output only:
\"\"\"This may be a medical emergency. Do not use this tool. Call 911 or go to the nearest emergency room immediately.\"\"\"
"""


# =====================
# UTIL: printing / spinner
# =====================

PRINT_LOCK = threading.Lock()

def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)
        sys.stdout.flush()


class Spinner:
    def __init__(self, label: str = "Working"):
        self.label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        with PRINT_LOCK:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

    def _run(self):
        frames = "|/-\\"
        i = 0
        while not self._stop.is_set():
            with PRINT_LOCK:
                sys.stdout.write(f"\r{self.label}… {frames[i % len(frames)]}")
                sys.stdout.flush()
            i += 1
            time.sleep(0.1)


# =====================
# UTIL: timed ollama calls with retry + backoff + failover
# =====================

def _ollama_api_post(path: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for host in get_candidate_hosts():
        url = f"{host}{path}"
        t0 = time.perf_counter()
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            _record_api_stat(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, host=host)
            _set_active_host(host)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON from {url}: {raw[:200]}") from e
        except urlerror.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            last_err = RuntimeError(f"HTTP {e.code} from {url}: {detail}")
        except urlerror.URLError as e:
            last_err = RuntimeError(f"Failed connecting to {url}: {e}")
        except Exception as e:
            last_err = e

        _record_api_stat(ok=False, latency_ms=(time.perf_counter() - t0) * 1000, host=host, err=last_err)
        continue

    raise last_err if last_err else RuntimeError("No Ollama host candidates available.")


def _ollama_api_get(path: str, timeout_s: int = 30) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for host in get_candidate_hosts():
        url = f"{host}{path}"
        t0 = time.perf_counter()
        req = urlrequest.Request(url, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            _record_api_stat(ok=True, latency_ms=(time.perf_counter() - t0) * 1000, host=host)
            _set_active_host(host)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON from {url}: {raw[:200]}") from e
        except urlerror.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            last_err = RuntimeError(f"HTTP {e.code} from {url}: {detail}")
        except urlerror.URLError as e:
            last_err = RuntimeError(f"Failed connecting to {url}: {e}")
        except Exception as e:
            last_err = e

        _record_api_stat(ok=False, latency_ms=(time.perf_counter() - t0) * 1000, host=host, err=last_err)
        continue

    raise last_err if last_err else RuntimeError("No Ollama host candidates available.")


def get_candidate_hosts() -> List[str]:
    hosts = [OLLAMA_HOST] + FALLBACK_OLLAMA_HOSTS
    seen = set()
    cleaned = []
    for h in hosts:
        hh = (h or "").strip().rstrip("/")
        if not hh or hh in seen:
            continue
        seen.add(hh)
        cleaned.append(hh)
    return cleaned


def _set_active_host(host: str):
    global ACTIVE_OLLAMA_HOST
    ACTIVE_OLLAMA_HOST = host


def _record_api_stat(ok: bool, latency_ms: float, host: str, err: Optional[Exception] = None):
    API_STATS["calls"] += 1
    API_STATS["latency_ms_total"] += max(0.0, float(latency_ms))
    API_STATS["last_host"] = host
    if ok:
        API_STATS["success"] += 1
    else:
        API_STATS["fail"] += 1
        API_STATS["last_error"] = str(err) if err else "unknown"


def set_ollama_host(host: str):
    global OLLAMA_HOST
    h = (host or "").strip().rstrip("/")
    if not h.startswith("http://") and not h.startswith("https://"):
        raise ValueError("Host must start with http:// or https://")
    OLLAMA_HOST = h
    _set_active_host(h)


def get_api_stats_snapshot() -> Dict[str, Any]:
    calls = API_STATS["calls"]
    avg_ms = (API_STATS["latency_ms_total"] / calls) if calls else 0.0
    return {
        "calls": calls,
        "success": API_STATS["success"],
        "fail": API_STATS["fail"],
        "avg_latency_ms": round(avg_ms, 2),
        "last_host": API_STATS["last_host"],
        "last_error": API_STATS["last_error"],
    }


def _ollama_chat_worker(model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
                        out: Dict[str, Any], err: Dict[str, Exception]):
    try:
        resp = _ollama_api_post(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "options": options,
                "stream": False,
            },
            timeout_s=360,
        )
        out["text"] = ((resp.get("message") or {}).get("content") or "")
    except Exception as e:
        err["e"] = e


def timed_ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any],
    timeout_s: int = 45,
    retries: int = 1,
    backoff_base_s: float = 0.7
) -> str:
    """
    Runs ollama.chat in a thread and enforces a hard timeout.
    Retries on timeout / transient errors with exponential backoff.
    """
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        out: Dict[str, Any] = {}
        err: Dict[str, Exception] = {}
        t = threading.Thread(
            target=_ollama_chat_worker,
            args=(model, messages, options, out, err),
            daemon=True
        )
        t.start()
        t.join(timeout=timeout_s)

        if t.is_alive():
            last_err = TimeoutError(f"Ollama call timed out after {timeout_s}s (attempt {attempt+1}/{retries+1}).")
        elif "e" in err:
            last_err = err["e"]
        else:
            return (out.get("text") or "").strip()

        # Backoff before retrying
        if attempt < retries:
            sleep_s = backoff_base_s * (2 ** attempt)
            time.sleep(sleep_s)

    raise last_err if last_err else RuntimeError("Unknown Ollama error.")


def try_models_with_failover(
    models_to_try: List[str],
    messages: List[Dict[str, str]],
    options: Dict[str, Any],
    timeout_s: int,
    retries: int
) -> Tuple[str, str]:
    """
    Try a list of models; return (text, model_used).
    """
    last_err: Optional[Exception] = None
    for m in models_to_try:
        if not m:
            continue
        try:
            text = timed_ollama_chat(
                model=m,
                messages=messages,
                options=options,
                timeout_s=timeout_s,
                retries=retries,
            )
            return text, m
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("All models failed.")


def ollama_list_models(timeout_s: int = 10) -> List[str]:
    """
    Lists available models via Ollama /api/tags.
    If the server is wedged, this can also hang; so we timebox it.
    """
    out: Dict[str, Any] = {}
    err: Dict[str, Exception] = {}

    def worker():
        try:
            out["resp"] = _ollama_api_get("/api/tags", timeout_s=timeout_s)
        except Exception as e:
            err["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise TimeoutError(f"ollama.list() timed out after {timeout_s}s.")
    if "e" in err:
        raise err["e"]

    resp = out.get("resp") or {}
    models = resp.get("models") or []
    names = []
    for item in models:
        name = item.get("name")
        if name:
            names.append(name)
    return names


# =====================
# AGENT CLASS
# =====================

class Agent:
    """
    A chat agent wrapper with:
      - rolling history
      - message caps
      - failover models
      - timeouts
    """
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.6,
        max_history: int = 16,
        max_memory: int = 20,
        timeout_s: int = 60,
        retries: int = 1,
        failover_models: Optional[List[str]] = None,
        use_spinner: bool = True,
    ):
        self.name = name
        self.model = model
        self.base_prompt = system_prompt
        self.temperature = temperature
        self.max_history = max_history
        self.max_memory = max_memory
        self.timeout_s = timeout_s
        self.retries = retries
        self.failover_models = failover_models or []
        self.use_spinner = use_spinner

        self.memory: List[str] = []
        self.mood = "neutral"
        self.style = "default"

        self.messages: List[Dict[str, str]] = []
        self._rebuild_system()

    def _rebuild_system(self):
        system = f"""{self.base_prompt}

CURRENT MOOD: {self.mood}
CURRENT STYLE: {self.style}

MEMORY (keep brief):
{self.format_memory()}
"""
        non_system = [m for m in self.messages if m.get("role") != "system"]
        self.messages = [{"role": "system", "content": system}] + non_system
        self._trim_history()

    def _trim_history(self):
        system = self.messages[0]
        rest = self.messages[1:]
        if len(rest) > self.max_history:
            rest = rest[-self.max_history:]
        self.messages = [system] + rest

    def format_memory(self) -> str:
        if not self.memory:
            return "None"
        return "\n".join(f"- {m}" for m in self.memory[-self.max_memory:])

    def add_memory(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        if len(text) > 240:
            text = text[:240].rstrip() + "…"
        self.memory.append(text)
        self._rebuild_system()

    def set_mood(self, mood: str):
        self.mood = (mood or "neutral").strip()
        self._rebuild_system()

    def set_style(self, style: str):
        self.style = (style or "default").strip()
        self._rebuild_system()

    def receive(self, role: str, content: str):
        content = (content or "").strip()
        if not content:
            return
        # cap message size (prevents prompt bloat)
        if len(content) > 3500:
            content = content[:3500].rstrip() + "…"
        self.messages.append({"role": role, "content": content})
        self._trim_history()

    def compact_history(self, keep_last_n: int = 8):
        """
        Drop older non-system messages more aggressively.
        """
        system = self.messages[0:1]
        rest = self.messages[1:]
        if len(rest) > keep_last_n:
            rest = rest[-keep_last_n:]
        self.messages = system + rest

    def respond(self, phase_label: str = "generating") -> str:
        spinner: Optional[Spinner] = None
        if self.use_spinner:
            spinner = Spinner(label=f"{self.name} {phase_label}")
            spinner.start()

        try:
            models_to_try = [self.model] + [m for m in self.failover_models if m and m != self.model]
            options = {
                "temperature": self.temperature,
                "repeat_penalty": 1.6,
            }

            text, used = try_models_with_failover(
                models_to_try=models_to_try,
                messages=self.messages,
                options=options,
                timeout_s=self.timeout_s,
                retries=self.retries,
            )

            # If failover used, lock to it to keep session stable
            self.model = used

        finally:
            if spinner:
                spinner.stop()

        self.messages.append({"role": "assistant", "content": text})
        self._trim_history()
        return text

    def reset(self):
        self.memory.clear()
        self.messages = []
        self._rebuild_system()


# =====================
# CLINIC FLOW
# =====================

def consult_doctor(doctor: Agent, patient_message: str, nurse_reply: str) -> str:
    """
    Tight consult payload to prevent big prompts.
    """
    consult = (
        "NURSE-TO-DOCTOR CONSULT\n"
        f"Patient said: {patient_message}\n"
        f"My (nurse) reply: {nurse_reply}\n"
        "Please provide your Doctor Note."
    )
    doctor.receive("user", consult)
    return doctor.respond(phase_label="consulting")

def nurse_update_from_doctor(nurse: Agent, doctor_note: str) -> None:
    nurse.receive("user", "DOCTOR (Scarlett) guidance for you:\n" + doctor_note)


class ClinicSession:
    """
    Reusable session API for web or service integrations.
    Call process_patient_message(...) repeatedly with follow-up info.
    """
    def __init__(self, use_spinner: bool = False, patient_id: str = "anonymous"):
        self.nurse = Agent(
            name="Zesty (Nurse)",
            model=DEFAULT_NURSE_MODEL,
            system_prompt=ZESTY_PROMPT,
            temperature=0.65,
            max_history=18,
            max_memory=20,
            timeout_s=60,
            retries=1,
            failover_models=FALLBACK_MODELS,
            use_spinner=use_spinner,
        )
        self.doctor = Agent(
            name="Scarlett (Doctor)",
            model=DEFAULT_DOCTOR_MODEL,
            system_prompt=SCARLETT_PROMPT,
            temperature=0.55,
            max_history=12,
            max_memory=12,
            timeout_s=60,
            retries=1,
            failover_models=FALLBACK_MODELS,
            use_spinner=use_spinner,
        )
        self.visit_turns: List[Dict[str, Any]] = []
        self.patient_id = (patient_id or "anonymous").strip()
        self.case_snapshot: Dict[str, Any] = {
            "started_at": datetime.utcnow().isoformat() + "Z",
            "patient_id": self.patient_id,
            "latest_red_flags": [],
            "symptom_updates": [],
        }

    def set_patient_id(self, patient_id: str) -> None:
        self.patient_id = (patient_id or "anonymous").strip()
        self.case_snapshot["patient_id"] = self.patient_id

    def get_previous_records(self, limit: int = 5) -> List[Dict[str, Any]]:
        return find_patient_records(self.patient_id, limit=limit)

    def get_case_summary(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "turn_count": len(self.visit_turns),
            "latest_red_flags": self.case_snapshot.get("latest_red_flags") or [],
            "recent_symptom_updates": (self.case_snapshot.get("symptom_updates") or [])[-5:],
            "latest_models": {
                "nurse": self.nurse.model,
                "doctor": self.doctor.model,
            },
        }

    def persist_visit_to_db(self) -> Dict[str, Any]:
        record = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "patient_id": self.patient_id,
            "case_summary": self.get_case_summary(),
            "turns": self.visit_turns,
        }
        append_visit_db(record)
        return record

    def process_patient_message(self, user_text: str) -> Dict[str, Any]:
        patient_text = normalize_patient_input(user_text)
        if not patient_text:
            ui = build_ui_actions([], doctor_note=None, ok=False, error="low_info_input")
            return {
                "ok": False,
                "error": "Input too short/low information. Ask user for one complete sentence.",
                "red_flags": [],
                "ui": ui,
            }

        red_flags = detect_red_flags(patient_text)
        doctor_note = None
        self.case_snapshot["latest_red_flags"] = red_flags
        self.case_snapshot["symptom_updates"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "text": patient_text,
            "red_flags": red_flags,
        })

        # Safety-first interruption: do not continue roleplay on potential emergencies.
        if should_interrupt_for_emergency(red_flags):
            emergency_msg = EMERGENCY_OVERRIDE_MESSAGE
            turn = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "patient": patient_text,
                "red_flags": red_flags,
                "nurse_initial": emergency_msg,
                "doctor_note": "Emergency interruption activated.",
                "nurse_followup": emergency_msg,
                "models": {
                    "nurse": self.nurse.model,
                    "doctor": self.doctor.model,
                },
                "interrupted": True,
            }
            self.visit_turns.append(turn)
            return {
                "ok": True,
                "interrupted": True,
                "red_flags": red_flags,
                "nurse_initial": emergency_msg,
                "doctor_note": "Emergency interruption activated.",
                "nurse_followup": emergency_msg,
                "turn": turn,
                "ui": build_ui_actions(red_flags, doctor_note="Emergency interruption activated.", ok=True),
                "case_summary": self.get_case_summary(),
                "previous_records_count": len(self.get_previous_records(limit=20)),
                "visit_open": True,
                "triage_card": classify_triage(red_flags, patient_text),
                "recommended_next_input": "If safe to continue later, provide non-emergency symptom history.",
            }

        try:
            self.nurse.receive("user", patient_text)
            nurse_reply = self.nurse.respond(phase_label="triaging")

            try:
                doctor_note = consult_doctor(self.doctor, patient_message=patient_text, nurse_reply=nurse_reply)
                nurse_update_from_doctor(self.nurse, doctor_note)
            except Exception as e:
                doctor_note = f"Doctor consult unavailable: {e}"
                self.nurse.receive(
                    "user",
                    "Doctor is unavailable. Continue safely: ask red-flag questions and give general guidance.",
                )

            self.nurse.receive("user", "Now respond to the patient. Be clear and ask the next 1–3 questions if needed.")
            nurse_followup = self.nurse.respond(phase_label="follow-up")

            turn = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "patient": patient_text,
                "red_flags": red_flags,
                "nurse_initial": nurse_reply,
                "doctor_note": doctor_note,
                "nurse_followup": nurse_followup,
                "models": {
                    "nurse": self.nurse.model,
                    "doctor": self.doctor.model,
                },
            }
            self.visit_turns.append(turn)

            return {
                "ok": True,
                "red_flags": red_flags,
                "nurse_initial": nurse_reply,
                "doctor_note": doctor_note,
                "nurse_followup": nurse_followup,
                "turn": turn,
                "ui": build_ui_actions(red_flags, doctor_note=doctor_note, ok=True),
                "case_summary": self.get_case_summary(),
                "previous_records_count": len(self.get_previous_records(limit=20)),
                "visit_open": True,
                "triage_card": classify_triage(red_flags, patient_text),
                "recommended_next_input": "Add follow-up details: onset, severity, triggers, and prior episodes.",
            }
        except Exception as e:
            ui = build_ui_actions(red_flags, doctor_note=doctor_note, ok=False, error=str(e))
            return {
                "ok": False,
                "error": str(e),
                "red_flags": red_flags,
                "ui": ui,
            }


# =====================
# COMMANDS / DIAGNOSTICS
# =====================

def print_commands():
    safe_print("\n--- COMMANDS ---")
    safe_print("/m <agent> <text>        → add memory (agent: zesty/scarlett)")
    safe_print("/mood <agent> <mood>     → set mood")
    safe_print("/style <agent> <style>   → set style")
    safe_print("/reset <agent>           → reset agent")
    safe_print("/compact                 → shrink history (both agents)")
    safe_print("/models                  → list local Ollama models")
    safe_print("/setmodel <agent> <name> → set a model explicitly")
    safe_print("/settimeout <sec>        → set timeout seconds (both agents)")
    safe_print("/host                    → show configured Ollama hosts")
    safe_print("/sethost <url>           → set primary Ollama host")
    safe_print("/safety <text>           → run red-flag detection on text")
    safe_print("/stats                   → show API call stats")
    safe_print("/runtests                → run built-in 25-case safety suite")
    safe_print("/setpatient <id>         → set active patient id for record lookup")
    safe_print("/records [id]            → show previous visits from local DB")
    safe_print("/casesummary             → show in-memory case summary")
    safe_print("/savevisit               → persist current visit to local DB")
    safe_print("/ping                    → quick generation test")
    safe_print("/health                  → deeper health check")
    safe_print("/debug                   → internal status")
    safe_print("/notes                   → show recent visit notes")
    safe_print("/export [file]           → export visit notes to JSON")
    safe_print("/help                    → show commands")
    safe_print("quit")

def ping_model(model: str) -> str:
    msgs = [
        {"role": "system", "content": "Reply with exactly: OK"},
        {"role": "user", "content": "ping"},
    ]
    text = timed_ollama_chat(model=model, messages=msgs, options={"temperature": 0}, timeout_s=10, retries=0)
    return text

def health_check(models: List[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"ok": False, "results": []}
    for m in models:
        if not m:
            continue
        try:
            got = ping_model(m)
            report["results"].append({"model": m, "status": "ok", "reply": got})
        except Exception as e:
            report["results"].append({"model": m, "status": "fail", "error": str(e)})
    report["ok"] = any(r["status"] == "ok" for r in report["results"])
    return report

def show_debug(nurse: Agent, doctor: Agent):
    safe_print("\n[debug]")
    safe_print(f"  nurse model: {nurse.model}")
    safe_print(f"  doctor model: {doctor.model}")
    safe_print(f"  nurse messages: {len(nurse.messages)} (incl system)")
    safe_print(f"  doctor messages: {len(doctor.messages)} (incl system)")
    safe_print(f"  nurse memory items: {len(nurse.memory)}")
    safe_print(f"  doctor memory items: {len(doctor.memory)}")
    if nurse.messages:
        last = nurse.messages[-1]
        safe_print(f"  nurse last msg: {last.get('role')} / {repr((last.get('content') or '')[:140])}")
    if doctor.messages:
        last = doctor.messages[-1]
        safe_print(f"  doctor last msg: {last.get('role')} / {repr((last.get('content') or '')[:140])}")


# =====================
# INPUT HELPERS
# =====================

def normalize_patient_input(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None

    # Prevent “Patient: It” / fragments from sending junk into the model
    very_short = len(s) < 3
    low_info = s.lower() in {"it", "idk", "help", "yes", "no", "ok", "okay"}

    if very_short or low_info:
        safe_print("Can you finish the thought with one full sentence?")
        safe_print("Example: 'It hurts when I turn my head left and my right hand tingles.'")
        return None

    # cap patient input so it doesn't explode the prompt
    if len(s) > 2000:
        s = s[:2000].rstrip() + "…"
    return s


def detect_red_flags(text: str) -> List[str]:
    """
    Keyword + phrase safety net so urgent symptoms are called out immediately.
    Uses light normalization so wording variations still match.
    """
    t = (text or "").lower().strip()
    normalized = re.sub(r"[^a-z0-9\s]", " ", t)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    def has_phrase(*phrases: str) -> bool:
        return any(p in normalized for p in phrases)

    def has_all_words(*words: str) -> bool:
        return all(w in normalized for w in words)

    hits: List[str] = []

    no_chest_pain = has_phrase("no chest pain", "without chest pain", "denies chest pain")
    if (not no_chest_pain) and has_phrase("chest pain", "pressure in chest", "crushing pain", "chest pressure"):
        hits.append("possible chest pain emergency")

    if has_phrase(
        "cant breathe",
        "cannot breathe",
        "shortness of breath",
        "trouble breathing",
        "hard to breathe",
        "difficulty breathing",
        "breathless",
    ) or has_all_words("shortness", "breath"):
        hits.append("possible breathing emergency")

    if has_phrase(
        "face droop",
        "facial droop",
        "slurred speech",
        "one side weak",
        "weakness on one side",
        "one sided weakness",
        "numb on one side",
        "sudden weakness one side",
    ):
        hits.append("possible stroke symptoms")

    if has_phrase(
        "heavy bleeding",
        "wont stop bleeding",
        "won t stop bleeding",
        "coughing blood",
        "vomiting blood",
        "blood in vomit",
    ):
        hits.append("possible severe bleeding")

    if has_phrase(
        "suicidal",
        "want to die",
        "kill myself",
        "harm myself",
        "end my life",
        "self harm",
        "overdose tonight",
        "dont want to wake up",
        "do not want to wake up",
    ) and not has_phrase("not suicidal", "no suicidal thoughts", "deny suicidal thoughts", "denies suicidal thoughts"):
        hits.append("possible self-harm crisis")

    if has_phrase(
        "swollen tongue",
        "tongue swelling",
        "tongue feels swollen",
        "swollen throat",
        "throat swelling",
        "anaphylaxis",
        "throat closing",
        "my throat is closing",
        "airway swelling",
    ) or (has_all_words("tongue", "swollen") or has_all_words("throat", "swollen")):
        hits.append("possible anaphylaxis/allergic emergency")

    if has_phrase(
        "fainted",
        "passed out",
        "unconscious",
        "lost consciousness",
        "blackout",
        "blacked out",
        "syncope",
    ):
        hits.append("possible loss-of-consciousness concern")

    if has_phrase(
        "seizure",
        "convulsion",
        "tonic clonic",
        "shaking episode",
        "postictal",
    ):
        hits.append("possible seizure emergency")

    if has_phrase(
        "hematemesis",
        "hemoptysis",
        "coughing up blood",
        "black tarry stool",
        "melena",
        "bright red blood per rectum",
    ):
        hits.append("possible severe bleeding")

    # Preserve order but deduplicate in case of overlapping rules.
    if hits:
        hits = list(dict.fromkeys(hits))
    return hits


def build_ui_actions(red_flags: List[str], doctor_note: Optional[str], ok: bool, error: Optional[str] = None) -> Dict[str, Any]:
    """
    Web-UI friendly action metadata.
    """
    severity = "routine"
    if red_flags:
        severity = "urgent"
    if any("self-harm" in f or "seizure" in f for f in red_flags):
        severity = "emergency"
    if not ok:
        severity = "error"

    actions = [
        {"id": "submit_followup", "label": "Submit Follow-up", "style": "primary", "enabled": ok},
        {"id": "export_visit", "label": "Export Visit JSON", "style": "secondary", "enabled": True},
        {"id": "show_notes", "label": "Show Recent Notes", "style": "secondary", "enabled": True},
    ]

    if severity in {"urgent", "emergency"}:
        actions.insert(0, {"id": "seek_urgent_care", "label": "Urgent Care Guidance", "style": "danger", "enabled": True})
    if any("self-harm" in f for f in red_flags):
        actions.insert(0, {"id": "crisis_resources", "label": "Crisis Resources", "style": "danger", "enabled": True})
    if not ok:
        actions.insert(0, {"id": "retry", "label": "Retry", "style": "warning", "enabled": True})

    hints = []
    if doctor_note:
        hints.append("Display doctor note in a collapsible card.")
    if red_flags:
        hints.append("Pin emergency guidance at the top of the UI.")
    if error:
        hints.append("Offer host switch (/sethost equivalent) in settings UI.")

    return {
        "severity": severity,
        "actions": actions,
        "hints": hints,
    }


def classify_triage(red_flags: List[str], text: str) -> Dict[str, str]:
    """
    Final triage card for UI surfaces.
    """
    t = (text or "").lower()
    if red_flags:
        if any(k in " ".join(red_flags) for k in ["self-harm", "seizure", "breathing", "chest pain", "stroke", "bleeding"]):
            level = "Seek emergency care now"
        else:
            level = "Seek urgent care today"
    elif any(k in t for k in ["worse", "severe", "high fever", "persistent"]):
        level = "Seek urgent care today"
    elif t.strip():
        level = "Routine appointment"
    else:
        level = "General discussion only"
    return {
        "level": level,
        "label": "This is AI guidance, not a diagnosis.",
    }


def should_interrupt_for_emergency(red_flags: List[str]) -> bool:
    emergency_tokens = (
        "chest pain",
        "breathing",
        "stroke",
        "seizure",
        "loss-of-consciousness",
        "self-harm",
        "bleeding",
        "anaphylaxis",
    )
    joined = " ".join(red_flags)
    return any(tok in joined for tok in emergency_tokens)


def run_hypothetical_red_flag_suite() -> Dict[str, Any]:
    """
    25-case detector benchmark: 20 common + 5 rare/edge cases.
    """
    cases = [
        {"name": "Mild cold", "text": "Runny nose and mild cough for two days.", "expect": []},
        {"name": "Chest pain with dyspnea", "text": "I have chest pain and shortness of breath.", "expect": ["possible chest pain emergency", "possible breathing emergency"]},
        {"name": "Crushing chest pressure", "text": "Crushing pain and pressure in my chest.", "expect": ["possible chest pain emergency"]},
        {"name": "Trouble breathing", "text": "I am having trouble breathing right now.", "expect": ["possible breathing emergency"]},
        {"name": "Hard to breathe", "text": "It is hard to breathe and I am breathless.", "expect": ["possible breathing emergency"]},
        {"name": "Stroke phrase", "text": "My father has facial droop and slurred speech.", "expect": ["possible stroke symptoms"]},
        {"name": "One-sided weakness", "text": "Sudden weakness on one side of my body.", "expect": ["possible stroke symptoms"]},
        {"name": "Heavy bleeding", "text": "There is heavy bleeding that won't stop.", "expect": ["possible severe bleeding"]},
        {"name": "Vomiting blood", "text": "I am vomiting blood.", "expect": ["possible severe bleeding"]},
        {"name": "Coughing blood", "text": "I am coughing up blood.", "expect": ["possible severe bleeding"]},
        {"name": "Self-harm direct", "text": "I want to die and harm myself.", "expect": ["possible self-harm crisis"]},
        {"name": "Self-harm indirect", "text": "I do not want to wake up tomorrow.", "expect": ["possible self-harm crisis"]},
        {"name": "Anaphylaxis phrase", "text": "My tongue feels swollen and my throat is closing.", "expect": ["possible anaphylaxis/allergic emergency"]},
        {"name": "Throat swelling", "text": "I have throat swelling after peanuts.", "expect": ["possible anaphylaxis/allergic emergency"]},
        {"name": "Syncope", "text": "I blacked out and lost consciousness in the shower.", "expect": ["possible loss-of-consciousness concern"]},
        {"name": "Seizure wording", "text": "He had a seizure and postictal confusion.", "expect": ["possible seizure emergency"]},
        {"name": "Negated chest pain", "text": "No chest pain, only mild nausea.", "expect": []},
        {"name": "Negated suicidality", "text": "I am not suicidal but I feel stressed.", "expect": []},
        {"name": "Syncope medical term", "text": "Episode of syncope after standing.", "expect": ["possible loss-of-consciousness concern"]},
        {"name": "No red flags GI", "text": "Stomach ache after spicy food.", "expect": []},

        # Rare / hard-to-detect edge cases
        {"name": "Rare: hematemesis", "text": "Possible hematemesis after alcohol use.", "expect": ["possible severe bleeding"]},
        {"name": "Rare: hemoptysis", "text": "New hemoptysis with chest discomfort.", "expect": ["possible severe bleeding"]},
        {"name": "Rare: melena", "text": "Black tarry stool for two days.", "expect": ["possible severe bleeding"]},
        {"name": "Rare: tonic-clonic", "text": "Witnessed tonic clonic event at home.", "expect": ["possible seizure emergency"]},
        {"name": "Rare: BRBPR", "text": "Bright red blood per rectum this morning.", "expect": ["possible severe bleeding"]},
    ]

    rows = []
    for c in cases:
        got = detect_red_flags(c["text"])
        missing = [e for e in c["expect"] if e not in got]
        rows.append({
            "name": c["name"],
            "ok": len(missing) == 0,
            "expected": c["expect"],
            "got": got,
            "missing_expected": missing,
        })

    summary = {
        "cases": len(rows),
        "passes": sum(1 for r in rows if r["ok"]),
        "fails": sum(1 for r in rows if not r["ok"]),
    }
    return {"summary": summary, "rows": rows}


def save_visit_log(path: str, turns: List[Dict[str, Any]]) -> str:
    path = (path or "visit_log.json").strip()
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "turns": turns,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_visit_db() -> Dict[str, Any]:
    if not VISIT_DB_PATH.exists():
        return {"visits": []}
    try:
        with open(VISIT_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("visits"), list):
            return data
    except Exception:
        pass
    return {"visits": []}


def append_visit_db(record: Dict[str, Any]) -> None:
    data = load_visit_db()
    data["visits"].append(record)
    with open(VISIT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_patient_records(patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    pid = (patient_id or "").strip()
    if not pid:
        return []
    data = load_visit_db()
    visits = data.get("visits") or []
    matches = [v for v in visits if (v.get("patient_id") or "") == pid]
    return matches[-max(1, limit):]


WEB_SESSIONS: Dict[str, ClinicSession] = {}


def get_or_create_session(patient_id: str) -> ClinicSession:
    pid = (patient_id or "anonymous").strip() or "anonymous"
    if pid not in WEB_SESSIONS:
        WEB_SESSIONS[pid] = ClinicSession(use_spinner=False, patient_id=pid)
    return WEB_SESSIONS[pid]


def nurse_page_html() -> str:
    # Safety copy intentionally exact per product requirement.
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qrix Nurse Roleplay Intake</title>
  <style>
    :root { --bg:#0f172a; --panel:#111827; --ink:#e5e7eb; --muted:#94a3b8; --accent:#22c55e; --warn:#f97316; --danger:#ef4444; --line:#1f2937; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,system-ui,Arial,sans-serif; background:linear-gradient(180deg,#0b1220,#111827); color:var(--ink); }
    .wrap { max-width:1100px; margin:0 auto; padding:16px; }
    .banner { background:#7f1d1d; border:1px solid #ef4444; border-radius:12px; padding:14px; line-height:1.45; }
    .cards { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:12px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; min-height:220px; }
    .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }
    .value { margin-top:6px; white-space:pre-wrap; }
    .panel-empty { color:var(--muted); font-style:italic; }
    .toolbar, .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
    button { background:#1f2937; color:var(--ink); border:1px solid #334155; border-radius:10px; padding:8px 12px; cursor:pointer; }
    button.primary { background:#2563eb; border-color:#2563eb; }
    button.danger { background:#991b1b; border-color:#dc2626; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    textarea,input { width:100%; background:#0b1220; color:var(--ink); border:1px solid #334155; border-radius:10px; padding:10px; }
    .row { display:grid; grid-template-columns:1fr 2fr; gap:10px; margin-top:12px; }
    .note { color:var(--muted); font-size:13px; }
    .state { margin-top:8px; padding:8px 10px; border-radius:10px; background:#0b1220; border:1px solid #334155; }
    .triage { margin-top:12px; border:1px solid #374151; border-radius:12px; padding:12px; background:#111827; }
    .triage strong { color:#fbbf24; }
    @media (max-width: 900px){ .cards { grid-template-columns:1fr; } .row { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<div class="wrap">
  <h1>/nurse — Roleplay intake</h1>
  <div class="banner" role="alert" aria-live="assertive">
    <div>Educational roleplay only — not medical advice, diagnosis, treatment, or emergency care.</div>
    <div>If you think you may be having a medical emergency, call 911 or go to the nearest emergency room now.</div>
    <div>Do not use this tool for chest pain, trouble breathing, stroke symptoms, severe bleeding, seizures, suicidal thoughts, allergic reactions, overdose, or any urgent condition.</div>
  </div>

  <div class="row">
    <div class="card">
      <div class="label">What this tool is / is not</div>
      <div class="value"><strong>DOES:</strong><br/>- organizes symptoms into structured notes<br/>- simulates intake + doctor-style summary<br/>- asks follow-up questions<br/><br/><strong>DOES NOT:</strong><br/>- diagnose<br/>- prescribe<br/>- replace real doctors<br/>- handle emergencies<br/><br/><strong>This tool may be wrong, incomplete, or unsafe.</strong></div>
    </div>
    <div class="card">
      <div class="label">Privacy + data handling</div>
      <div class="value">Stored data includes user input, timestamps, and AI outputs.<br/>Storage location: browser localStorage (patient id + UI state) and server JSON file (visit records).<br/>Delete logs: use “Delete logs (patient)” below.<br/><br/><strong>Do not enter sensitive real-world personal information unless you understand how it is stored.</strong></div>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <div class="label">Visit controls</div>
    <label for="pid">Patient ID</label>
    <input id="pid" aria-label="Patient ID" placeholder="patient-123" />
    <label for="msg" style="margin-top:8px; display:block;">Complaint / update</label>
    <textarea id="msg" rows="4" aria-label="Complaint input" placeholder="Describe symptoms..."></textarea>
    <div class="note">Include when it started, where it hurts, severity, and what makes it better or worse.</div>
    <div class="note">Example prompts: “I’ve had a sore throat and fever for two days.” · “My stomach hurts after eating.” · “I twisted my ankle yesterday and it’s swollen.” · “I’ve been coughing all week.”</div>
    <div class="state" id="state">State: Idle</div>
    <div class="toolbar">
      <button class="primary" id="sendBtn" aria-label="Send to clinic">Send to clinic</button>
      <button id="saveBtn" aria-label="Save visit log">Save visit log</button>
      <button id="exportBtn" aria-label="Export visit summary">Export visit summary</button>
      <button id="resetBtn" aria-label="Reset current visit">Reset visit</button>
      <button class="danger" id="deleteBtn" aria-label="Delete logs for patient">Delete logs (patient)</button>
    </div>
  </div>

  <div class="cards">
    <div class="card"><div class="label">Roleplay intake</div><div id="nursePanel" class="value panel-empty">No intake yet. Submit a complaint to begin.</div><button data-copy="nursePanel">Copy</button></div>
    <div class="card"><div class="label">Doctor-style note</div><div id="docPanel" class="value panel-empty">No doctor-style note yet.</div><button data-copy="docPanel">Copy</button></div>
    <div class="card"><div class="label">AI follow-up</div><div id="followPanel" class="value panel-empty">No AI follow-up yet.</div><button data-copy="followPanel">Copy</button></div>
  </div>

  <div class="triage" id="triageCard"><strong>This is AI guidance, not a diagnosis.</strong><div id="triageText">General discussion only</div></div>
  <div class="card" style="margin-top:12px;">
    <div class="label">Visit history</div>
    <div id="history" class="value panel-empty">No saved records yet.</div>
  </div>
</div>

<script>
const stateEl = document.getElementById('state');
const pidEl = document.getElementById('pid');
const msgEl = document.getElementById('msg');
const nursePanel = document.getElementById('nursePanel');
const docPanel = document.getElementById('docPanel');
const followPanel = document.getElementById('followPanel');
const triageText = document.getElementById('triageText');
const historyEl = document.getElementById('history');
const states = ['Idle','Collecting intake','Nurse reviewing','Doctor reviewing','Nurse follow-up ready','Visit complete'];
pidEl.value = localStorage.getItem('nurse_patient_id') || 'anonymous';

function setState(i){ stateEl.textContent = 'State: ' + states[Math.max(0, Math.min(states.length-1, i))]; }
function setPanel(el, text){ el.textContent = text || 'No output yet.'; el.classList.remove('panel-empty'); }

async function fetchHistory(){
  const pid = pidEl.value.trim() || 'anonymous';
  const r = await fetch('/api/nurse/history?patient_id=' + encodeURIComponent(pid));
  const j = await r.json();
  if(!j.records || !j.records.length){ historyEl.textContent = 'No saved records yet.'; historyEl.classList.add('panel-empty'); return; }
  historyEl.classList.remove('panel-empty');
  historyEl.textContent = j.records.map((x,i)=>`${i+1}. ${x.saved_at} | turns=${(x.case_summary||{}).turn_count} | flags=${((x.case_summary||{}).latest_red_flags||[]).join(', ')}`).join('\\n');
}

document.getElementById('sendBtn').onclick = async () => {
  const pid = pidEl.value.trim() || 'anonymous';
  const message = msgEl.value.trim();
  if(!message){ alert('Please enter complaint/update text.'); return; }
  localStorage.setItem('nurse_patient_id', pid);
  setState(1); setState(2);
  const r = await fetch('/api/nurse/message', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({patient_id: pid, message})});
  const j = await r.json();
  if(j.interrupted){ setState(5); }
  else { setState(3); setState(4); setState(5); }
  setPanel(nursePanel, j.nurse_initial);
  setPanel(docPanel, j.doctor_note);
  setPanel(followPanel, j.nurse_followup);
  triageText.textContent = (j.triage_card||{}).level || 'General discussion only';
  msgEl.value = '';
  await fetchHistory();
};

document.getElementById('saveBtn').onclick = async () => {
  const pid = pidEl.value.trim() || 'anonymous';
  await fetch('/api/nurse/save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({patient_id: pid})});
  await fetchHistory();
  alert('Visit saved.');
};

document.getElementById('resetBtn').onclick = async () => {
  const pid = pidEl.value.trim() || 'anonymous';
  await fetch('/api/nurse/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({patient_id: pid})});
  setState(0);
  nursePanel.textContent = 'No intake yet. Submit a complaint to begin.'; nursePanel.classList.add('panel-empty');
  docPanel.textContent = 'No doctor-style note yet.'; docPanel.classList.add('panel-empty');
  followPanel.textContent = 'No AI follow-up yet.'; followPanel.classList.add('panel-empty');
  triageText.textContent = 'General discussion only';
  alert('Visit reset. You can continue by sending a new intake message.');
};

document.getElementById('deleteBtn').onclick = async () => {
  const pid = pidEl.value.trim() || 'anonymous';
  await fetch('/api/nurse/history?patient_id=' + encodeURIComponent(pid), {method:'DELETE'});
  await fetchHistory();
  alert('Deleted saved logs for patient: ' + pid);
};

document.getElementById('exportBtn').onclick = async () => {
  const pid = pidEl.value.trim() || 'anonymous';
  const r = await fetch('/api/nurse/export?patient_id=' + encodeURIComponent(pid));
  const j = await r.json();
  navigator.clipboard.writeText(JSON.stringify(j, null, 2));
  alert('Visit summary copied to clipboard.');
};

document.querySelectorAll('button[data-copy]').forEach(btn => btn.onclick = async () => {
  const id = btn.getAttribute('data-copy');
  const text = document.getElementById(id).textContent || '';
  await navigator.clipboard.writeText(text);
  alert('Copied.');
});

fetchHistory();
</script>
</body></html>"""


def create_web_app() -> Any:
    if Flask is None:
        raise RuntimeError("Flask is required for web mode. Install flask first.")
    app = Flask(__name__)

    @app.get("/nurse")
    def nurse_page():
        return Response(nurse_page_html(), status=HTTPStatus.OK, mimetype="text/html")

    @app.post("/api/nurse/message")
    def api_message():
        body = request.get_json(force=True, silent=True) or {}
        pid = (body.get("patient_id") or "anonymous").strip()
        msg = body.get("message") or ""
        s = get_or_create_session(pid)
        out = s.process_patient_message(msg)
        return jsonify(out), HTTPStatus.OK

    @app.post("/api/nurse/save")
    def api_save():
        body = request.get_json(force=True, silent=True) or {}
        pid = (body.get("patient_id") or "anonymous").strip()
        s = get_or_create_session(pid)
        rec = s.persist_visit_to_db()
        return jsonify({"ok": True, "record": rec}), HTTPStatus.OK

    @app.get("/api/nurse/history")
    def api_history():
        pid = (request.args.get("patient_id") or "anonymous").strip()
        return jsonify({"ok": True, "records": find_patient_records(pid, limit=20)}), HTTPStatus.OK

    @app.delete("/api/nurse/history")
    def api_history_delete():
        pid = (request.args.get("patient_id") or "anonymous").strip()
        data = load_visit_db()
        visits = data.get("visits") or []
        data["visits"] = [v for v in visits if (v.get("patient_id") or "") != pid]
        with open(VISIT_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True, "deleted_patient_id": pid}), HTTPStatus.OK

    @app.post("/api/nurse/reset")
    def api_reset():
        body = request.get_json(force=True, silent=True) or {}
        pid = (body.get("patient_id") or "anonymous").strip()
        WEB_SESSIONS[pid] = ClinicSession(use_spinner=False, patient_id=pid)
        return jsonify({"ok": True}), HTTPStatus.OK

    @app.get("/api/nurse/export")
    def api_export():
        pid = (request.args.get("patient_id") or "anonymous").strip()
        s = get_or_create_session(pid)
        return jsonify({
            "ok": True,
            "patient_id": pid,
            "case_summary": s.get_case_summary(),
            "turns": s.visit_turns,
            "triage_card": classify_triage(s.case_snapshot.get("latest_red_flags") or [], " ".join(
                [x.get("text") or "" for x in (s.case_snapshot.get("symptom_updates") or [])]
            )),
        }), HTTPStatus.OK

    return app


# =====================
# MAIN
# =====================

def main():
    session = ClinicSession(use_spinner=True)
    nurse = session.nurse
    doctor = session.doctor

    safe_print("\n=== Clinic Roleplay Engine (Patient ↔ Nurse Zesty ↔ Doctor Scarlett) ===\n")
    safe_print("You are the patient. Describe what brings you in today.\n")
    safe_print("Tip: If Ollama ever stalls, try /health or /models.\n")
    safe_print(f"Active patient id: {session.patient_id}")
    safe_print(f"Ollama host (primary): {OLLAMA_HOST}")
    safe_print(f"Ollama host candidates: {get_candidate_hosts()}\n")

    visit_turns = session.visit_turns

    while True:
        print_commands()
        try:
            raw = input("\nPatient: ")
        except KeyboardInterrupt:
            safe_print("\n(^C) Exiting.")
            break

        user_input = (raw or "").strip()

        if user_input.lower() == "quit":
            break

        if user_input == "/help":
            print_commands()
            continue

        if user_input.startswith("/setpatient "):
            parts = user_input.split(" ", 1)
            if len(parts) != 2 or not parts[1].strip():
                safe_print("Usage: /setpatient <id>")
                continue
            session.set_patient_id(parts[1].strip())
            safe_print(f"Active patient id set to: {session.patient_id}")
            continue

        if user_input.startswith("/records"):
            parts = user_input.split(" ", 1)
            lookup_id = parts[1].strip() if len(parts) == 2 and parts[1].strip() else session.patient_id
            records = find_patient_records(lookup_id, limit=5)
            safe_print(f"Previous records for patient '{lookup_id}': {len(records)}")
            for idx, rec in enumerate(records, 1):
                summ = rec.get("case_summary") or {}
                safe_print(
                    f"{idx}. saved_at={rec.get('saved_at')} turns={summ.get('turn_count')} flags={summ.get('latest_red_flags')}"
                )
            continue

        if user_input == "/casesummary":
            safe_print(json.dumps(session.get_case_summary(), indent=2))
            continue

        if user_input == "/savevisit":
            rec = session.persist_visit_to_db()
            safe_print(
                f"Saved visit for patient '{rec.get('patient_id')}' with {len(rec.get('turns') or [])} turns to {VISIT_DB_PATH}"
            )
            continue

        if user_input == "/host":
            safe_print(f"Primary host: {OLLAMA_HOST}")
            safe_print(f"Active host: {ACTIVE_OLLAMA_HOST}")
            safe_print(f"Candidates: {get_candidate_hosts()}")
            continue

        if user_input.startswith("/sethost "):
            parts = user_input.split(" ", 1)
            if len(parts) != 2 or not parts[1].strip():
                safe_print("Usage: /sethost <url>")
                continue
            try:
                set_ollama_host(parts[1].strip())
                safe_print(f"Primary host updated to: {OLLAMA_HOST}")
            except Exception as e:
                safe_print(f"Invalid host: {e}")
            continue

        if user_input.startswith("/safety "):
            parts = user_input.split(" ", 1)
            check_text = parts[1].strip() if len(parts) == 2 else ""
            if not check_text:
                safe_print("Usage: /safety <text>")
                continue
            flags = detect_red_flags(check_text)
            if flags:
                safe_print("Red flags found:")
                for f in flags:
                    safe_print(f" - {f}")
            else:
                safe_print("No red flags detected.")
            continue

        if user_input == "/stats":
            safe_print(json.dumps(get_api_stats_snapshot(), indent=2))
            continue

        if user_input == "/runtests":
            rep = run_hypothetical_red_flag_suite()
            safe_print("\nBuilt-in 25-case safety suite:")
            safe_print(json.dumps(rep["summary"], indent=2))
            failed = [r for r in rep.get("rows", []) if not r.get("ok")]
            if failed:
                safe_print("Failed cases:")
                for row in failed[:10]:
                    safe_print(f" - {row.get('name')}: missing={row.get('missing_expected')}, got={row.get('got')}")
            else:
                safe_print("All cases passed.")
            continue

        # ---- diagnostics commands ----
        if user_input == "/debug":
            show_debug(nurse, doctor)
            continue

        if user_input == "/compact":
            nurse.compact_history(keep_last_n=8)
            doctor.compact_history(keep_last_n=6)
            safe_print("History compacted (both agents).")
            continue

        if user_input == "/models":
            try:
                s = Spinner("Listing models")
                s.start()
                names = ollama_list_models(timeout_s=10)
                s.stop()
                safe_print("\nAvailable models:")
                for n in names:
                    safe_print(f" - {n}")
            except Exception as e:
                try:
                    s.stop()
                except Exception:
                    pass
                safe_print(f"\n[models] FAILED: {e}")
            continue

        if user_input.startswith("/setmodel "):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                safe_print("Usage: /setmodel <agent> <name>")
                continue
            _, who, name = parts
            who_l = who.lower().strip()
            if who_l == "zesty":
                nurse.model = name.strip()
                safe_print(f"Set nurse model to: {nurse.model}")
            elif who_l == "scarlett":
                doctor.model = name.strip()
                safe_print(f"Set doctor model to: {doctor.model}")
            else:
                safe_print("Agent must be 'zesty' or 'scarlett'.")
            continue

        if user_input.startswith("/settimeout "):
            parts = user_input.split(" ", 1)
            if len(parts) != 2:
                safe_print("Usage: /settimeout <sec>")
                continue
            try:
                sec = int(parts[1].strip())
                sec = max(5, min(sec, 300))
                nurse.timeout_s = sec
                doctor.timeout_s = sec
                safe_print(f"Timeout set to {sec}s for both agents.")
            except ValueError:
                safe_print("Timeout must be an integer seconds value.")
            continue

        if user_input == "/ping":
            # Try nurse primary then fallbacks
            candidates = [nurse.model] + nurse.failover_models
            ok = False
            for m in candidates:
                if not m:
                    continue
                try:
                    s = Spinner(f"Pinging {m}")
                    s.start()
                    reply = ping_model(m)
                    s.stop()
                    safe_print(f"\n[ping] model={m} → {reply!r}")
                    ok = True
                    break
                except Exception as e:
                    try:
                        s.stop()
                    except Exception:
                        pass
                    safe_print(f"\n[ping] model={m} FAILED: {e}")
            if not ok:
                safe_print("\n[ping] All candidates failed. Ollama may be wedged—restart the Ollama service.")
            continue

        if user_input == "/health":
            candidates = [nurse.model, doctor.model] + nurse.failover_models
            # remove duplicates while keeping order
            seen = set()
            uniq = []
            for c in candidates:
                if c and c not in seen:
                    uniq.append(c)
                    seen.add(c)

            try:
                s = Spinner("Running health check")
                s.start()
                rep = health_check(uniq)
                s.stop()
                safe_print("\nHealth check:")
                safe_print(json.dumps(rep, indent=2))
                if not rep.get("ok"):
                    safe_print("\nNo model responded. Ollama likely needs a restart or the host is out of resources.")
            except Exception as e:
                try:
                    s.stop()
                except Exception:
                    pass
                safe_print(f"\n[health] FAILED: {e}")
            continue

        if user_input == "/notes":
            if not visit_turns:
                safe_print("No visit notes yet.")
                continue
            safe_print("\nRecent visit notes:")
            for idx, turn in enumerate(visit_turns[-5:], 1):
                safe_print(f"{idx}. {turn.get('timestamp')} | patient={repr((turn.get('patient') or '')[:90])}")
                if turn.get("red_flags"):
                    safe_print(f"   red_flags={', '.join(turn['red_flags'])}")
            continue

        if user_input.startswith("/export"):
            parts = user_input.split(" ", 1)
            out_path = parts[1].strip() if len(parts) == 2 and parts[1].strip() else "visit_log.json"
            try:
                written = save_visit_log(out_path, visit_turns)
                safe_print(f"Saved visit log to: {written}")
            except Exception as e:
                safe_print(f"Failed to export visit log: {e}")
            continue

        # ---- memory/mood/style/reset commands ----
        if user_input.startswith("/m "):
            parts = user_input.split(" ", 2)
            if len(parts) == 3:
                _, who, text = parts
                target = nurse if who.lower() == "zesty" else doctor
                target.add_memory(text)
                safe_print("Memory added.")
            else:
                safe_print("Usage: /m <agent> <text>")
            continue

        if user_input.startswith("/mood "):
            parts = user_input.split(" ", 2)
            if len(parts) == 3:
                _, who, mood = parts
                target = nurse if who.lower() == "zesty" else doctor
                target.set_mood(mood)
                safe_print("Mood updated.")
            else:
                safe_print("Usage: /mood <agent> <mood>")
            continue

        if user_input.startswith("/style "):
            parts = user_input.split(" ", 2)
            if len(parts) == 3:
                _, who, style = parts
                target = nurse if who.lower() == "zesty" else doctor
                target.set_style(style)
                safe_print("Style updated.")
            else:
                safe_print("Usage: /style <agent> <style>")
            continue

        if user_input.startswith("/reset "):
            parts = user_input.split(" ", 1)
            if len(parts) == 2:
                _, who = parts
                target = nurse if who.lower() == "zesty" else doctor
                target.reset()
                safe_print("Agent reset.")
            else:
                safe_print("Usage: /reset <agent>")
            continue

        # ---- normal patient flow ----
        patient_text = normalize_patient_input(user_input)
        if not patient_text:
            continue

        red_flags = detect_red_flags(patient_text)
        if red_flags:
            safe_print("\n[safety-check] Potential red flags detected:")
            for rf in red_flags:
                safe_print(f" - {rf}")
            safe_print("If symptoms are severe or worsening, seek urgent/emergency care now.")

        result = session.process_patient_message(patient_text)
        if not result.get("ok"):
            err = result.get("error") or "unknown error"
            safe_print(f"\n[error] {err}")
            safe_print("Try /debug, /health, or /compact to recover.")
            continue

        safe_print(f"\n[{nurse.name}] {result.get('nurse_initial')}")
        doctor_note = result.get("doctor_note")
        if doctor_note:
            if str(doctor_note).startswith("Doctor consult unavailable:"):
                safe_print(f"\n[warn] {doctor_note}")
            else:
                safe_print(f"\n[{doctor.name}] {doctor_note}")
        safe_print(f"\n[{nurse.name}] {result.get('nurse_followup')}")

if __name__ == "__main__":
    if "--web" in sys.argv:
        app = create_web_app()
        port = 8080
        for i, a in enumerate(sys.argv):
            if a == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                except ValueError:
                    pass
        safe_print(f"Starting web server on http://0.0.0.0:{port}/nurse")
        app.run(host="0.0.0.0", port=port, debug=False)
    else:
        main()
