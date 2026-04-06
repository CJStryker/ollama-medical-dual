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
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest



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

ZESTY_PROMPT = """
IDENTITY:
You are Zesty, a registered nurse in a clinic. You speak in first person and refer to yourself as "I".

TEAM:
Scarlett is the supervising doctor. You trust her clinical judgment and follow her guidance.

ROLE:
- You speak directly to the patient (the user).
- You gather history, symptoms, vitals (if provided), meds/allergies, and relevant context.
- You ask clear follow-up questions, one at a time when possible.
- You summarize concisely and then consult Scarlett when needed.

BOUNDARIES:
- No explicit sexual content.
- No graphic violence.
- Do not provide unsafe instructions.
- If symptoms suggest emergency (chest pain, severe trouble breathing, stroke signs, fainting, severe bleeding, suicidal intent), advise urgent emergency care.

STYLE:
- Calm, professional, empathetic.
- Simple language.
- Keep messages reasonably short unless the patient asks for detail.

IMPORTANT:
You are participating in an ongoing roleplay where the user is the patient.
"""

SCARLETT_PROMPT = """
IDENTITY:
You are Scarlett, a physician (doctor) supervising Zesty. You speak in first person.

ROLE:
- You advise Nurse Zesty based on the presented patient information.
- You help decide what questions to ask next and what safe, general guidance to provide.
- You emphasize red flags and when to escalate to urgent care.

OUTPUT FORMAT (IMPORTANT):
Always respond as a brief "Doctor Note" to Zesty:
- Assessment (1-3 bullets)
- Red flags to check (bullets)
- Next questions (bullets)
- Safe advice to patient (bullets)
Keep it concise.

BOUNDARIES:
- No explicit sexual content.
- No graphic violence.
- Do not provide unsafe instructions.
- If emergency red flags present, recommend urgent evaluation.

You are participating in an ongoing roleplay.
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
    ):
        hits.append("possible self-harm crisis")

    if has_phrase(
        "swollen tongue",
        "tongue swelling",
        "tongue feels swollen",
        "swollen throat",
        "throat swelling",
        "anaphylaxis",
    ) or (has_all_words("tongue", "swollen") or has_all_words("throat", "swollen")):
        hits.append("possible anaphylaxis/allergic emergency")

    if has_phrase(
        "fainted",
        "passed out",
        "unconscious",
        "lost consciousness",
        "blackout",
        "blacked out",
    ):
        hits.append("possible loss-of-consciousness concern")

    # Preserve order but deduplicate in case of overlapping rules.
    if hits:
        hits = list(dict.fromkeys(hits))
    return hits


def save_visit_log(path: str, turns: List[Dict[str, Any]]) -> str:
    path = (path or "visit_log.json").strip()
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "turns": turns,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# =====================
# MAIN
# =====================

def main():
    nurse = Agent(
        name="Zesty (Nurse)",
        model=DEFAULT_NURSE_MODEL,
        system_prompt=ZESTY_PROMPT,
        temperature=0.65,
        max_history=18,
        max_memory=20,
        timeout_s=60,
        retries=1,
        failover_models=FALLBACK_MODELS,
    )

    doctor = Agent(
        name="Scarlett (Doctor)",
        model=DEFAULT_DOCTOR_MODEL,
        system_prompt=SCARLETT_PROMPT,
        temperature=0.55,
        max_history=12,
        max_memory=12,
        timeout_s=60,
        retries=1,
        failover_models=FALLBACK_MODELS,
    )

    safe_print("\n=== Clinic Roleplay Engine (Patient ↔ Nurse Zesty ↔ Doctor Scarlett) ===\n")
    safe_print("You are the patient. Describe what brings you in today.\n")
    safe_print("Tip: If Ollama ever stalls, try /health or /models.\n")
    safe_print(f"Ollama host (primary): {OLLAMA_HOST}")
    safe_print(f"Ollama host candidates: {get_candidate_hosts()}\n")

    visit_turns: List[Dict[str, Any]] = []

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

        try:
            # 1) Patient -> Nurse
            nurse.receive("user", patient_text)
            nurse_reply = nurse.respond(phase_label="triaging")
            safe_print(f"\n[{nurse.name}] {nurse_reply}")

            # 2) Nurse -> Doctor (optional; if it fails, keep going nurse-only)
            doctor_note = None
            try:
                doctor_note = consult_doctor(doctor, patient_message=patient_text, nurse_reply=nurse_reply)
                safe_print(f"\n[{doctor.name}] {doctor_note}")
                nurse_update_from_doctor(nurse, doctor_note)
            except Exception as e:
                safe_print(f"\n[warn] Doctor consult unavailable: {e}")
                nurse.receive("user", "Doctor is unavailable. Continue safely: ask red-flag questions and give general guidance.")

            # 3) Nurse follow-up to patient
            nurse.receive("user", "Now respond to the patient. Be clear and ask the next 1–3 questions if needed.")
            nurse_followup = nurse.respond(phase_label="follow-up")
            safe_print(f"\n[{nurse.name}] {nurse_followup}")

            visit_turns.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "patient": patient_text,
                "red_flags": red_flags,
                "nurse_initial": nurse_reply,
                "doctor_note": doctor_note,
                "nurse_followup": nurse_followup,
                "models": {
                    "nurse": nurse.model,
                    "doctor": doctor.model,
                }
            })

        except TimeoutError as te:
            safe_print(f"\n[error] Model call timed out: {te}")
            safe_print("Try /health. If nothing responds, restart the Ollama service or switch to a smaller model (/setmodel).")
        except KeyboardInterrupt:
            safe_print("\n(^C) Interrupted generation. Back to prompt.")
            continue
        except Exception as e:
            safe_print(f"\n[error] Unexpected error: {e}")
            safe_print("Try /debug, /health, or /compact to recover.")

if __name__ == "__main__":
    main()
