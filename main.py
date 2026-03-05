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
- Ensure your model names are correct (PopPooB-D / PopPooB-Pin-Yin or change below)
"""

import os
import sys
import time
import json
import threading
from typing import List, Dict, Optional, Any, Tuple

import ollama


# =====================
# CONFIG
# =====================

DEFAULT_NURSE_MODEL = os.getenv("NURSE_MODEL", "PopPooB-D")
DEFAULT_DOCTOR_MODEL = os.getenv("DOCTOR_MODEL", "PopPooB-Pin-Yin")

# Fallback models to try if the primary model stalls.
# IMPORTANT: Put models you actually have installed on your Ollama server.
FALLBACK_MODELS = [
    os.getenv("FALLBACK_MODEL_1", "llama3:8b"),
    os.getenv("FALLBACK_MODEL_2", "phi3:mini"),
]

# If you are using a remote Ollama host, set:
#   export OLLAMA_HOST=http://YOUR_HOST:11434
# The ollama python client typically respects this env var.


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

def _ollama_chat_worker(model: str, messages: List[Dict[str, str]], options: Dict[str, Any],
                        out: Dict[str, Any], err: Dict[str, Exception]):
    try:
        resp = ollama.chat(model=model, messages=messages, options=options)
        out["text"] = resp["message"]["content"]
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
    Lists locally available models via ollama.list().
    If the server is wedged, this can also hang; so we timebox it.
    """
    out: Dict[str, Any] = {}
    err: Dict[str, Exception] = {}

    def worker():
        try:
            out["resp"] = ollama.list()
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
    safe_print("/ping                    → quick generation test")
    safe_print("/health                  → deeper health check")
    safe_print("/debug                   → internal status")
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
            if who.lower() == "zesty":
                nurse.model = name.strip()
                safe_print(f"Set nurse model to: {nurse.model}")
            else:
                doctor.model = name.strip()
                safe_print(f"Set doctor model to: {doctor.model}")
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
