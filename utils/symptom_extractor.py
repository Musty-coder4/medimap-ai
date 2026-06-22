"""
MediMap AI — Natural Language Symptom Extractor  v2.0
======================================================
Extracts known symptom names from free-text patient descriptions using a
3-layer hybrid pipeline:

  Layer 1 — Gemini Flash AI   (cloud, context-aware, handles idioms/negation)
             Activates automatically when GEMINI_API_KEY is set in .env
             Get a free key at: https://aistudio.google.com/apikey

  Layer 2 — Enhanced Synonym Map  (offline, 300+ entries including Nigerian
             English, Pidgin, body-part references, intensity modifiers,
             and negation stripping)

  Layer 3 — Fuzzy / Direct Match  (unchanged from v1.0 as a final safety net)

Author : MediMap AI Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from difflib import get_close_matches
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# GEMINI AI CONFIGURATION  (Layer 1 — optional)
# =============================================================================

# Paste your free Gemini key here, or set the GEMINI_API_KEY env variable.
# Get a free key in 2 minutes at: https://aistudio.google.com/apikey
# Leave blank ("") to skip AI extraction and use the offline engine only.
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

_GEMINI_MODEL = "gemini-2.5-flash-lite"  # working free-tier model on this key


# =============================================================================
# EXPANDED SYNONYM MAP  (Layer 2 — 300+ entries)
# =============================================================================

_SYNONYMS: dict[str, str] = {

    # ── Fever / Temperature ───────────────────────────────────────────────────
    "fever":                        "high_fever",
    "high fever":                   "high_fever",
    "high temperature":             "high_fever",
    "running temperature":          "high_fever",
    "running a temperature":        "high_fever",
    "feeling hot":                  "high_fever",
    "body dey hot":                 "high_fever",
    "body is hot":                  "high_fever",
    "temperature is high":          "high_fever",
    "burning up":                   "high_fever",
    "hot to the touch":             "high_fever",
    "mild fever":                   "mild_fever",
    "low grade fever":              "mild_fever",
    "slight fever":                 "mild_fever",
    "little fever":                 "mild_fever",
    "warm":                         "mild_fever",
    "feel warm":                    "mild_fever",

    # ── Respiratory ───────────────────────────────────────────────────────────
    "cough":                        "cough",
    "coughing":                     "cough",
    "dry cough":                    "cough",
    "wet cough":                    "cough",
    "persistent cough":             "cough",
    "been coughing":                "cough",
    "cannot breathe":               "breathlessness",
    "can't breathe":                "breathlessness",
    "difficulty breathing":         "breathlessness",
    "hard to breathe":              "breathlessness",
    "shortness of breath":          "breathlessness",
    "short of breath":              "breathlessness",
    "out of breath":                "breathlessness",
    "breathless":                   "breathlessness",
    "breathing problems":           "breathlessness",
    "my breath":                    "breathlessness",
    "cannot catch my breath":       "breathlessness",
    "laboured breathing":           "breathlessness",
    "labored breathing":            "breathlessness",
    "runny nose":                   "runny_nose",
    "nose is running":              "runny_nose",
    "nose dey run":                 "runny_nose",
    "stuffy nose":                  "congestion",
    "blocked nose":                 "congestion",
    "nasal congestion":             "congestion",
    "nose is blocked":              "congestion",
    "nose blocked":                 "congestion",
    "sneezing":                     "continuous_sneezing",
    "sneeze":                       "continuous_sneezing",
    "sneezing a lot":               "continuous_sneezing",
    "phlegm":                       "phlegm",
    "mucus":                        "phlegm",
    "sputum":                       "phlegm",
    "coughing up":                  "phlegm",
    "sore throat":                  "throat_irritation",
    "throat pain":                  "throat_irritation",
    "pain when swallowing":         "throat_irritation",
    "pain swallowing":              "throat_irritation",
    "throat is sore":               "throat_irritation",
    "throat hurts":                 "throat_irritation",
    "scratchy throat":              "throat_irritation",
    "sinus":                        "sinus_pressure",
    "sinus pressure":               "sinus_pressure",
    "sinus pain":                   "sinus_pressure",
    "chest pain":                   "chest_pain",
    "chest tightness":              "chest_pain",
    "chest hurts":                  "chest_pain",
    "pain in my chest":             "chest_pain",
    "chest is tight":               "chest_pain",
    "tightness in chest":           "chest_pain",
    "heavy chest":                  "chest_pain",
    "brick on my chest":            "chest_pain",
    "pressure on chest":            "chest_pain",
    "chest feels like":             "chest_pain",
    "chest discomfort":             "chest_pain",

    # ── Gastrointestinal ──────────────────────────────────────────────────────
    "nausea":                       "nausea",
    "feel sick":                    "nausea",
    "feeling sick":                 "nausea",
    "sick to stomach":              "nausea",
    "want to vomit":                "nausea",
    "feel like vomiting":           "nausea",
    "queasy":                       "nausea",
    "throwing up":                  "vomiting",
    "threw up":                     "vomiting",
    "vomit":                        "vomiting",
    "vomiting":                     "vomiting",
    "been sick":                    "vomiting",
    "puking":                       "vomiting",
    "puke":                         "vomiting",
    "stomach pain":                 "stomach_pain",
    "tummy pain":                   "stomach_pain",
    "stomach ache":                 "stomach_pain",
    "stomach hurts":                "stomach_pain",
    "tummy ache":                   "stomach_pain",
    "stomach is paining me":        "stomach_pain",
    "my stomach":                   "stomach_pain",
    "belly pain":                   "belly_pain",
    "abdominal pain":               "abdominal_pain",
    "stomach cramps":               "cramps",
    "cramps":                       "cramps",
    "menstrual cramps":             "cramps",
    "diarrhea":                     "diarrhoea",
    "diarrhoea":                    "diarrhoea",
    "loose stool":                  "diarrhoea",
    "loose motion":                 "diarrhoea",
    "watery stool":                 "diarrhoea",
    "runny stool":                  "diarrhoea",
    "frequent stool":               "diarrhoea",
    "frequent toilet":              "diarrhoea",
    "running stomach":              "diarrhoea",
    "stomach running":              "diarrhoea",
    "toilet many times":            "diarrhoea",
    "constipated":                  "constipation",
    "constipation":                 "constipation",
    "cannot poop":                  "constipation",
    "can't poop":                   "constipation",
    "cannot pass stool":            "constipation",
    "hard to pass stool":           "constipation",
    "no bowel movement":            "constipation",
    "heartburn":                    "acidity",
    "acid reflux":                  "acidity",
    "burning in throat":            "acidity",
    "indigestion":                  "indigestion",
    "bloating":                     "passage_of_gases",
    "bloated":                      "passage_of_gases",
    "gas":                          "passage_of_gases",
    "flatulence":                   "passage_of_gases",
    "stomach making noise":         "passage_of_gases",
    "stomach gurgling":             "passage_of_gases",
    "no appetite":                  "loss_of_appetite",
    "loss of appetite":             "loss_of_appetite",
    "not hungry":                   "loss_of_appetite",
    "cannot eat":                   "loss_of_appetite",
    "can't eat":                    "loss_of_appetite",
    "don't feel like eating":       "loss_of_appetite",
    "no interest in food":          "loss_of_appetite",
    "food doesn't appeal":          "loss_of_appetite",
    "very hungry":                  "excessive_hunger",
    "increased appetite":           "increased_appetite",

    # ── Pain ──────────────────────────────────────────────────────────────────
    "headache":                     "headache",
    "head pain":                    "headache",
    "head hurts":                   "headache",
    "head is paining me":           "headache",
    "my head is paining":           "headache",
    "head is paining":              "headache",
    "migraine":                     "headache",
    "head is heavy":                "headache",
    "pounding head":                "headache",
    "pounding headache":            "headache",
    "throbbing head":               "headache",
    "back pain":                    "back_pain",
    "back hurts":                   "back_pain",
    "pain in my back":              "back_pain",
    "backache":                     "back_pain",
    "lower back pain":              "back_pain",
    "my back":                      "back_pain",
    "neck pain":                    "neck_pain",
    "stiff neck":                   "stiff_neck",
    "neck is stiff":                "stiff_neck",
    "can't turn neck":              "stiff_neck",
    "joint pain":                   "joint_pain",
    "joints hurt":                  "joint_pain",
    "aching joints":                "joint_pain",
    "joint ache":                   "joint_pain",
    "knee pain":                    "knee_pain",
    "knees hurt":                   "knee_pain",
    "hip pain":                     "hip_joint_pain",
    "muscle pain":                  "muscle_pain",
    "muscle ache":                  "muscle_pain",
    "body ache":                    "muscle_pain",
    "body pain":                    "muscle_pain",
    "body is paining me":           "muscle_pain",
    "body dey pain me":             "muscle_pain",
    "my whole body hurts":          "muscle_pain",
    "everything hurts":             "muscle_pain",
    "aching all over":              "muscle_pain",
    "general body pain":            "muscle_pain",
    "feel achy":                    "muscle_pain",
    "pain all over":                "muscle_pain",
    "painful urination":            "burning_micturition",
    "burning urination":            "burning_micturition",
    "burning when urinating":       "burning_micturition",
    "pain when peeing":             "burning_micturition",
    "pain during urination":        "burning_micturition",
    "stinging when peeing":         "burning_micturition",

    # ── Skin ──────────────────────────────────────────────────────────────────
    "rash":                         "skin_rash",
    "skin rash":                    "skin_rash",
    "rashes":                       "skin_rash",
    "spots on skin":                "skin_rash",
    "red patches":                  "skin_rash",
    "spots on my body":             "skin_rash",
    "itchy":                        "itching",
    "itch":                         "itching",
    "itching":                      "itching",
    "scratching":                   "itching",
    "my skin is itching":           "itching",
    "skin dey itch":                "itching",
    "yellow skin":                  "yellowish_skin",
    "skin turned yellow":           "yellowish_skin",
    "jaundice":                     "yellowish_skin",
    "yellowing of skin":            "yellowish_skin",
    "skin is yellow":               "yellowish_skin",
    "yellow eyes":                  "yellowing_of_eyes",
    "eyes are yellow":              "yellowing_of_eyes",
    "yellowing of eyes":            "yellowing_of_eyes",
    "blister":                      "blister",
    "blisters":                     "blister",
    "pimples":                      "pus_filled_pimples",
    "blackheads":                   "blackheads",
    "skin peeling":                 "skin_peeling",
    "peeling skin":                 "skin_peeling",

    # ── General / Systemic ────────────────────────────────────────────────────
    "tired":                        "fatigue",
    "exhausted":                    "fatigue",
    "no energy":                    "fatigue",
    "weakness":                     "fatigue",
    "weak":                         "fatigue",
    "fatigue":                      "fatigue",
    "very weak":                    "fatigue",
    "feel weak":                    "fatigue",
    "body is weak":                 "fatigue",
    "body dey weak":                "fatigue",
    "feel off":                     "fatigue",
    "feel unwell":                  "fatigue",
    "feeling unwell":               "fatigue",
    "not feeling well":             "fatigue",
    "under the weather":            "fatigue",
    "run down":                     "fatigue",
    "drained":                      "fatigue",
    "lethargic":                    "lethargy",
    "lethargy":                     "lethargy",
    "sluggish":                     "lethargy",
    "dehydrated":                   "dehydration",
    "dehydration":                  "dehydration",
    "thirsty":                      "dehydration",
    "very thirsty":                 "dehydration",
    "sweating":                     "sweating",
    "night sweats":                 "sweating",
    "excessive sweating":           "sweating",
    "sweating a lot":               "sweating",
    "chills":                       "chills",
    "feeling cold":                 "chills",
    "cold and shivery":             "chills",
    "shivering":                    "shivering",
    "shaking":                      "shivering",
    "shaking with cold":            "shivering",
    "weight loss":                  "weight_loss",
    "losing weight":                "weight_loss",
    "lost weight":                  "weight_loss",
    "lose weight":                  "weight_loss",
    "weight gain":                  "weight_gain",
    "gained weight":                "weight_gain",
    "swollen lymph":                "swelled_lymph_nodes",
    "swollen glands":               "swelled_lymph_nodes",
    "swelling":                     "swelling_joints",
    "swollen joints":               "swelling_joints",
    "swollen legs":                 "swollen_legs",
    "puffy legs":                   "swollen_legs",

    # ── Neurological / Mental ─────────────────────────────────────────────────
    "dizzy":                        "dizziness",
    "dizziness":                    "dizziness",
    "feel dizzy":                   "dizziness",
    "light headed":                 "dizziness",
    "lightheaded":                  "dizziness",
    "room is spinning":             "spinning_movements",
    "spinning":                     "spinning_movements",
    "vertigo":                      "spinning_movements",
    "loss of balance":              "loss_of_balance",
    "unsteady":                     "unsteadiness",
    "keep falling":                 "unsteadiness",
    "blurry vision":                "blurred_and_distorted_vision",
    "blurred vision":               "blurred_and_distorted_vision",
    "can't see clearly":            "blurred_and_distorted_vision",
    "vision is blurry":             "blurred_and_distorted_vision",
    "can't smell":                  "loss_of_smell",
    "loss of smell":                "loss_of_smell",
    "no sense of smell":            "loss_of_smell",
    "anxious":                      "anxiety",
    "anxiety":                      "anxiety",
    "nervous":                      "anxiety",
    "worried all the time":         "anxiety",
    "depressed":                    "depression",
    "depression":                   "depression",
    "very sad":                     "depression",
    "mood swings":                  "mood_swings",
    "irritable":                    "irritability",
    "easily irritated":             "irritability",
    "can't concentrate":            "lack_of_concentration",
    "lack of concentration":        "lack_of_concentration",
    "forgetful":                    "lack_of_concentration",
    "confused":                     "lack_of_concentration",
    "memory problems":              "lack_of_concentration",

    # ── Cardiac / Vascular ────────────────────────────────────────────────────
    "heart racing":                 "fast_heart_rate",
    "fast heartbeat":               "fast_heart_rate",
    "racing heart":                 "fast_heart_rate",
    "heart beating fast":           "fast_heart_rate",
    "heart pounding":               "palpitations",
    "palpitations":                 "palpitations",
    "irregular heartbeat":          "palpitations",
    "heart skipping beats":         "palpitations",
    "heart is jumping":             "palpitations",

    # ── Urinary ───────────────────────────────────────────────────────────────
    "frequent urination":           "continuous_feel_of_urine",
    "need to pee often":            "continuous_feel_of_urine",
    "always need to pee":           "continuous_feel_of_urine",
    "passing urine frequently":     "continuous_feel_of_urine",
    "dark urine":                   "dark_urine",
    "brown urine":                  "dark_urine",
    "urine is dark":                "dark_urine",
    "blood in urine":               "burning_micturition",
    "urinary discomfort":           "bladder_discomfort",
    "bladder pain":                 "bladder_discomfort",

    # ── Eyes ──────────────────────────────────────────────────────────────────
    "red eyes":                     "redness_of_eyes",
    "eyes are red":                 "redness_of_eyes",
    "bloodshot eyes":               "redness_of_eyes",
    "eye pain":                     "pain_behind_the_eyes",
    "pain behind eyes":             "pain_behind_the_eyes",
    "eyes hurt":                    "pain_behind_the_eyes",
    "watery eyes":                  "watering_from_eyes",
    "tears":                        "watering_from_eyes",
    "eyes watering":                "watering_from_eyes",
    "eyes are watering":            "watering_from_eyes",

    # ── Nigerian / Pidgin English expressions ─────────────────────────────────
    "e dey pain me":                "muscle_pain",
    "im dey pain":                  "muscle_pain",
    "body no balance":              "unsteadiness",
    "head dey spin":                "spinning_movements",
    "head dey heavy":               "headache",
    "stomach dey do me":            "stomach_pain",
    "stomach dey turn":             "nausea",
    "e dey make me feel somehow":   "fatigue",
    "my yansh dey pain me":         "abdominal_pain",
    "i feel anyhow":                "fatigue",
    "i dey feel somehow":           "fatigue",
    "temperature don rise":         "high_fever",
    "e dey hot":                    "high_fever",
    "e dey cold":                   "chills",
    "e dey shake":                  "shivering",
    "body dey itch":                "itching",
}


# =============================================================================
# NEGATION PATTERNS  — strip negated phrases before symptom matching
# =============================================================================

# e.g. "I don't have a fever" → remove "fever" from the matching window
_NEGATION_PREFIXES = (
    r"don'?t\s+have",
    r"do\s+not\s+have",
    r"no\s+",
    r"without\s+",
    r"denies?\s+",
    r"no\s+sign\s+of",
    r"doesn'?t\s+have",
    r"does\s+not\s+have",
    r"haven'?t\s+had",
    r"has\s+not\s+had",
    r"not\s+experiencing",
    r"absence\s+of",
)

_NEG_PATTERN = re.compile(
    r"(" + "|".join(_NEGATION_PREFIXES) + r")\s+(\w[\w\s]{0,30})",
    re.IGNORECASE,
)


def _strip_negations(text: str) -> str:
    """
    Remove negated symptom phrases from text so they don't get matched.
    e.g. "I don't have a fever but I do have a headache"
         → "I  but I do have a headache"
    """
    return _NEG_PATTERN.sub("", text)


# =============================================================================
# LAYER 1 — GEMINI AI EXTRACTION
# =============================================================================

def _extract_with_gemini(
    text: str,
    all_symptoms: list[str],
) -> Optional[tuple[list[str], list[str]]]:
    """
    Use Gemini Flash (free tier) to extract symptoms from natural language text.
    Uses the new google.genai SDK.

    Returns (matched_symptoms, explanations) or None if API is unavailable.
    Set GEMINI_API_KEY in .env or in the constant above to activate.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        symptoms_list = "\n".join(f"- {s}" for s in all_symptoms)

        prompt = f"""You are a medical symptom extraction assistant.
A patient has written the following description of how they feel:

PATIENT TEXT:
\"\"\"{text}\"\"\"

TASK:
Read the patient text carefully. Identify which symptoms from the SYMPTOM LIST below
are mentioned or strongly implied — including indirect descriptions, body-part references,
idioms, and non-English (e.g. Nigerian English or Pidgin) expressions.

IMPORTANT RULES:
1. Only return symptoms from the SYMPTOM LIST — do not invent new ones.
2. Do NOT include symptoms that are explicitly denied (e.g. "I don't have a fever").
3. For each match, explain in plain English why you matched it.
4. Return ONLY valid JSON — no markdown, no explanation outside the JSON.

SYMPTOM LIST:
{symptoms_list}

RESPONSE FORMAT (return ONLY this JSON):
{{
  "matched": ["symptom_name_1", "symptom_name_2"],
  "explanations": [
    "symptom_name_1: matched because ...",
    "symptom_name_2: matched because ..."
  ]
}}"""

        # Retry up to 3 times with exponential backoff on rate limit
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=prompt,
                )
                raw = response.text.strip()
                # Strip markdown code fences if present
                raw = re.sub(r"^```json\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                matched = [s for s in data.get("matched", []) if s in all_symptoms]
                explanations = data.get("explanations", [])
                logger.info("Gemini extracted %d symptoms from description.", len(matched))
                return sorted(matched), explanations
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    wait = 2 ** attempt
                    logger.warning("Gemini rate limited. Waiting %ds...", wait)
                    time.sleep(wait)
                    continue
                logger.warning("Gemini extraction error: %s", e)
                return None

        logger.warning("Gemini: all retries exhausted. Falling back to offline.")
        return None

    except ImportError:
        logger.warning("google-genai not installed. Run: pip install google-genai")
        return None
    except Exception as exc:
        logger.warning("Gemini unavailable: %s", exc)
        return None


# =============================================================================
# LAYER 2 + 3 — OFFLINE EXTRACTION (synonym map + fuzzy)
# =============================================================================

def _extract_offline(
    text: str,
    all_symptoms: list[str],
    fuzzy: bool = True,
    fuzzy_cutoff: float = 0.82,
) -> tuple[list[str], list[str]]:
    """
    Offline symptom extraction using the expanded synonym map and fuzzy matching.
    Includes negation stripping.
    """
    text_clean = _strip_negations(text.lower().strip())
    matched: set[str] = set()
    explanations: list[str] = []

    sym_display = {s.replace("_", " ").strip().lower(): s for s in all_symptoms}

    # ── Pass 1: expanded synonym map ─────────────────────────────────────────
    for phrase, token in _SYNONYMS.items():
        if phrase in text_clean:
            for sym in all_symptoms:
                if sym == token or sym.startswith(token.rstrip("_")):
                    if sym not in matched:
                        matched.add(sym)
                        explanations.append(
                            f'"{phrase}" → **{sym.replace("_", " ")}**'
                        )
                    break

    # ── Pass 2: direct keyword match (whole-word boundary) ───────────────────
    for display, original in sym_display.items():
        if original in matched:
            continue
        pattern = r"\b" + re.escape(display) + r"\b"
        if re.search(pattern, text_clean):
            matched.add(original)
            explanations.append(
                f'"{display}" → **{display}** (direct match)'
            )

    # ── Pass 3: fuzzy word-level match ────────────────────────────────────────
    if fuzzy:
        words = re.findall(r"\b\w{4,}\b", text_clean)
        display_names = list(sym_display.keys())
        for word in set(words):
            closes = get_close_matches(word, display_names, n=1, cutoff=fuzzy_cutoff)
            if closes:
                original = sym_display[closes[0]]
                if original not in matched:
                    matched.add(original)
                    explanations.append(
                        f'"{word}" ≈ **{closes[0]}** (fuzzy match)'
                    )

    return sorted(matched), explanations


# =============================================================================
# PUBLIC API
# =============================================================================

def extract_symptoms_from_text(
    text: str,
    all_symptoms: list[str],
    fuzzy: bool = True,
    fuzzy_cutoff: float = 0.82,
) -> tuple[list[str], list[str]]:
    """
    Extract symptom names from a free-text patient description.

    Uses a 3-layer hybrid pipeline:
      1. Gemini Flash AI (if GEMINI_API_KEY is set) — context-aware
      2. Expanded synonym map (offline, 300+ entries, negation stripping)
      3. Fuzzy / direct match (fallback safety net)

    Parameters
    ----------
    text : str
        Patient's free-text description (any language/case/punctuation).
    all_symptoms : list[str]
        Full list of known symptom column names (underscore-separated).
    fuzzy : bool
        Whether to apply fuzzy matching in offline mode.
    fuzzy_cutoff : float
        Minimum similarity ratio for fuzzy matches (0–1).

    Returns
    -------
    matched : list[str]
        Symptom column names found in the text (sorted).
    explanations : list[str]
        Human-readable strings explaining each match (for UI display).
    engine : str   ← also returned as 3rd value for UI badge
        "gemini" | "offline"
    """
    if not text or not text.strip():
        return [], [], "offline"

    # Layer 1: Gemini AI
    ai_result = _extract_with_gemini(text, all_symptoms)
    if ai_result is not None:
        matched, explanations = ai_result
        return matched, explanations, "gemini"

    # Layers 2+3: offline
    matched, explanations = _extract_offline(text, all_symptoms, fuzzy, fuzzy_cutoff)
    return matched, explanations, "offline"


# Back-compat alias — keeps the old 2-return-value signature working
def extract_symptoms(
    text: str,
    all_symptoms: list[str],
    fuzzy: bool = True,
    fuzzy_cutoff: float = 0.82,
) -> tuple[list[str], list[str]]:
    """Alias for extract_symptoms_from_text (returns matched, explanations only)."""
    matched, explanations, _ = extract_symptoms_from_text(
        text, all_symptoms, fuzzy, fuzzy_cutoff
    )
    return matched, explanations
