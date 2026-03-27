from __future__ import annotations
"""
Onboarding: Sammelt User-Daten und berechnet Tages-Goals.
Harris-Benedict Formel fuer Grundumsatz + Aktivitaetsfaktor.
"""


def calculate_goals(age: int, gender: str, weight_kg: float,
                    height_cm: float, goal: str) -> dict:
    """
    Berechnet Tages-Makro-Goals basierend auf Harris-Benedict.

    gender: "m" oder "f"
    goal: "abnehmen", "halten", "aufbauen"
    """
    # Grundumsatz (BMR)
    if gender == "m":
        bmr = 88.36 + (13.4 * weight_kg) + (4.8 * height_cm) - (5.7 * age)
    else:
        bmr = 447.6 + (9.2 * weight_kg) + (3.1 * height_cm) - (4.3 * age)

    # Moderater Aktivitaetsfaktor (1-2x Sport/Woche als Default)
    tdee = bmr * 1.55

    # Ziel-Anpassung
    if goal == "abnehmen":
        kcal = tdee - 500
    elif goal == "aufbauen":
        kcal = tdee + 300
    else:  # halten
        kcal = tdee

    kcal = round(kcal)

    # Makros berechnen
    # Protein: 2g/kg bei Aufbau, 1.8g/kg bei Halten, 1.6g/kg bei Abnehmen
    protein_factor = {"aufbauen": 2.0, "halten": 1.8, "abnehmen": 1.6}
    protein_g = round(weight_kg * protein_factor.get(goal, 1.8))

    # Fett: 25% der Kalorien
    fat_g = round((kcal * 0.25) / 9)

    # Carbs: Rest
    carbs_g = round((kcal - (protein_g * 4) - (fat_g * 9)) / 4)

    return {
        "daily_kcal": kcal,
        "daily_protein_g": protein_g,
        "daily_carbs_g": carbs_g,
        "daily_fat_g": fat_g,
        "daily_fiber_g": 30,  # Standard-Empfehlung
        "goal": goal,
        "tdee": round(tdee),
    }


# Onboarding-Schritte als State Machine
STEPS = ["gender", "age", "weight", "height", "goal"]

STEP_PROMPTS = {
    "gender": (
        "Willkommen! Ich berechne deine persoenlichen Ernaehrungs-Ziele.\n\n"
        "Schritt 1/5 - Geschlecht:\n"
        "Antworte mit *m* (maennlich) oder *f* (weiblich)"
    ),
    "age": "Schritt 2/5 - Alter:\nWie alt bist du? (z.B. *25*)",
    "weight": "Schritt 3/5 - Gewicht:\nWie viel wiegst du in kg? (z.B. *80*)",
    "height": "Schritt 4/5 - Groesse:\nWie gross bist du in cm? (z.B. *180*)",
    "goal": (
        "Schritt 5/5 - Ziel:\n\n"
        "*1* - Abnehmen (-500 kcal/Tag)\n"
        "*2* - Gewicht halten\n"
        "*3* - Muskelaufbau (+300 kcal/Tag)\n\n"
        "Antworte mit 1, 2 oder 3"
    ),
}

GOAL_MAP = {"1": "abnehmen", "2": "halten", "3": "aufbauen"}
GENDER_MAP = {"m": "m", "männlich": "m", "maennlich": "m", "male": "m",
              "f": "f", "weiblich": "f", "female": "f", "w": "f"}


class OnboardingState:
    """Verwaltet den Onboarding-Fortschritt eines Users."""

    def __init__(self):
        # {telegram_id: {"step": "gender", "data": {...}}}
        self.sessions: dict = {}

    def is_active(self, telegram_id: int) -> bool:
        return telegram_id in self.sessions

    def start(self, telegram_id: int) -> str:
        self.sessions[telegram_id] = {"step": "gender", "data": {}}
        return STEP_PROMPTS["gender"]

    def process(self, telegram_id: int, text: str) -> tuple[str, dict | None]:
        """
        Verarbeitet User-Input fuer den aktuellen Schritt.
        Returns: (response_text, goals_dict_or_None)
        goals_dict ist gesetzt wenn Onboarding abgeschlossen.
        """
        session = self.sessions.get(telegram_id)
        if not session:
            return "Kein aktives Onboarding.", None

        step = session["step"]
        data = session["data"]
        lower = text.strip().lower()

        # --- Gender ---
        if step == "gender":
            gender = GENDER_MAP.get(lower)
            if not gender:
                return "Bitte antworte mit *m* oder *f*.", None
            data["gender"] = gender
            session["step"] = "age"
            return STEP_PROMPTS["age"], None

        # --- Age ---
        if step == "age":
            try:
                age = int(text.strip())
                if not 10 <= age <= 100:
                    raise ValueError
            except ValueError:
                return "Bitte eine gueltige Zahl eingeben (z.B. *25*).", None
            data["age"] = age
            session["step"] = "weight"
            return STEP_PROMPTS["weight"], None

        # --- Weight ---
        if step == "weight":
            try:
                weight = float(text.strip().replace(",", "."))
                if not 30 <= weight <= 300:
                    raise ValueError
            except ValueError:
                return "Bitte ein gueltiges Gewicht eingeben (z.B. *80*).", None
            data["weight_kg"] = weight
            session["step"] = "height"
            return STEP_PROMPTS["height"], None

        # --- Height ---
        if step == "height":
            try:
                height = float(text.strip().replace(",", "."))
                if not 100 <= height <= 250:
                    raise ValueError
            except ValueError:
                return "Bitte eine gueltige Groesse eingeben (z.B. *180*).", None
            data["height_cm"] = height
            session["step"] = "goal"
            return STEP_PROMPTS["goal"], None

        # --- Goal ---
        if step == "goal":
            # Manuelle Eingabe erlauben
            if lower.startswith("manuell") or lower.startswith("manual"):
                del self.sessions[telegram_id]
                return (
                    "OK! Gib deine Ziele manuell ein:\n"
                    "Format: *ziel: 2000 kcal, 150g protein, 60g fett, 200g carbs*"
                ), None

            goal_key = GOAL_MAP.get(text.strip())
            if not goal_key:
                return "Bitte antworte mit *1*, *2* oder *3*.", None

            data["goal"] = goal_key
            goals = calculate_goals(
                age=data["age"],
                gender=data["gender"],
                weight_kg=data["weight_kg"],
                height_cm=data["height_cm"],
                goal=goal_key,
            )

            del self.sessions[telegram_id]

            summary = (
                f"Perfekt! Deine Tages-Ziele:\n\n"
                f"Ziel: *{goal_key.capitalize()}*\n"
                f"Grundumsatz: {goals['tdee']} kcal\n"
                f"Tagesziel: *{goals['daily_kcal']} kcal*\n"
                f"Protein: *{goals['daily_protein_g']}g*\n"
                f"Kohlenhydrate: *{goals['daily_carbs_g']}g*\n"
                f"Fett: *{goals['daily_fat_g']}g*\n"
                f"Ballaststoffe: *{goals['daily_fiber_g']}g*\n\n"
                f"Los geht's! Schick mir ein Foto oder schreib was du gegessen hast."
            )
            return summary, goals

        return "Unbekannter Schritt.", None