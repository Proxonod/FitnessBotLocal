from __future__ import annotations
"""
Orchestrator: The brain of the Health Tracker.

Handles the flow:
  1. Detect intent (photo analysis, meal logging, query, etc.)
  2. Route to appropriate model (LLaVA for images, Llama for text)
  3. Interact with database
  4. Return response to user

Sprint 1: Basic food recognition + calorie estimation + meal logging
Sprint 2: Product DB with Hybrid Search, OCR for labels
Sprint 3: Plotting, cookbook, advanced agent logic
"""

import json
from src.model_manager import ModelManager, UsageStats
from src.database import Database
from src.config import TEXT_MODEL, VISION_MODEL
from src.vector_store import VectorStore
from src.food_lookup import FoodLookup

SYSTEM_PROMPT = """Du bist ein freundlicher Ernährungs-Tracker-Assistent.
Du hilfst beim Tracken von Mahlzeiten, Kalorien und Makronährstoffen.

Wenn du ein Foto eines Gerichts analysierst, antworte IMMER im folgenden JSON-Format:
{
  "description": "Kurze Beschreibung des Gerichts",
  "items": [
    {
      "name": "Zutat/Gericht",
      "amount_g": 200,
      "kcal": 300,
      "protein_g": 15,
      "carbs_g": 40,
      "fat_g": 10,
      "fiber_g": 3
    }
  ],
  "confidence": "low/medium/high",
  "notes": "Optionale Hinweise, z.B. 'Portion schwer zu schätzen'"
}

Wenn der User eine Menge angibt (z.B. "300g Käse"), nutze diese Info.
Wenn nicht, schätze eine typische Portion.
Sei ehrlich wenn du unsicher bist — lieber nachfragen als falsch schätzen.
Antworte auf Deutsch."""


PORTION_PROMPT = """Der User hat gesagt: "{user_text}"

Kontext: Heute wurden bereits diese Mahlzeiten geloggt:
{meals_today}

Berechne die Nährwerte basierend auf der Angabe.
Wenn der User eine relative Menge angibt (z.B. "die Hälfte davon",
"1/3 von gestern"), berechne das entsprechend.

Antworte im JSON-Format:
{{
  "description": "Was wurde gegessen",
  "items": [...],
  "total": {{"kcal": X, "protein_g": X, "carbs_g": X, "fat_g": X, "fiber_g": X}},
  "portion": 1.0
}}"""


class Orchestrator:
    def __init__(self):
        self.model_manager = ModelManager()
        self.db = Database()
        self.vector_store = VectorStore()
        self.food_lookup = FoodLookup()
        # Conversation state per user: {telegram_id: {"last_action": ..., "last_meal": ...}}
        self.user_state = {}

    def _detect_intent(self, text: str, telegram_id: int) -> dict:
        """
        Deterministische Intent-Erkennung OHNE LLM.
        Sammelt alle Signale aus der Nachricht, dann baut es einen
        kombinierten Intent zusammen.
        """
        import re
        lower = text.lower().strip()
        state = self.user_state.get(telegram_id, {})
        last_action = state.get("last_action")

        # --- Signal-Sammler ---
        signals = {
            "is_command": False,
            "command": None,
            "is_confirmation": False,
            "is_correction": False,
            "is_nachtrag": False,
            "macros": {},
            "product_ref": None,
            "amount_g": None,
            "portion": None,
            "is_greeting": False,
            "raw_text": text,
        }

        # --- 1. Befehle (sofort raus, eindeutig) ---
        if lower.startswith("/") or lower.startswith("\\"):
            return {"intent": "command", "data": lower}

        # --- 2. Grüße ---
        greetings = ("hi", "hallo", "hey", "moin", "start", "servus", "sers")
        if lower in greetings:
            return {"intent": "command", "data": "/start"}

        # --- 3. Bestätigungs-Wörter suchen ---
        confirm_words = {"passt", "ja", "ok", "stimmt", "korrekt", "yes",
                         "da", "richtig", "mach", "jap", "jep", "genau",
                         "👍", "korrekt", "perfekt", "super", "da"}
        words = set(re.findall(r"\w+", lower))
        if words & confirm_words:  # Schnittmenge
            signals["is_confirmation"] = True

        # --- 4. Korrektur-Signale suchen ---
        correction_starts = ("nein", "falsch", "korrigiere", "korrektur",
                             "no", "nicht", "änder", "änderung", "stimmt nicht",
                             "falsch", "war eigentlich", "sollte sein")
        correction_words = {"dumm", "dummbatz", "idiot", "falsch", "fehler",
                            "vergessen", "übersehen"}
        if any(lower.startswith(c) for c in correction_starts):
            signals["is_correction"] = True
        if words & correction_words:
            signals["is_correction"] = True

        # --- 5. Makro-Angaben extrahieren ---
        macro_aliases = {
            "eiweiß": "protein_g", "eiweiss": "protein_g", "eiweis": "protein_g",
            "protein": "protein_g", "prot": "protein_g",
            "carbs": "carbs_g", "karbs": "carbs_g", "kohlenhydrate": "carbs_g",
            "kohlh": "carbs_g", "kh": "carbs_g",
            "fett": "fat_g", "fat": "fat_g",
            "ballaststoffe": "fiber_g", "ballast": "fiber_g", "ballst": "fiber_g",
            "fiber": "fiber_g",
            "kcal": "kcal", "kalorien": "kcal", "cal": "kcal",
        }

        macro_hits = re.findall(r"(\d+)\s*g?\s+(\w+)", lower)
        for value, name in macro_hits:
            key = macro_aliases.get(name)
            if key:
                signals["macros"][key] = int(value)

        # --- 6. Nachtrag-Signale ---
        nachtrag_words = {"noch", "dazu", "außerdem", "plus", "zusätzlich",
                          "hinzu", "füg", "füge", "ergänze", "add", "achja"}
        if words & nachtrag_words:
            signals["is_nachtrag"] = True

        # --- 7. Produkt-Referenz: "für den Gouda", "zum Käse" ---
        product_match = re.search(r"(?:für|vom|zum|den|dem|das)\s+(\w+)", lower)
        if product_match:
            signals["product_ref"] = product_match.group(1)

        # --- 8. Einzelprodukt mit Menge: "300g Käse" ---
        single_match = re.match(r"^(\d+)\s*g\s+(.+)$", lower)
        if single_match and not signals["macros"]:
            signals["amount_g"] = int(single_match.group(1))
            signals["product_ref"] = single_match.group(2).strip()

        # --- 9. Portions-Angaben ---
        portion_patterns = {
            "die hälfte": 0.5, "hälfte": 0.5, "halb": 0.5,
            "ein drittel": 1 / 3, "1/3": 1 / 3, "drittel": 1 / 3,
            "ein viertel": 0.25, "1/4": 0.25, "viertel": 0.25,
            "dreiviertel": 0.75, "3/4": 0.75,
            "alles": 1.0, "komplett": 1.0,
        }
        for pattern, factor in portion_patterns.items():
            if pattern in lower:
                signals["portion"] = factor
                break

        # ============================================
        # INTENT-AUFLÖSUNG: Signale kombinieren
        # ============================================

        has_macros = bool(signals["macros"])
        has_product = bool(signals["product_ref"])
        has_portion = signals["portion"] is not None
        is_awaiting = last_action == "awaiting_confirmation"

        # Fall 1: "Ja dummbatz, füg noch 20g Eiweiß hinzu"
        #  → Bestätigung + Nachtrag + Makros = add_macros
        if signals["is_confirmation"] and signals["is_nachtrag"] and has_macros:
            return {
                "intent": "confirm_and_add",
                "data": {
                    "macros": signals["macros"],
                    "product_ref": signals["product_ref"],
                },
            }

        # Fall 2: "Achja noch 10g fiber"
        #  → Nachtrag + Makros (ohne explizite Bestätigung)
        if signals["is_nachtrag"] and has_macros:
            return {
                "intent": "add_macros",
                "data": signals["macros"],
            }

        # Fall 3: Reine Bestätigung: "Passt", "Ja korrekt"
        if signals["is_confirmation"] and not has_macros and not signals["is_correction"]:
            if is_awaiting:
                return {"intent": "confirm_meal", "data": state.get("last_meal")}

        # Fall 4: Korrektur mit neuen Werten: "Nein, waren 400 kcal"
        if signals["is_correction"] and has_macros:
            return {
                "intent": "correct_with_values",
                "data": signals["macros"],
            }

        # Fall 5: Korrektur ohne Werte: "Das stimmt nicht"
        if signals["is_correction"] and not has_macros:
            return {"intent": "correct_meal", "data": text}

        # Fall 6: Nur Makros ohne Kontext: "20g Protein, 30g Carbs"
        if has_macros:
            return {"intent": "quick_macros", "data": signals["macros"]}

        # Fall 7: Produkt mit Menge: "300g Käse"
        if signals["amount_g"] and has_product:
            return {
                "intent": "quick_amount",
                "data": {"amount_g": signals["amount_g"], "name": signals["product_ref"]},
            }

        # Fall 8: Portions-Angabe: "Die Hälfte davon"
        if has_portion:
            return {"intent": "portion_update", "data": signals["portion"]}

        # Fall 9: Nichts Konkretes erkannt → LLM
        return {"intent": "llm_needed", "data": text}

    async def handle_photo(self, telegram_id: int, user_name: str,
                           photo_bytes: bytes,
                           caption: str | None = None) -> str:
        """Handle a photo message — food recognition or label OCR."""
        self.db.get_or_create_user(telegram_id, user_name)

        # Build prompt based on whether there's a caption
        # Step 1: LLaVA beschreibt frei was es sieht (kein JSON-Zwang!)
        if caption:
            vision_prompt = (
                f"Der User sagt dazu: '{caption}'\n\n"
                f"Beschreibe was du auf dem Foto siehst. "
                f"Wenn es Essen ist, schätze Zutaten und Mengen. "
                f"Wenn es ein Nährwert-Etikett ist, lies alle Werte ab."
            )
        else:
            vision_prompt = (
                "Beschreibe was du auf dem Foto siehst. "
                "Wenn es Essen ist, benenne die Zutaten und schätze Mengen. "
                "Wenn es ein Nährwert-Etikett ist, lies alle Werte ab."
            )

        # Step 1: Vision model analyzes the image
        print(f"  👁️ Analyzing photo with {VISION_MODEL}...")
        vision_response, vision_stats = self.model_manager.vision(
            photo_bytes, vision_prompt
        )

        # Log usage
        self.db.log_usage(telegram_id, {
            "model": VISION_MODEL,
            "prompt_tokens": vision_stats.prompt_tokens,
            "completion_tokens": vision_stats.completion_tokens,
            "duration_seconds": vision_stats.duration_seconds,
            "energy_kwh": vision_stats.energy_kwh,
            "energy_cost_eur": vision_stats.energy_cost_eur,
            "estimated_api_cost_usd": vision_stats.estimated_api_cost_usd,
            "action": "photo_analysis",
        })

        # Step 2: Llama parst die Beschreibung in strukturiertes JSON
        parse_prompt = f"""Der User hat ein Foto geschickt. Hier ist die Bildbeschreibung:

        "{vision_response}"

        Extrahiere die Nährwerte und antworte NUR mit diesem JSON, kein anderer Text:
        {{
          "description": "kurze Beschreibung",
          "items": [
            {{
              "name": "Zutat",
              "amount_g": 100,
              "kcal": 0,
              "protein_g": 0,
              "carbs_g": 0,
              "fat_g": 0,
              "fiber_g": 0
            }}
          ],
          "confidence": "low/medium/high"
        }}"""

        print(f"  🧠 Parsing with {TEXT_MODEL}...")
        parse_response, parse_stats = self.model_manager.reason([
            {"role": "system", "content": "Du bist ein JSON-Parser. Antworte NUR mit validem JSON, kein anderer Text."},
            {"role": "user", "content": parse_prompt},
        ])

        # Log usage for step 2
        self.db.log_usage(telegram_id, {
            "model": TEXT_MODEL,
            "prompt_tokens": parse_stats.prompt_tokens,
            "completion_tokens": parse_stats.completion_tokens,
            "duration_seconds": parse_stats.duration_seconds,
            "energy_kwh": parse_stats.energy_kwh,
            "energy_cost_eur": parse_stats.energy_cost_eur,
            "estimated_api_cost_usd": parse_stats.estimated_api_cost_usd,
            "action": "photo_parsing",
        })

        # Step 2: Try to parse as structured data
        meal_data = self._try_parse_meal(parse_response)

        if meal_data:
            # Calculate totals
            total = {"kcal": 0, "protein_g": 0, "carbs_g": 0,
                     "fat_g": 0, "fiber_g": 0}
            for item in meal_data.get("items", []):
                for key in total:
                    total[key] += item.get(key, 0) or 0

            meal_data["total"] = total
            meal_data["portion"] = 1.0

            # Save to DB
            self.db.log_meal(telegram_id, meal_data)
            self.user_state[telegram_id] = {
                "last_action": "idle",
                "last_meal": meal_data,  # behalten für Korrekturen
            }

            # Format nice response
            response = self._format_meal_response(meal_data)
            return response
        else:
            # Model couldn't structure it — return raw response
            return (
                f"🔍 Das habe ich erkannt:\n\n{vision_response}\n\n"
                f"Ich konnte die Nährwerte nicht automatisch parsen. "
                f"Kannst du mir die Mengen genauer sagen?"
            )

    async def handle_text(self, telegram_id: int, user_name: str,
                          text: str) -> str:
        """Handle a text message with intent detection."""
        self.db.get_or_create_user(telegram_id, user_name)

        intent = self._detect_intent(text, telegram_id)
        print(f"  🎯 Intent: {intent['intent']}")

        # --- Commands ---
        if intent["intent"] == "command":
            cmd = intent["data"]
            if cmd in ("/start", "start"):
                return (
                    f"👋 Hey {user_name}! Ich bin dein Ernährungs-Tracker.\n\n"
                    f"📸 Schick mir ein Foto von deinem Essen\n"
                    f"📝 Schreib mir was du gegessen hast (z.B. '300g Hähnchen')\n"
                    f"📊 /today → Tagesübersicht\n"
                    f"📈 /stats → Verbrauch\n"
                    f"🧀 /products → Produkt-DB\n\n"
                    f"💡 _Du kannst jederzeit die letzte Mahlzeit korrigieren:_\n"
                    f"  • 'korrigiere: Protein war 30g'\n"
                    f"  • 'nein, waren nur 400 kcal'\n"
                    f"  • 'füg noch 20g Eiweiß hinzu'"
                )
            if cmd in ("/today", "heute"):
                return self._daily_summary(telegram_id)
            if cmd in ("/stats", "stats"):
                return self._usage_summary(telegram_id)
            if cmd in ("/products", "produkte"):
                return self._products_summary(telegram_id)

        # --- Bestätigung ---
        if intent["intent"] == "confirm_meal":
            self.user_state[telegram_id] = {"last_action": "idle"}
            return "✅ Gespeichert! Guten Appetit 🍽️"
        # --- Bestätigung + Nachtrag: "ja, aber füg noch 20g Eiweiß hinzu" ---
        if intent["intent"] == "confirm_and_add":
            macros = intent["data"]["macros"]
            state = self.user_state.get(telegram_id, {})
            last_meal = state.get("last_meal")
            if last_meal and "total" in last_meal:
                for key, value in macros.items():
                    last_meal["total"][key] = last_meal["total"].get(key, 0) + value
                self.db.log_meal(telegram_id, last_meal)
                self.user_state[telegram_id] = {"last_action": "idle"}
                found = ", ".join(f"{v}{'g' if 'kcal' not in k else ''} {k.replace('_g', '').replace('_', '')}"
                                  for k, v in macros.items())
                return f"✅ Gespeichert + ergänzt: {found}\n\n{self._format_meal_response(last_meal)}"
            self.user_state[telegram_id] = {"last_action": "idle"}
            return "✅ Gespeichert!"

        # --- Korrektur ---
        if intent["intent"] == "correct_meal":
            self.user_state[telegram_id] = {"last_action": "awaiting_correction"}
            return "✏️ Was soll ich korrigieren? Sag z.B. 'Protein war 25g' oder 'waren nur 200 kcal'"
        # --- Korrektur mit Werten: "Nein, waren 400 kcal" ---
        if intent["intent"] == "correct_with_values":
            macros = intent["data"]
            state = self.user_state.get(telegram_id, {})
            last_meal = state.get("last_meal")
            if last_meal and "total" in last_meal:
                for key, value in macros.items():
                    last_meal["total"][key] = value  # Überschreiben, nicht addieren!
                self.db.log_meal(telegram_id, last_meal)
                self.user_state[telegram_id] = {"last_action": "idle"}
                return f"✏️ Korrigiert!\n\n{self._format_meal_response(last_meal)}"
            return "🤔 Keine letzte Mahlzeit zum Korrigieren gefunden."
        # --- Quick kcal: "20 kcal" ---
        if intent["intent"] == "quick_kcal":
            kcal = intent["data"]
            meal_data = {
                "description": f"{kcal} kcal Snack",
                "items": [{"name": "Snack", "amount_g": None, "kcal": kcal,
                           "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fiber_g": 0}],
                "total": {"kcal": kcal, "protein_g": 0, "carbs_g": 0,
                          "fat_g": 0, "fiber_g": 0},
                "portion": 1.0,
                "confidence": "high",
            }
            self.db.log_meal(telegram_id, meal_data)
            self.user_state[telegram_id] = {"last_action": "idle"}
            return f"✅ {kcal} kcal geloggt!"
        # --- Quick macros: "20g Eiweiß, 30g Carbs" ---
        if intent["intent"] == "quick_macros":
            macros = intent["data"]
            meal_data = {
                "description": "Manuelle Makro-Eingabe",
                "items": [{"name": "Manuell", **macros}],
                "total": {
                    "kcal": macros.get("kcal", 0),
                    "protein_g": macros.get("protein_g", 0),
                    "carbs_g": macros.get("carbs_g", 0),
                    "fat_g": macros.get("fat_g", 0),
                    "fiber_g": macros.get("fiber_g", 0),
                },
                "portion": 1.0,
                "confidence": "high",
            }
            self.db.log_meal(telegram_id, meal_data)
            self.user_state[telegram_id] = {"last_action": "idle"}
            found = ", ".join(f"{v}{'g' if 'kcal' not in k else ''} {k.replace('_g', '').replace('_', '')}"
                              for k, v in macros.items())
            return f"✅ Geloggt: {found}"

        # --- Quick amount: "300g Käse" ---
        if intent["intent"] == "quick_amount":
            name = intent["data"]["name"]
            amount = intent["data"]["amount_g"]
            return await self._lookup_and_estimate(telegram_id, name, amount)

        # --- Portion update: "die Hälfte" ---
        if intent["intent"] == "portion_update":
            factor = intent["data"]
            state = self.user_state.get(telegram_id, {})
            last_meal = state.get("last_meal")
            if last_meal:
                scaled = {k: v * factor for k, v in last_meal.get("total", {}).items()}
                return (
                    f"📊 {factor:.0%} der letzten Mahlzeit:\n"
                    f"🔥 {scaled.get('kcal', 0):.0f} kcal\n"
                    f"🥩 {scaled.get('protein_g', 0):.0f}g Protein"
                )
            return "🤔 Wovon die Hälfte? Ich hab keine letzte Mahlzeit gespeichert."

        # --- LLM needed ---
        return await self._llm_estimate(telegram_id, text)

    async def _lookup_and_estimate(self, telegram_id: int,
                                   name: str, amount_g: float) -> str:
        """
        Lookup flow:
          1. Weaviate hybrid search
          2. OpenFoodFacts fallback (+ cache result in Weaviate)
          3. LLM fallback
        """
        # --- Step 1: Weaviate ---
        print(f"  🔍 Weaviate lookup: '{name}'...")
        product = self.vector_store.search_best(name)

        if product and product["_score"] > 0.3:
            print(f"  ✅ Weaviate hit: {product['name']} (score={product['_score']:.2f})")
            return self._calculate_portion(product, amount_g)

        # --- Step 2: OpenFoodFacts ---
        print(f"  🌍 OpenFoodFacts lookup: '{name}'...")
        off_product = self.food_lookup.search_best(name)

        if off_product:
            print(f"  ✅ OpenFoodFacts hit: {off_product['name']}")
            # Cache in Weaviate for next time
            off_product["added_by"] = telegram_id
            self.vector_store.add_product(off_product)

            return self._calculate_portion(off_product, amount_g)

        # --- Step 3: LLM fallback ---
        print(f"  🧠 No DB hit — falling back to LLM for '{name}'")
        return await self._llm_estimate(telegram_id, f"{amount_g}g {name}")

    def _calculate_portion(self, product: dict, amount_g: float) -> str:
        """Calculate macros for a given amount based on per-100g values."""
        factor = amount_g / 100.0
        per100 = product["per_100g"]

        total = {k: round(v * factor, 1) for k, v in per100.items()}

        name = product["name"]
        brand = product.get("brand", "")
        source = product.get("source", "")
        source_tag = "📦 OpenFoodFacts" if source == "openfoodfacts" else "🧀 OWN Produkt-DB"

        lines = [
            f"🍽️ **{name}**{f' ({brand})' if brand else ''}\n",
            f"📏 Portion: {amount_g}g\n",
            f"🔥 {total.get('kcal', 0)} kcal",
            f"🥩 {total.get('protein_g', 0)}g Protein",
            f"🍞 {total.get('carbs_g', 0)}g Kohlenhydrate",
            f"🧈 {total.get('fat_g', 0)}g Fett",
            f"🌿 {total.get('fiber_g', 0)}g Ballaststoffe",
            f"\n{source_tag}",
        ]
        return "\n".join(lines)



    async def _llm_estimate(self, telegram_id: int, text: str) -> str:
        """Fallback: LLM schätzt Nährwerte aus Freitext."""
        meals_today = self.db.get_daily_meals(telegram_id)
        meals_context = json.dumps(meals_today, default=str,
                                   ensure_ascii=False) if meals_today else "Keine."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PORTION_PROMPT.format(
                user_text=text, meals_today=meals_context
            )},
        ]

        print(f"  🧠 LLM estimate for: {text[:50]}...")
        response, stats = self.model_manager.reason(messages)

        self.db.log_usage(telegram_id, {
            "model": TEXT_MODEL,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "duration_seconds": stats.duration_seconds,
            "energy_kwh": stats.energy_kwh,
            "energy_cost_eur": stats.energy_cost_eur,
            "estimated_api_cost_usd": stats.estimated_api_cost_usd,
            "action": "text_estimate",
        })

        meal_data = self._try_parse_meal(response)
        if meal_data:
            total = {"kcal": 0, "protein_g": 0, "carbs_g": 0,
                     "fat_g": 0, "fiber_g": 0}
            for item in meal_data.get("items", []):
                for key in total:
                    total[key] += item.get(key, 0) or 0
            meal_data["total"] = total
            meal_data["portion"] = 1.0

            self.db.log_meal(telegram_id, meal_data)
            self.user_state[telegram_id] = {
                "last_action": "idle",
                "last_meal": meal_data,  # behalten für Korrekturen
            }
            return self._format_meal_response(meal_data)

        return response
    def _try_parse_meal(self, text: str) -> dict | None:
        """Try to extract JSON meal data from model response."""
        try:
            # Find JSON in response (model might add text around it)
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _format_meal_response(self, meal_data: dict) -> str:
        """Format meal data as a nice Telegram message."""
        desc = meal_data.get("description", "Mahlzeit")
        total = meal_data.get("total", {})
        confidence = meal_data.get("confidence", "medium")

        conf_emoji = {"low": "🟡", "medium": "🟢", "high": "✅"}.get(
            confidence, "🟢")

        lines = [f"🍽️ **{desc}**\n"]

        for item in meal_data.get("items", []):
            amount = item.get("amount_g", "?")
            lines.append(f"  • {item['name']}: {amount}g → {item.get('kcal', '?')} kcal")

        lines.append(f"\n📊 **Gesamt:**")
        lines.append(f"  🔥 {total.get('kcal', 0):.0f} kcal")
        lines.append(f"  🥩 {total.get('protein_g', 0):.0f}g Protein")
        lines.append(f"  🍞 {total.get('carbs_g', 0):.0f}g Kohlenhydrate")
        lines.append(f"  🧈 {total.get('fat_g', 0):.0f}g Fett")
        lines.append(f"  🌿 {total.get('fiber_g', 0):.0f}g Ballaststoffe")
        lines.append(f"\n{conf_emoji} Konfidenz: {confidence}")

        notes = meal_data.get("notes")
        if notes:
            lines.append(f"💡 {notes}")


        return "\n".join(lines)

    def _daily_summary(self, telegram_id: int) -> str:
        """Generate daily nutrition summary."""
        totals = self.db.get_daily_totals(telegram_id)
        meals = self.db.get_daily_meals(telegram_id)

        if not meals:
            return "📭 Noch keine Mahlzeiten heute geloggt."

        lines = [f"📊 **Tagesübersicht** ({totals.get('meal_count', 0)} Mahlzeiten)\n"]
        lines.append(f"🔥 {totals.get('total_kcal', 0):.0f} kcal")
        lines.append(f"🥩 {totals.get('total_protein', 0):.0f}g Protein")
        lines.append(f"🍞 {totals.get('total_carbs', 0):.0f}g Kohlenhydrate")
        lines.append(f"🧈 {totals.get('total_fat', 0):.0f}g Fett")
        lines.append(f"🌿 {totals.get('total_fiber', 0):.0f}g Ballaststoffe")

        return "\n".join(lines)

    def _usage_summary(self, telegram_id: int) -> str:
        """Generate token/energy usage summary."""
        stats = self.db.get_usage_summary(telegram_id)

        if not stats:
            return "📭 Noch keine Nutzungsdaten."

        total_tokens = (stats.get("total_prompt_tokens", 0)
                        + stats.get("total_completion_tokens", 0))

        lines = [
            "📈 **Dein Verbrauch**\n",
            f"🔢 {total_tokens:,} Tokens gesamt",
            f"  ↳ {stats.get('total_prompt_tokens', 0):,} Input",
            f"  ↳ {stats.get('total_completion_tokens', 0):,} Output",
            f"📞 {stats.get('call_count', 0)} API-Calls\n",
            f"⚡ ~{stats.get('total_energy_kwh', 0) * 1000:.1f} Wh Strom",
            f"💰 ~{stats.get('total_cost_eur', 0):.4f} EUR Stromkosten\n",
            f"☁️ Das hätte als Cloud-API ~${stats.get('total_api_equiv_usd', 0):.4f} gekostet",
            f"   (Basis: GPT-4o-mini Preise)",
        ]
        return "\n".join(lines)

    def _products_summary(self, telegram_id: int) -> str:
        """List known products."""
        products = self.db.list_products()
        if not products:
            return "📭 Noch keine Produkte in der Datenbank. Scanne ein Etikett!"

        lines = ["🧀 **Deine Produkt-Datenbank**\n"]
        for p in products[:20]:  # Max 20
            per100 = p.get("per_100g", {})
            lines.append(
                f"  • {p['name']}: "
                f"{per100.get('kcal', '?')} kcal/100g, "
                f"{per100.get('protein_g', '?')}g Protein"
            )

        lines.append(f"\n_{len(products)} Produkte gesamt_")
        return "\n".join(lines)
