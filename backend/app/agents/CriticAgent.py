import json
from backend.app.agents.BaseAgent import BaseAgent
from backend.app.core.logger import logger


class CriticAgent(BaseAgent):
    """ 
    An agent to critique and provide feedback on the solution provided by the Professor Agent.
    Accepts solution and returns a json object with "Decision" and "Feedback" fields.
    """
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        system_prompt = """
        You are an EXTREMELY STRICT Math Critic Agent. Your job is to reject ANY solution that is not 100% complete, explicit, and faithful to the problem.

        CRITICAL RULES — VIOLATE ANY → "Refine":

        1. **EVERY mathematical rule must be EXPLICITLY STATED**:
        - Power rule → MUST say: "Using the power rule: ∫x^n dx = x^(n+1)/(n+1) + C"
        - Substitution → MUST show: "Let u = ..., du = ... dx"
        - Quadratic formula → MUST write the full formula

        2. **EVERY detail in the problem must be USED**:
        - "stops for 20 minutes" → MUST add 20 minutes to final time
        - "leaves at 2:00 PM" → final answer must include correct clock time
        - "using substitution" → MUST use substitution

        3. **NO ASSUMPTIONS ALLOWED**:
        - Do NOT assume units, speeds, or conditions
        - If problem says "continues at same speed", MUST recalculate time

        4. **STEP-BY-STEP MEANS EVERY STEP**:
        - No skipping: n → n+1 → divide → + C
        - No "obvious" steps

        5. **ONLY "Accept" IF PERFECT**:
        - All steps shown
        - All problem parts used
        - No shortcuts
        - Final answer matches all conditions

        OUTPUT MUST BE VALID JSON:
        {
            "Decision": "Accept" or "Refine",
            "Feedback": "One short, actionable sentence. Example: 'The 20-minute stop was not added to arrival time.'"
        }

        EXAMPLES:
        - Missing power rule → "Refine", "Feedback": "The power rule was not explicitly stated."
        - Stop time ignored → "Refine", "Feedback": "The 20-minute stop was not included in total time."
        - Perfect → "Accept", "Feedback": "All steps and details are correctly addressed."
        """

        super().__init__(model=model, system_prompt=system_prompt)
        self.memory = None  
        logger.info("Critic Agent initialized.")

    def critique(self, problem: str, solution: str):
        logger.debug("Critic Agent received problem and solution for evaluation.")
        user_input = f"Problem: {problem}\n\nProposed Solution: {solution}"

        messages = [("system", self.system_prompt), ("user", user_input)]
        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip()
            logger.info(f"Critic raw response: {raw}")

            # === FORCE JSON ===
            import json
            import re

            # Try to extract JSON block
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    decision = data.get("Decision") or data.get("decision")
                    feedback = data.get("Feedback") or data.get("feedback")
                    if decision and feedback:
                        logger.info(f"Critic Decision: {decision}")
                        return {"decision": decision.strip(), "feedback": feedback.strip()}
                except:
                    pass

            # === FALLBACK: Keyword detection ===
            raw_lower = raw.lower()
            if any(word in raw_lower for word in ["refine", "missing", "not shown", "error", "incorrect"]):
                return {
                    "decision": "Refine",
                    "feedback": "The solution is incomplete or unclear. Please show all steps explicitly."
                }
            else:
                return {
                    "decision": "Accept",
                    "feedback": "The solution appears correct and complete."
                }

        except Exception as e:
            logger.error(f"Critic error: {e}")
            return {
                "decision": "Refine",
                "feedback": f"Critic failed to evaluate: {str(e)}"
            }


