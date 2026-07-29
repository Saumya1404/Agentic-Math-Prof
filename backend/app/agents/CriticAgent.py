import json
import re
from backend.app.agents.BaseAgent import BaseAgent
from backend.app.core.logger import logger


class CriticAgent(BaseAgent):
    """ 
    An agent to critique and provide feedback on the solution provided by the Professor Agent.
    Accepts solution and returns a json object with "Decision" and "Feedback" fields.
    """
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
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
            "Feedback": "One short, actionable sentence.",
            "Severity": 1-5,
            "Scores": {
                "Correctness": 0-10,
                "Completeness": 0-10,
                "Clarity": 0-10
            }
        }

        SEVERITY (only used when Decision is "Refine"):
        1 - Minor: formatting, missing step label, slightly unclear wording
        2 - Moderate: unclear explanation, skips a trivial step
        3 - Notable: missing an intermediate step, insufficient justification
        4 - Major: incorrect formula, wrong calculation, key detail ignored
        5 - Critical: completely wrong answer, hallucinated method, dangerous advice

        SCORES (0-10 per axis):
        - Correctness: Is the math right? Relative to the problem statement.
        - Completeness: Are all parts of the problem addressed? No gaps?
        - Clarity: Is it well-structured, step-by-step, and easy to follow?

        EXAMPLES:
        - Missing power rule → "Refine", Severity 3, Scores: {8, 5, 6}, "Feedback": "The power rule was not explicitly stated."
        - Stop time ignored → "Refine", Severity 4, Scores: {4, 3, 7}, "Feedback": "The 20-minute stop was not included in total time."
        - Perfect → "Accept", Severity 1, Scores: {10, 10, 10}, "Feedback": "All steps and details are correctly addressed."
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

            # === EXTRACT JSON (strip markdown fences first) ===
            cleaned = re.sub(r'```json\s*', '', raw)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                    decision = data.get("Decision") or data.get("decision")
                    feedback = data.get("Feedback") or data.get("feedback")
                    severity = data.get("Severity") or data.get("severity")
                    scores = data.get("Scores") or data.get("scores") or {}
                    if decision and feedback:
                        out = {
                            "decision": decision.strip(),
                            "feedback": feedback.strip(),
                            "severity": max(1, min(5, int(severity or 3))),
                            "scores": {
                                "Correctness": max(0, min(10, int(scores.get("Correctness", 0)))),
                                "Completeness": max(0, min(10, int(scores.get("Completeness", 0)))),
                                "Clarity": max(0, min(10, int(scores.get("Clarity", 0)))),
                            }
                        }
                        logger.info(f"Critic Decision: {out['decision']} (severity={out['severity']})")
                        return out
                except Exception:
                    pass

            # === FALLBACK: Keyword detection ===
            raw_lower = raw.lower()
            if any(word in raw_lower for word in ["refine", "missing", "not shown", "error", "incorrect"]):
                return {
                    "decision": "Refine",
                    "feedback": "The solution is incomplete or unclear. Please show all steps explicitly.",
                    "severity": 3,
                    "scores": {"Correctness": 0, "Completeness": 0, "Clarity": 0}
                }
            else:
                return {
                    "decision": "Accept",
                    "feedback": "The solution appears correct and complete.",
                    "severity": 1,
                    "scores": {"Correctness": 10, "Completeness": 10, "Clarity": 10}
                }

        except Exception as e:
            logger.error(f"Critic error: {e}")
            return {
                "decision": "Refine",
                "feedback": f"Critic failed to evaluate: {str(e)}",
                "severity": 5,
                "scores": {"Correctness": 0, "Completeness": 0, "Clarity": 0}
            }


