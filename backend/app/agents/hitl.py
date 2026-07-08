import dspy
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dspy import Signature, InputField, OutputField
from backend.app.core.logger import logger
from backend.app.core.feedback_qdrant import add_feedback, get_top_k
try:
    # Prefer explicit teleprompt imports when available
    from dspy.teleprompt import BootstrapFewShot  # type: ignore
except Exception:  # pragma: no cover - fallback for older DSPy
    BootstrapFewShot = getattr(dspy, "BootstrapFewShot", None)


groq_lm = dspy.LM(
    model="groq/llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)
dspy.settings.configure(lm=groq_lm)


# In-memory training set (DSPy Examples) and raw records
feedback_examples: List[dspy.Example] = []
_raw_feedback_records: List[Dict[str, Any]] = []

# Location to persist training records (ignored by VCS per repo hygiene)
FEEDBACK_DIR = Path(__file__).resolve().parents[3] / "Data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_JSONL = FEEDBACK_DIR / "refiner_train.jsonl"

# Compiled (optimized) refiner cache
_compiled_refiner: Optional[dspy.Module] = None

@dspy.Tool
def human_feedback_tool(query: str, professor_response: str, critic_response: str) -> str:
    """
    Tool to collect human feedback on the agent's response.
    Prints the context and waits for user input.
    """
    print("\n--- Human Feedback Requested ---")
    print(f"Original Query: {query}")
    print(f"Professor Response: {professor_response}")
    print(f"Critic Evaluation: {critic_response}")
    feedback = input("Enter your feedback/correction (or 'approve' if good): ")
    return feedback

class HumanFeedbackModule(dspy.Module):
    """
    DSPy Module: Collects real human feedback using the @dspy.Tool.
    """
    def __init__(self):
        super().__init__()
        self.get_feedback = human_feedback_tool  # Real input

    def forward(self, query, professor_response, critic_response):
        return self.get_feedback(
            query=query,
            professor_response=professor_response,
            critic_response=critic_response
        )


class MathFeedbackRefinerSignature(Signature):
    """Refine math solution using human + critic feedback."""
    initial_response: str = InputField(desc="Original ProfessorAgent response")
    human_feedback: str = InputField(desc="Human correction or suggestion")
    critic_feedback: str = InputField(desc="CriticAgent evaluation")
    query: str = InputField(desc="Original math query")
    context: str = InputField(desc="Tool results and memory history")
    refined_response: str = OutputField(desc="Improved step-by-step + simplified explanation")


class MathFeedbackRefiner(dspy.Module):
    """
    DSPy Module: Refines response using real human feedback.
    """
    def __init__(self):
        super().__init__()
        self.refine = dspy.Predict(MathFeedbackRefinerSignature)

    def forward(self, initial_response, human_feedback, critic_feedback, query, context):
        return self.refine(
            initial_response=initial_response,
            human_feedback=human_feedback,
            critic_feedback=critic_feedback,
            query=query,
            context=context
        )


# ----------------------
# Persistence utilities
# ----------------------
def _record_to_example(record: Dict[str, Any]) -> dspy.Example:
    """Convert a raw record into a DSPy Example for the refiner signature."""
    ex = dspy.Example(
        initial_response=record.get("initial_response", ""),
        human_feedback=record.get("human_feedback", ""),
        critic_feedback=record.get("critic_feedback", ""),
        query=record.get("query", ""),
        context=record.get("context", "")
    ).with_inputs("initial_response", "human_feedback", "critic_feedback", "query", "context")
    # Optional supervised target if provided
    if "refined_response" in record and record["refined_response"]:
        setattr(ex, "refined_response", record["refined_response"])  # for supervised metrics
    return ex


def save_feedback_examples() -> None:
    """Persist raw feedback records to JSONL."""
    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_JSONL.open("w", encoding="utf-8") as f:
            for rec in _raw_feedback_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(_raw_feedback_records)} feedback records to {FEEDBACK_JSONL}")
    except Exception as e:
        logger.error(f"Failed to save feedback records: {e}")


def load_feedback_examples() -> None:
    """Load feedback records from disk into memory at import time."""
    if not FEEDBACK_JSONL.exists():
        return
    try:
        _raw_feedback_records.clear()
        feedback_examples.clear()
        with FEEDBACK_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                _raw_feedback_records.append(rec)
                feedback_examples.append(_record_to_example(rec))
        logger.info(f"Loaded {len(_raw_feedback_records)} feedback records from {FEEDBACK_JSONL}")
    except Exception as e:
        logger.error(f"Failed to load feedback records: {e}")


def add_feedback_record(record: Dict[str, Any]) -> None:
    """Append a raw record and matching DSPy Example; persist to Qdrant and in-memory."""
    try:
        # Persist into Qdrant for vector search
        add_feedback(record)
    except Exception:
        logger.exception("Failed to persist feedback to Qdrant")

    _raw_feedback_records.append(record)
    feedback_examples.append(_record_to_example(record))
    # also persist JSONL as a durability fallback
    save_feedback_examples()


# Load any prior training data at module import
load_feedback_examples()


# ----------------------
# Optimization (Teleprompting)
# ----------------------
def _refiner_metric(example: dspy.Example, pred: Dict[str, Any], trace=None) -> float:
    """
    A simple metric for refiner quality.
    Strategy:
    - Reward alignment with human feedback (pred output includes key phrases from human_feedback)
    - Optionally, ask CriticAgent to accept/refine and reward accept = 1.0
    """
    try:
        output = (pred.get("refined_response") or "").lower()
    except Exception:
        output = ""

    # Heuristic: coverage of human feedback tokens
    hf = (getattr(example, "human_feedback", "") or "").lower()
    if not output:
        score = 0.0
    else:
        hf_tokens = [t for t in hf.split() if len(t) > 3]
        covered = sum(1 for t in set(hf_tokens) if t in output)
        score = covered / max(1, len(set(hf_tokens)))

    
    
    # Optional enhancement: If output contains explicit math formulas/steps, boost score
    if any(marker in output for marker in ["step", "using", "formula", "="]):
        score = min(1.0, score + 0.1)

    return float(score)


def maybe_compile_refiner(min_examples: int = 5) -> Optional[dspy.Module]:
    """
    Compile/optimize the refiner using collected feedback examples once enough data exists.
    Uses BootstrapFewShot teleprompting by default.
    """
    global _compiled_refiner

    if BootstrapFewShot is None:
        logger.warning("DSPy teleprompting (BootstrapFewShot) not available; skipping optimization.")
        return None

    if len(feedback_examples) < min_examples:
        logger.info(f"Not enough feedback to compile refiner (have {len(feedback_examples)}/{min_examples})")
        return None

    try:
        program = MathFeedbackRefiner()
        compiler = BootstrapFewShot(
            metric=_refiner_metric,
            max_bootstrapped_demos=min(8, len(feedback_examples)),
            max_labeled_demos=min(12, len(feedback_examples))
        )
        logger.info("Compiling MathFeedbackRefiner with DSPy BootstrapFewShot…")

        # DSPy has had several compile() signatures across versions. Try a few
        # common variants to remain compatible:
        # 1) compile(trainset=..., program=...)
        # 2) compile(program, trainset=...)
        # 3) compile(trainset=..., module=...)
        # 4) compile(trainset)
        compiled = None
        try:
            compiled = compiler.compile(trainset=feedback_examples, program=program)
            logger.info("Used compile(trainset=..., program=...) signature")
        except TypeError as e1:
            logger.debug(f"compile(trainset=..., program=...) failed: {e1}")
            try:
                compiled = compiler.compile(program, trainset=feedback_examples)
                logger.info("Used compile(program, trainset=...) signature")
            except TypeError as e2:
                logger.debug(f"compile(program, trainset=...) failed: {e2}")
                try:
                    compiled = compiler.compile(trainset=feedback_examples, module=program)
                    logger.info("Used compile(trainset=..., module=...) signature")
                except TypeError as e3:
                    logger.debug(f"compile(trainset=..., module=...) failed: {e3}")
                    try:
                        # Some DSPy versions accept only the trainset positional arg
                        compiled = compiler.compile(feedback_examples)
                        logger.info("Used compile(feedback_examples) signature")
                    except Exception as e4:
                        logger.error(f"All compile() signatures failed: {e4}")
                        compiled = None

        if compiled is None:
            logger.error("Refiner compilation failed: no compatible compile() signature found.")
            return None

        _compiled_refiner = compiled
        logger.info("Refiner compiled successfully; optimized module cached.")
        return _compiled_refiner
    except Exception as e:
        logger.error(f"Failed to compile refiner (unexpected error): {e}")
        return None


def get_refiner_module() -> dspy.Module:
    """Return the compiled refiner if available, else the default MathFeedbackRefiner."""
    return _compiled_refiner or MathFeedbackRefiner()