from typing import TypedDict, Literal,List,Optional
from langgraph.graph import START,END,StateGraph
from backend.app.core.logger import logger
from backend.app.agents.BaseAgent import BaseAgent
from backend.app.agents.GuardrailAgent import GuardrailAgent
from backend.app.agents.CriticAgent import CriticAgent
from backend.app.agents.hitl import (
    MathFeedbackRefiner,
    HumanFeedbackModule,
    feedback_examples,
    add_feedback_record,
    maybe_compile_refiner,
    save_feedback_examples,
)
from backend.app.agents.ProfessorAgent import ProfessorAgent
from backend.app.tools.RetrieverTool import QdrantClientManager
from backend.app.state import tasks, hitl_events
import dspy 
import asyncio
from backend.app.core.registry import get_professor
from backend.app.Memory.custom_memory import SummarizedMemory

class OrchestrationState(TypedDict):
    query: str
    task_id: Optional[str]
    guardrail_result: dict
    professor_response: str
    critic_response: str
    human_feedback: str
    refinement_feedback: str
    tool_usage: List[str]
    context: str
    iterations: int

def guardrail_node(State:OrchestrationState)->OrchestrationState:
    guardrail_agent = GuardrailAgent()
    result = guardrail_agent.call_llm(State['query'])
    State['guardrail_result'] = result
    return {"guardrail_result": result}

async def professor_node(state: OrchestrationState) -> OrchestrationState:
    # Use the shared Professor instance from the registry to avoid re-initializing
    # heavy resources on every orchestration step.
    professor = get_professor()

    # Initialize MCP tools asynchronously if they were deferred during construction
    await professor._initialize_mcp_tools_async()

    query = state["query"]
    task_id = state.get("task_id")
    guardrail_result = state["guardrail_result"]
    previous_solution = state.get("professor_response")
    feedback = state.get("refinement_feedback")

    # Pass the per-call memory (created in run()) into the professor so each run is isolated
    response, tool_used = await professor.call_llm(
        user_input=query,
        guardrail_result=guardrail_result,
        previous_solution=previous_solution,
        feedback=feedback,
        memory=state.get("per_call_memory")
    )

    # Use tool_used directly from professor.call_llm() which properly tracks all tools
    # (KB retrievers, web search tools: search, extract, crawl, analyze_content, math_solver, LLM, etc.)

    # Prefer per-call memory for building history/context; fallback to professor.memory
    mem = state.get("per_call_memory") or professor.memory
    history = "\n".join([
        f"Q: {r}\nA: {c}" for r, c in mem.get_tuple_messages_without_summary()[-3:]
    ])
    context = f"History: {history}\nLast Response: {response}"
    
    # FIX: Update task dict with refined response so frontend sees the updated answer
    if task_id and task_id in tasks:
        tasks[task_id].update({
            "status": "processing",  # Reset from needs_feedback to processing
            "professor_response": response,
            "answer": response,  # Update the answer field that frontend displays
            "iterations": state["iterations"] + 1
        })
        logger.info(f"Updated task {task_id} with refined response (iteration {state['iterations'] + 1})")

    return {
        "professor_response": response,
        "tool_usage": tool_used,
        "context": context,
        "iterations": state["iterations"] + 1
    }

async def critic_human_node(state: OrchestrationState) -> OrchestrationState:
    """
    Async node: Critic evaluates → if refine → pause and wait for human feedback via API.
    """
    critic = CriticAgent()
    query = state["query"]
    task_id = state.get("task_id")
    prof_resp = state["professor_response"]

    # Run Critic
    crit = critic.critique(query, prof_resp)
    decision = crit["decision"].strip().lower()
    feedback = crit["feedback"]

    logger.info(f"Critic Decision: {decision}")

    human_fb = "approve"
    refine_fb = None

    if "refine" in decision:
        if not task_id:
            logger.error("task_id missing in state during HITL")
            return state

        # === 1. Signal frontend: needs feedback ===
        if task_id not in tasks:
            tasks[task_id] = {}
        tasks[task_id].update({
            "status": "needs_feedback",
            "professor_response": prof_resp,
            "answer": prof_resp,
            "critic_feedback": feedback,
            "query": query
        })

        # Create/prepare event: use it ONLY as the "feedback received" signal
        # Frontend learns about needs_feedback by polling /status, so don't pre-set the event.
        if task_id not in hitl_events:
            hitl_events[task_id] = asyncio.Event()
        else:
            # Clear any stale signal from a previous iteration before waiting
            hitl_events[task_id].clear()

        print("\n" + "="*60)
        print(f"HUMAN IN THE LOOP - FEEDBACK REQUIRED (Iteration {state['iterations']})")
        print("="*60)
        print(f"Query: {query}")
        print(f"Professor:\n{prof_resp}")
        print(f"Critic:\n{feedback}")
        print("="*60)

        # === 2. Wait for human feedback via API ===
        raw_feedback = ""
        try:
            await asyncio.wait_for(hitl_events[task_id].wait(), timeout=300)  # 5 min
        except asyncio.TimeoutError:
            logger.warning("HITL feedback timeout for task_id: %s", task_id)
            human_fb = "approve"
        else:
            raw_feedback = tasks[task_id].get("human_feedback", "").strip()
            # Default to approve if empty feedback submitted
            human_fb = (raw_feedback.lower() or "approve")

        # === 3. Save feedback for DSPy training ===
        if human_fb != "approve":
            add_feedback_record({
                "initial_response": prof_resp,
                "human_feedback": raw_feedback,
                "critic_feedback": feedback,
                "query": query,
                "context": state.get("context", "")
            })
            logger.info(f"Stored feedback example #{len(feedback_examples)}")
            
            # PERFORMANCE FIX: Compile refiner in background to avoid blocking workflow
            # BootstrapFewShot can take 30-180 seconds; defer to async task
            asyncio.create_task(_async_compile_refiner())
            
            # Persist to disk for durability
            save_feedback_examples()
        else:
            logger.info("Human approved the response.")

        # === 4. Build refinement input (only when there's actionable human feedback) ===
        if human_fb != "approve":
            refine_fb = f"Critic Feedback: {feedback}\nHuman Feedback: {raw_feedback}"
        else:
            refine_fb = None

        # Reset event for next loop
        hitl_events[task_id].clear()

    return {
        "critic_response": feedback,
        "human_feedback": human_fb,
        "refinement_feedback": refine_fb
    }
# === Routing ===
def route_guardrail(state: OrchestrationState):
    """Route to professor if allowed, otherwise end workflow."""
    return "professor" if state["guardrail_result"].get("status") == "allowed" else END

def route_critic(state: OrchestrationState):
    """Route back to professor for refinement or end workflow.
    
    Flow:
    - Iteration 0 (initial): If critic says refine → ask human → refine
    - Iteration 1 (first refinement): Critic checks again. If refine → ask human → refine
    - Iteration 2 (second refinement): END (no more loops, human's second feedback is final)
    
    This allows 2 refinements max while ensuring:
    - Human sees critic feedback on initial response
    - Critic validates the first refinement
    - Human's second feedback always gets processed (no critic check after iteration 2)
    """
    if (state["refinement_feedback"] and
        state["human_feedback"].lower() != "approve" and
        state["iterations"] <= 2):  # Allow refinement at iterations 0 and 1
        return "professor"
    return END

async def _async_compile_refiner():
    """Background task to compile refiner without blocking orchestration workflow."""
    try:
        logger.info("Background: Starting refiner compilation...")
        # Run CPU-bound compile in thread pool to avoid blocking event loop
        await asyncio.to_thread(maybe_compile_refiner, min_examples=5)
        logger.info("Background: Refiner compilation complete")
    except Exception as e:
        logger.error(f"Background refiner compilation failed: {e}")

workflow = StateGraph(OrchestrationState)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("professor", professor_node)
workflow.add_node("critic_human", critic_human_node)

workflow.add_edge(START, "guardrail")
workflow.add_conditional_edges("guardrail", route_guardrail, {"professor": "professor", END: END})
workflow.add_edge("professor", "critic_human")
workflow.add_conditional_edges("critic_human", route_critic, {"professor": "professor", END: END})

app = workflow.compile()

async def run(query: str, task_id: str) -> dict:
    """
    Entry point from api.py
    """
    logger.info(f"ORCHESTRATION START: '{query}' (task_id: {task_id})")
    professor = get_professor()
    await professor._initialize_mcp_tools_async()
    per_call_summary = SummarizedMemory(llm = professor.llm,max_messages=10)
    try:
        result = await app.ainvoke({
            "query": query,
            "task_id": task_id,
            "guardrail_result": {},
            "professor_response": "",
            "critic_response": "",
            "human_feedback": "",
            "refinement_feedback": "",
            "tool_usage": [],
            "context": "",
            "iterations": 0,
            "per_call_memory" : per_call_summary
        })

        print("\n" + "="*60)
        print("FINAL OUTPUT")
        print("="*60)
        print(f"Query: {query}")
        print(f"Answer:\n{result['professor_response']}")
        print(f"Tools: {', '.join(result['tool_usage']) or 'None'}")
        print(f"Iterations: {result['iterations']}")
        print(f"Feedback Collected: {len(feedback_examples)}")
        print("="*60)
        return result
    except Exception as e:
        logger.exception("Orchestration failed")
        raise
    finally:
        # Keep Qdrant clients alive across orchestration runs. They will be
        # cleaned up on application shutdown by the registry close handler.
        logger.debug("Orchestration run complete; leaving Qdrant clients open for reuse.")

if __name__ == "__main__":
    
    import uuid
    test_query = "A store sells apples for $2 each and oranges for $3 each. You buy 5 fruits and spend $13. How many of each did you buy? Solve using substitution method only."
    test_task_id = str(uuid.uuid4())
    asyncio.run(run(test_query, test_task_id))