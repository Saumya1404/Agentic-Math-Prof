from typing import TypedDict, Literal,List,Optional
from langgraph.graph import START,END,StateGraph
from backend.app.core.logger import logger
from backend.app.agents.BaseAgent import BaseAgent
from backend.app.agents.GuardrailAgent import GuardrailAgent
from backend.app.agents.CriticAgent import CriticAgent
from backend.app.agents.hitl import MathFeedbackRefiner,HumanFeedbackModule,feedback_examples
from backend.app.agents.ProfessorAgent import ProfessorAgent
from backend.app.tools.RetrieverTool import QdrantClientManager
from backend.app.state import tasks, hitl_events
import dspy 
import asyncio

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
    professor = ProfessorAgent()
    
    # Initialize MCP tools asynchronously if needed
    await professor._initialize_mcp_tools_async()
    
    query = state["query"]
    guardrail_result = state["guardrail_result"]
    previous_solution = state.get("professor_response")
    feedback = state.get("refinement_feedback")

    response, tool_used = await professor.call_llm(
        user_input=query,
        guardrail_result=guardrail_result,
        previous_solution=previous_solution,
        feedback=feedback
    )

    tool_usage = []
    if "GSM8K_Retriever" in response or "gsm8k" in response.lower():
        tool_usage.append("GSM8K_Retriever")
    if "Orca200k_Retriever" in response or "orca" in response.lower():
        tool_usage.append("Orca200k_Retriever")
    if "math_solver" in response:
        tool_usage.append("math_solver")
    if previous_solution and feedback:
        tool_usage.append("MathFeedbackRefiner")

    history = "\n".join([
        f"Q: {r}\nA: {c}" for r, c in professor.memory.get_tuple_messages_without_summary()[-3:]
    ])
    context = f"History: {history}\nLast Response: {response}"

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
            "critic_feedback": feedback,
            "query": query
        })

        # Create event if not exists
        if task_id not in hitl_events:
            hitl_events[task_id] = asyncio.Event()
        hitl_events[task_id].set()  # Tell frontend to show feedback box

        print("\n" + "="*60)
        print("HUMAN IN THE LOOP - FEEDBACK REQUIRED")
        print("="*60)
        print(f"Query: {query}")
        print(f"Professor:\n{prof_resp}")
        print(f"Critic:\n{feedback}")
        print("="*60)

        # === 2. Wait for human feedback via API ===
        try:
            await asyncio.wait_for(hitl_events[task_id].wait(), timeout=300)  # 5 min
        except asyncio.TimeoutError:
            logger.warning("HITL feedback timeout for task_id: %s", task_id)
            human_fb = "approve"
        else:
            raw_feedback = tasks[task_id].get("human_feedback", "").strip()
            human_fb = raw_feedback.lower()

        # === 3. Save feedback for DSPy training ===
        if human_fb != "approve":
            ex = dspy.Example(
                query=query,
                professor_response=prof_resp,
                ideal_response=raw_feedback
            ).with_inputs("query")
            feedback_examples.append(ex)
            logger.info(f"Stored feedback example #{len(feedback_examples)}")
        else:
            logger.info("Human approved the response.")

        # === 4. Build refinement input ===
        refine_fb = f"Critic Feedback: {feedback}\nHuman Feedback: {raw_feedback}"

        # Reset event for next loop
        hitl_events[task_id].clear()

    return {
        "critic_response": feedback,
        "human_feedback": human_fb,
        "refinement_feedback": refine_fb
    }
# === Routing ===
def route_guardrail(state: OrchestrationState) -> Literal["professor", END]:
    return "professor" if state["guardrail_result"].get("status") == "allowed" else END

def route_critic(state: OrchestrationState) -> Literal["professor", END]:
    if (state["refinement_feedback"] and
        state["human_feedback"].lower() != "approve" and
        state["iterations"] < 3):
        return "professor"
    return END

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
            "iterations": 0
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
        QdrantClientManager.close_all()
        logger.info("Cleaned up all QdrantClient instances")

if __name__ == "__main__":
    
    import uuid
    test_query = "A store sells apples for $2 each and oranges for $3 each. You buy 5 fruits and spend $13. How many of each did you buy? Solve using substitution method only."
    test_task_id = str(uuid.uuid4())
    asyncio.run(run(test_query, test_task_id))