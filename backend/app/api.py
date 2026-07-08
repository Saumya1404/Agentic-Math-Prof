from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uuid
from backend.app.orchestration import run as run_orchestration
from backend.app.agents.hitl import feedback_examples
import logging
from backend.app.state import tasks, hitl_events
from backend.app.core.registry import init_professor_once,close_professor,get_professor

logger = logging.getLogger(__name__)
app = FastAPI(title="Math Professor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class SolveRequest(BaseModel):
    query: str

class SolveResponse(BaseModel):
    task_id:str
    status:str
    answer:Optional[str]=None
    tools: Optional[List[str]]=None
    iterations: Optional[int]=None
    feedback_collected: Optional[int]=None
    critic_feedback: Optional[str]=None

class FeedbackRequest(BaseModel):
    task_id: str
    status: str
    feedback: str

async def run_in_background(task_id:str, query:str):
    try:
        tasks[task_id]["status"] = "processing"
        result = await run_orchestration(query,task_id)

        tasks[task_id].update({
            "status": "completed",
            "answer": result["professor_response"],
            "tools": result["tool_usage"],
            "iterations": result["iterations"],
            "feedback_collected": len(feedback_examples),
            "critic_feedback": result.get("critic_response"),
        })
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = str(e)
        logger.exception("Orchestration failed")

        if task_id in hitl_events and tasks[task_id].get("status") == "needs_feedback":
            hitl_events[task_id].set()
    

@app.post("/solve", response_model=SolveResponse)
async def solve_math(request: SolveRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "query": request.query
    }
    hitl_events[task_id] = asyncio.Event()

    asyncio.create_task(run_in_background(task_id, request.query))
    return SolveResponse(task_id=task_id, status="processing")


@app.get("/status/{task_id}", response_model=SolveResponse)
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")
    
    task = tasks[task_id]
    status = task["status"]

    if status == "completed":
        return SolveResponse(
            task_id=task_id,
            status="completed",
            answer=task.get("answer"),
            tools=task.get("tools"),
            iterations=task.get("iterations"),
            feedback_collected=task.get("feedback_collected"),
            critic_feedback=task.get("critic_feedback"),
        )
    elif status == "needs_feedback":
        return SolveResponse(
            task_id=task_id,
            status="needs_feedback",
            answer=task.get("professor_response"),
            critic_feedback=task.get("critic_feedback"),
        )
    elif status == "processing":
        # Return current answer during refinement so frontend can show updated response
        return SolveResponse(
            task_id=task_id,
            status="processing",
            answer=task.get("answer") or task.get("professor_response"),
            iterations=task.get("iterations"),
        )
    elif status == "error":
        return SolveResponse(
            task_id=task_id,
            status="error",
            answer=f"Error: {task.get('error', 'Unknown error occurred')}"
        )
    else:
        return SolveResponse(task_id=task_id, status=status)

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    if request.task_id not in tasks:
        raise HTTPException(404, "Task not found")
    
    task = tasks[request.task_id]
    task["human_feedback"] = request.feedback
    hitl_events[request.task_id].set()  
    return {"status": "feedback_received"}

@app.on_event("startup")
async def startup_event():
    await init_professor_once()
    

@app.on_event("shutdown")
async def shutdown_event():
    await close_professor()
    