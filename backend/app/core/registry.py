from typing import Optional
import asyncio
from backend.app.agents.ProfessorAgent import ProfessorAgent

_professor: Optional[ProfessorAgent] = None
_professor_lock = asyncio.Lock()

async def init_professor_once(model: str | None = None, mcp_server_path: str | None = None, warm_mcp: bool = True) -> ProfessorAgent:
    """
    Initialize a single ProfessorAgent and warm its MCP tools (if any).
    Safe to call multiple times; uses an async lock to avoid races.
    """
    global _professor
    async with _professor_lock:
        if _professor is None:
            # Construct the professor once. Keep args optional to use defaults.
            _professor = ProfessorAgent(model=model or "llama-3.3-70b-versatile", mcp_server_path=mcp_server_path)
            if warm_mcp and getattr(_professor, "_mcp_needs_init", False):
                try:
                    await _professor._initialize_mcp_tools_async()
                except Exception:
                    pass
        return _professor

def get_professor() -> ProfessorAgent:
    """
    Return the shared Professor instance. 
    """
    global _professor
    if _professor is None:
        raise RuntimeError("Professor not initialized. Call init_professor_once() at startup.")
    return _professor

async def close_professor():
    """
    Clean up resources owned by Professor (Qdrant clients, MCP client).
    """
    global _professor
    if _professor is None:
        return
    try:
        # Try to gracefully close MCP client if present
        if getattr(_professor, "mcp_client", None):
            try:
                await _professor.mcp_client.close()
            except Exception:
                pass
        try:
            _professor.cleanup()
        except Exception:
            pass
        # Also close any shared Qdrant clients managed by the RetrieverTool
        try:
            # Import here to avoid top-level circular imports
            from backend.app.tools.RetrieverTool import QdrantClientManager
            QdrantClientManager.close_all()
        except Exception:
            # Don't fail close_professor if Qdrant cleanup errors
            pass
    finally:
        _professor = None