from backend.app.agents.BaseAgent import BaseAgent
from backend.app.core.logger import logger
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from sympy import sympify, solve
from backend.app.Memory.custom_memory import SummarizedMemory
from backend.app.tools.RetrieverTool import QdrantRetrieverTool
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# --- Load root .env (this should have the API keys) ---
# From backend/app/agents/ProfessorAgent.py to MATH_PROF/.env
root_env = Path(__file__).resolve().parents[3] / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=True)
    logger.info(f"Loaded root .env from: {root_env}")
else:
    logger.warning(f" Root .env not found at: {root_env}")
    # Try alternative path
    alt_root_env = Path(__file__).resolve().parents[4] / ".env"
    if alt_root_env.exists():
        load_dotenv(dotenv_path=alt_root_env, override=True)
        logger.info(f" Loaded root .env from alternative path: {alt_root_env}")

# --- Check that keys are loaded ---
required_keys = ["FIRECRAWL_API_KEY", "GROQ_API_KEY"]
for key in required_keys:
    value = os.getenv(key)
    if value is None:
        logger.warning(f" {key} is not set in environment!")
    else:
        logger.info(f"{key} is loaded (length: {len(value)})")


class ProfessorAgent(BaseAgent):
    """
    A specialized agent that solves math queries and provides simplified explanations.
    Uses Qdrant KB, SymPy solver, and MCP web search for external info.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", mcp_server_path: str = None):
        system_prompt = """
        You are a math professor. Solve the query step-by-step using mathematical reasoning.
        Then provide a simplified explanation suitable for a high school student.

        Process:
        1. ALWAYS first check the knowledge base (KB) using the 'query_kb' tool for similar problems or solutions.
        2. If KB has no relevant information, use 'math_web_search' to fetch reliable math resources.
        3. If neither KB nor web search provides sufficient information, respond honestly:
           "Information not available in the knowledge base or reliable online sources. I cannot provide an accurate solution without verified data."
        4. For calculations, use 'math_solver' as needed.
        """
        super().__init__(model=model, system_prompt=system_prompt)

        # --- Summarized memory ---
        self.memory = SummarizedMemory(llm=self.llm, max_messages=10)
        logger.info("Professor Agent initialized with summarized memory.")

        # --- Tools ---
        self.tools = {
            "GSM8K_Retriever": QdrantRetrieverTool(
                name="GSM8K_Retriever",
                description="Retrieves examples from the GSM8K reasoning dataset.",
                collection_name="gsm8k_knowledge_base",
                persist_dir="./Data/knowledge_base/qdrant_db"
            ),
            "Orca200k_Retriever": QdrantRetrieverTool(
                name="Orca200k_Retriever",
                description="Retrieves reasoning traces from the Orca 200k dataset.",
                collection_name="orca_200k_sample",
                persist_dir="./Data/knowledge_base/qdrant_db_orca_sample"
            ),
            "math_solver": self.math_solver,
        }

        # --- MCP Servers ---
        self.mcp_client = None
        self.mcp_tools = {}
        self.mcp_server_path = mcp_server_path
        self._initialize_mcp()

    def _find_mcp_server_path(self):
        """Find the MCP server path - try multiple common locations"""
        current_file = Path(__file__).resolve()
        
        # Based on structure: MATH_PROF/backend/app/agents/ProfessorAgent.py
        # We need to go to: MATH_PROF/mcp_servers/websearch/ (AT ROOT LEVEL)
        
        # Try different possible locations
        possible_paths = [
            # Custom path if provided
            Path(self.mcp_server_path) if self.mcp_server_path else None,
            
            # CORRECT: Root level - MATH_PROF/mcp_servers/websearch/
            # From backend/app/agents/ -> backend/app/ -> backend/ -> MATH_PROF/ -> mcp_servers/
            current_file.parents[3] / "mcp_servers" / "websearch" / "main.py",
            current_file.parents[3] / "mcp_servers" / "websearch" / "webtools.py",
            current_file.parents[3] / "mcp_servers" / "websearch" / "__init__.py",
            
            # Also try backend location in case structure changes
            current_file.parents[2] / "mcp_servers" / "websearch" / "main.py",
            current_file.parents[2] / "mcp_servers" / "websearch" / "webtools.py",
        ]
        
        for path in possible_paths:
            if path and path.exists():
                logger.info(f" Found MCP server at: {path}")
                return path
        
        logger.error("MCP server not found in any expected location")
        logger.info("Searched locations:")
        for path in possible_paths:
            if path:
                logger.info(f"  - {path} (exists: {path.exists()})")
        
        # Additional debug info
        logger.info(f"Current file: {current_file}")
        logger.info(f"Parent[0] (agents): {current_file.parent}")
        logger.info(f"Parent[1] (app): {current_file.parents[1]}")
        logger.info(f"Parent[2] (backend): {current_file.parents[2]}")
        logger.info(f"Parent[3] (MATH_PROF): {current_file.parents[3]}")
        
        # Show what's actually in the expected directory
        expected_mcp_dir = current_file.parents[3] / "mcp_servers" / "websearch"
        if expected_mcp_dir.exists():
            logger.info(f"Contents of {expected_mcp_dir}:")
            for item in expected_mcp_dir.iterdir():
                logger.info(f"  - {item.name}")
        
        return None

    def _load_mcp_env(self, mcp_server_path: Path):
        """Load the MCP server's .env file if it exists"""
        if not mcp_server_path:
            return
            
        # MCP server is at MATH_PROF/mcp_servers/websearch/
        # Look for .env in the websearch directory
        mcp_env_path = mcp_server_path.parent / ".env"
        if mcp_env_path.exists():
            # Don't override root env, but load any additional vars
            load_dotenv(dotenv_path=mcp_env_path, override=False)
            logger.info(f"✓ Loaded MCP .env from: {mcp_env_path}")
        else:
            logger.info(f"ℹNo .env file at: {mcp_env_path} (using root .env only)")

    def _initialize_mcp(self):
        """Initialize MCP client and tools"""
        try:
            # Find MCP server
            mcp_path = self._find_mcp_server_path()
            
            if not mcp_path:
                logger.error("Cannot initialize MCP - server path not found")
                logger.info("Tip: Pass mcp_server_path parameter when creating ProfessorAgent")
                logger.info("   Example: ProfessorAgent(mcp_server_path='/path/to/mcp/server/main.py')")
                return

            # Load MCP server's env file
            self._load_mcp_env(mcp_path)

            # Verify API keys are still available
            firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
            groq_key = os.getenv("GROQ_API_KEY")
            
            if not firecrawl_key or not groq_key:
                logger.error("API keys not found after loading env files")
                return

            mcp_servers = {
                "websearch": {
                    "transport": "stdio",
                    "command": "python",
                    "args": [str(mcp_path)],
                    "env": {
                        "FIRECRAWL_API_KEY": firecrawl_key,
                        "GROQ_API_KEY": groq_key,
                        # Pass through any other env vars the MCP server might need
                        "PATH": os.getenv("PATH", ""),
                        "PYTHONPATH": os.getenv("PYTHONPATH", ""),
                    }
                }
            }

            logger.info(f"Initializing MCP client with server at: {mcp_path}")

            # Initialize MCP client
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            
            # Get MCP tools - use proper async handling
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                logger.info(" Fetching MCP tools...")
                mcp_tools_list = loop.run_until_complete(self.mcp_client.get_tools())
                
                if not mcp_tools_list:
                    logger.warning("⚠️ No MCP tools returned")
                    return
                
                # Add MCP tools to self.tools
                for tool in mcp_tools_list:
                    self.mcp_tools[tool.name] = tool
                    self.tools[tool.name] = tool
                    logger.info(f"  ✓ Loaded MCP tool: {tool.name}")
                
                logger.info(f" MCP initialized successfully with {len(self.mcp_tools)} tools")
                
            except asyncio.TimeoutError:
                logger.error(" Timeout while fetching MCP tools")
            except ConnectionError as e:
                logger.error(f" Connection error with MCP server: {e}")
            except Exception as e:
                logger.error(f" Unexpected error fetching MCP tools: {e}")
                logger.exception(e)
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            logger.exception(e)

    # --- Math solver using SymPy ---
    def math_solver(self, equation: str) -> str:
        """Solve mathematical equations using SymPy"""
        try:
            expr = sympify(equation)
            solution = solve(expr)
            return str(solution)
        except Exception as e:
            logger.error(f"Error in math_solver: {e}")
            return f"Error: {str(e)}"

    async def _call_mcp_tool_async(self, tool_name: str, arguments: dict) -> str:
        """Call MCP tool asynchronously"""
        try:
            if tool_name not in self.mcp_tools:
                return f"Tool {tool_name} not found"
            
            tool = self.mcp_tools[tool_name]
            result = await tool.ainvoke(arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error: {str(e)}"

    def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Synchronous wrapper for MCP tool calls"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._call_mcp_tool_async(tool_name, arguments))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in call_mcp_tool: {e}")
            return f"Error: {str(e)}"

    # --- LLM call ---
    def call_llm(self, user_input: str, guardrail_result: dict) -> str:
        if guardrail_result.get("status") != "allowed":
            logger.warning(f"Query blocked by guardrail: {user_input}")
            return "Error: Only math-related queries are allowed."

        # Build context from memory
        history_entries = self.memory.get_tuple_messages_without_summary()[-5:]
        history = "\n".join([f"Q: {role}\nA: {content}" for role, content in history_entries])
        tool_names = ", ".join(self.tools.keys())

        try:
            # Step 1: KB lookup
            kb_response = ""
            for kb_tool in ["GSM8K_Retriever", "Orca200k_Retriever"]:
                result = self.tools[kb_tool].invoke(user_input)
                if "No relevant info" not in result:
                    kb_response += f"{kb_tool} result: {result}\n"

            # Step 2: Web search fallback using MCP tools
            web_result = ""
            if not kb_response or "No relevant info" in kb_response:
                # Check if web search tool is available
                web_search_tool = None
                for tool_name in self.mcp_tools.keys():
                    if "search" in tool_name.lower():
                        web_search_tool = tool_name
                        break
                
                if web_search_tool:
                    logger.info(f"Using MCP tool: {web_search_tool}")
                    web_result = self.call_mcp_tool(
                        web_search_tool, 
                        {"query": user_input}
                    )
                    kb_response += f"\nWeb search result: {web_result}"
                else:
                    logger.warning("No web search MCP tool found")
                    kb_response += "\nWeb search: Not available"

            # Step 3: Generate response
            messages = [
                ("system", self.system_prompt),
                ("user", f"History: {history}\nQuery: {user_input}\nTools: {tool_names}\nTool Results: {kb_response}")
            ]
            response = self.llm.invoke(messages).content

            # Step 4: No info fallback
            if ("No relevant info" in kb_response and 
                ("Web search: Not available" in kb_response or not web_result)):
                response = "Information not available in the knowledge bases or reliable online sources. I cannot provide an accurate solution without verified data."

            # Update memory
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response) 
            logger.info(f"Professor Agent processed query: {user_input}")
            return response
            
        except Exception as e:
            logger.error(f"Error in Professor Agent: {e}")
            logger.exception(e)
            return "Error processing query."

    def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            logger.info("Cleaning up MCP client resources")
            self.mcp_client = None

# --- Main execution ---
if __name__ == "__main__":
    from backend.app.agents.GuardrailAgent import GuardrailAgent

    guardrail = GuardrailAgent()
    professor = ProfessorAgent()

    query = "what is the integral of x^2?"
    guard_result = guardrail.call_llm(query)
    response = professor.call_llm(query, guard_result)
    print(response)
