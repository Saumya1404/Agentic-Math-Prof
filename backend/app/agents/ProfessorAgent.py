from backend.app.agents.BaseAgent import BaseAgent
from backend.app.core.logger import logger
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from sympy import sympify, solve
from backend.app.Memory.custom_memory import SummarizedMemory
from backend.app.tools.RetrieverTool import QdrantRetrieverTool
import asyncio
import json
from dotenv import load_dotenv
from pathlib import Path
from backend.app.agents.hitl import MathFeedbackRefiner

# --- Load root .env ---
root_env = Path(__file__).resolve().parents[3] / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env, override=True)
    logger.info(f"Loaded root .env from: {root_env}")
else:
    logger.warning(f"Root .env not found at: {root_env}")
    alt_root_env = Path(__file__).resolve().parents[4] / ".env"
    if alt_root_env.exists():
        load_dotenv(dotenv_path=alt_root_env, override=True)
        logger.info(f"Loaded root .env from alternative path: {alt_root_env}")

# --- Check API keys ---
required_keys = ["FIRECRAWL_API_KEY", "GROQ_API_KEY"]
for key in required_keys:
    value = os.getenv(key)
    if value is None:
        logger.warning(f"{key} is not set in environment!")
    else:
        logger.info(f"{key} is loaded (length: {len(value)})")

class ProfessorAgent(BaseAgent):
    """
    A specialized agent that solves math queries, provides simplified explanations,
    and incorporates human feedback via DSPy.
    """
    def __init__(self, model: str = "llama-3.3-70b-versatile", mcp_server_path: str = None):
        system_prompt = """
        You are a math professor. Solve the query step-by-step using mathematical reasoning.
        Then provide a simplified explanation suitable for a high school student.

        Process:
        1. ALWAYS check the knowledge base (KB) using 'GSM8K_Retriever' and 'Orca200k_Retriever' for similar problems or solutions.
        2. If KB has relevant information OR if the query requires current/real-time data (like stock prices, recent research, etc.), use WebSearch MCP tools ('search', 'extract', 'crawl', 'scrape') for reliable math resources.
           - For 'search': Use for finding general mathematical information
           - For 'extract': Use for extracting specific information from web pages
           - For 'crawl': Use for exploring educational websites
           - For 'scrape': Use for getting content from specific math/finance sites
           - Always specify in your response when you use web search results
        3. If neither KB nor web search provides sufficient information, respond:
           "Information not available in the knowledge bases or reliable online sources. I cannot provide an accurate solution without verified data."
        4. For calculations, use 'math_solver' as needed.

        Output format (only if grounded info is available):
        Step-by-Step: [Detailed solution with equations, citing KB or web sources]
        Simplified: [Easy-to-understand explanation]
        
        IMPORTANT: Always mention which tools you used (KB, web search, math solver) in your response.
        """
        super().__init__(model=model, system_prompt=system_prompt)
        self.memory = SummarizedMemory(llm=self.llm, max_messages=10)
        self.feedback_refiner = MathFeedbackRefiner()  # DSPy HITL module
        logger.info("Professor Agent initialized with summarized memory and HITL.")

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
        possible_paths = [
            Path(self.mcp_server_path) if self.mcp_server_path else None,
            current_file.parents[3] / "mcp_servers" / "websearch" / "main.py",
            current_file.parents[3] / "mcp_servers" / "websearch" / "webtools.py",
            current_file.parents[3] / "mcp_servers" / "websearch" / "__init__.py",
            current_file.parents[2] / "mcp_servers" / "websearch" / "main.py",
            current_file.parents[2] / "mcp_servers" / "websearch" / "webtools.py",
        ]
        for path in possible_paths:
            if path and path.exists():
                logger.info(f"Found MCP server at: {path}")
                return path
        logger.error("MCP server not found in any expected location")
        logger.info("Searched locations:")
        for path in possible_paths:
            if path:
                logger.info(f"  - {path} (exists: {path.exists()})")
        logger.info(f"Current file: {current_file}")
        logger.info(f"Parent[0] (agents): {current_file.parent}")
        logger.info(f"Parent[1] (app): {current_file.parents[1]}")
        logger.info(f"Parent[2] (backend): {current_file.parents[2]}")
        logger.info(f"Parent[3] (MATH_PROF): {current_file.parents[3]}")
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
        mcp_env_path = mcp_server_path.parent / ".env"
        if mcp_env_path.exists():
            load_dotenv(dotenv_path=mcp_env_path, override=False)
            logger.info(f"Loaded MCP .env from: {mcp_env_path}")
        else:
            logger.info(f"No .env file at: {mcp_env_path} (using root .env only)")

    def _initialize_mcp(self):
        """Initialize MCP client and tools"""
        try:
            mcp_path = self._find_mcp_server_path()
            if not mcp_path:
                logger.error("Cannot initialize MCP - server path not found")
                return
            self._load_mcp_env(mcp_path)
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
                        "PATH": os.getenv("PATH", ""),
                        "PYTHONPATH": os.getenv("PYTHONPATH", ""),
                    }
                }
            }
            logger.info(f"Initializing MCP client with server at: {mcp_path}")
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            
            # Handle event loop properly
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                logger.info("Using existing event loop for MCP initialization")
                # If we're in an async context, we need to defer the MCP initialization
                # Store the MCP client for later async initialization
                self._mcp_needs_init = True
                logger.info("MCP client created, will initialize tools asynchronously")
            except RuntimeError:
                # No event loop running, create a new one
                logger.info("No event loop running, creating new one for MCP initialization")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    logger.info("Fetching MCP tools...")
                    mcp_tools_list = loop.run_until_complete(self.mcp_client.get_tools())
                    if not mcp_tools_list:
                        logger.warning("No MCP tools returned")
                        return
                    for tool in mcp_tools_list:
                        self.mcp_tools[tool.name] = tool
                        self.tools[tool.name] = tool
                        logger.info(f"Loaded MCP tool: {tool.name}")
                    logger.info(f"MCP initialized successfully with {len(self.mcp_tools)} tools")
                except asyncio.TimeoutError:
                    logger.error("Timeout while fetching MCP tools")
                except ConnectionError as e:
                    logger.error(f"Connection error with MCP server: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error fetching MCP tools: {e}")
                    logger.exception(e)
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            logger.exception(e)

    async def _initialize_mcp_tools_async(self):
        """Initialize MCP tools asynchronously when in an async context"""
        if not hasattr(self, '_mcp_needs_init') or not self._mcp_needs_init:
            return
        
        try:
            logger.info("Initializing MCP tools asynchronously...")
            mcp_tools_list = await self.mcp_client.get_tools()
            if not mcp_tools_list:
                logger.warning("No MCP tools returned")
                return
            for tool in mcp_tools_list:
                self.mcp_tools[tool.name] = tool
                self.tools[tool.name] = tool
                logger.info(f"Loaded MCP tool: {tool.name}")
            logger.info(f"MCP initialized successfully with {len(self.mcp_tools)} tools")
            self._mcp_needs_init = False
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools asynchronously: {e}")
            logger.exception(e)

    def math_solver(self, equation: str) -> str:
        """Solve mathematical equations using SymPy"""
        try:
            expr = sympify(equation)
            solution = solve(expr)
            logger.info(f"Used tool 'math_solver' for equation: {equation}")
            return str(solution)
        except Exception as e:
            logger.error(f"Error in math_solver: {e}")
            return f"Error: {str(e)}"

    async def _call_mcp_tool_async(self, tool_name: str, arguments: dict) -> str:
        """Call MCP tool asynchronously and coerce structured outputs to plain text."""
        try:
            if tool_name not in self.mcp_tools:
                return f"Tool {tool_name} not found"
            tool = self.mcp_tools[tool_name]
            result = await tool.ainvoke(arguments)

            # Normalize structured results to plain text
            if isinstance(result, str):
                return result

            # common pattern: tool returns object with .result (string)
            if hasattr(result, "result") and isinstance(result.result, str):
                return result.result

            # common pattern: SearchData with .web list of results
            if hasattr(result, "web"):
                parts = []
                for item in getattr(result, "web") or []:
                    title = getattr(item, "title", "") or ""
                    snippet = getattr(item, "snippet", "") or getattr(item, "excerpt", "") or ""
                    url = getattr(item, "url", "") or getattr(item, "link", "") or ""
                    line = " | ".join(p for p in (title, snippet, url) if p)
                    if line:
                        parts.append(line)
                if parts:
                    return "\n\n".join(parts)

            # pydantic/BaseModel-like -> dict -> json
            if hasattr(result, "dict"):
                try:
                    return json.dumps(result.dict(), ensure_ascii=False)
                except Exception:
                    pass

            # fallback to string conversion
            return str(result)

        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error: {str(e)}"
 
    def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Legacy sync wrapper for MCP calls (kept for compatibility). Prefer async _call_mcp_tool_async."""
        # If caller is running inside an event loop, don't try to create/drive a new loop here.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # no running loop -> safe to run one
            return asyncio.run(self._call_mcp_tool_async(tool_name, arguments))
        # If there is a running loop, the caller should use the async API instead of this sync wrapper.
        raise RuntimeError("call_mcp_tool() called while an event loop is running; use await _call_mcp_tool_async(...) instead")
 
    async def call_llm(self, user_input: str, guardrail_result: dict, previous_solution: str = None, feedback: str = None) -> tuple[str, list]:
        """Async: process math query with optional refinement. Await this from async callers."""
        if guardrail_result.get("status") != "allowed":
            logger.warning(f"Query blocked by guardrail: {user_input}")
            return "Error: Only math-related queries are allowed.", []

        # Track which tools are used
        tool_used = []

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
                    tool_used.append(kb_tool)
                    logger.info(f"Retrieved relevant info from {kb_tool}")

            # Step 2: Web search fallback
            web_result = ""
            # Check if we need current/real-time information that KB might not have
            needs_web_search = (
                not kb_response or 
                "No relevant info" in kb_response or
                any(keyword in user_input.lower() for keyword in [
                    "current", "latest", "recent", "today", "now", "find", "search", "look up",
                    "online", "web", "internet", "market price", "stock price", "real-time"
                ])
            )
            
            if needs_web_search:
                # Check if MCP tools need initialization
                if hasattr(self, '_mcp_needs_init') and self._mcp_needs_init:
                    logger.warning("MCP tools not initialized yet, skipping web search")
                    kb_response += "\nWeb search: MCP tools not available (initialization pending)"
                else:
                    web_search_tool = None
                    # Look for any available web search tools
                    for tool_name in self.mcp_tools.keys():
                        if any(web_tool in tool_name.lower() for web_tool in ['search', 'extract', 'crawl', 'scrape']):
                            web_search_tool = tool_name
                            break
                    
                    if web_search_tool:
                        logger.info(f"Using MCP tool: {web_search_tool}")
                        
                        # Multi-stage web search strategy: Search → Crawl
                        logger.info(f"Using multi-stage search strategy for math query")
                        
                        # Stage 1: Search for relevant websites
                        search_query = user_input
                        search_args = {"query": search_query}
                        search_result = await self._call_mcp_tool_async('search', search_args)
                        logger.info(f"Search result length: {len(search_result) if search_result else 0}")
                        
                        # Extract URLs from search results for crawling
                        urls_to_crawl = []
                        if search_result and "Error" not in search_result:
                            # Look for URLs in the search results
                            lines = search_result.split('\n')
                            for line in lines:
                                if 'URL:' in line:
                                    url = line.replace('URL:', '').strip()
                                    # Accept any URL from search results
                                    urls_to_crawl.append(url)
                        
                        # Stage 2: Crawl the found URLs
                        if urls_to_crawl:
                            logger.info(f"Found {len(urls_to_crawl)} URLs to crawl")
                            crawl_results = []
                            for url in urls_to_crawl[:3]:  # Limit to top 3 URLs
                                try:
                                    crawl_args = {
                                        "url": url,
                                        "maxDepth": 1,
                                        "limit": 1
                                    }
                                    crawl_result = await self._call_mcp_tool_async('crawl', crawl_args)
                                    if crawl_result and "Error" not in crawl_result:
                                        crawl_results.append(f"URL: {url}\nContent: {crawl_result}")
                                except Exception as e:
                                    logger.warning(f"Failed to crawl {url}: {e}")
                            
                            # Combine search and crawl results
                            web_result = f"Search Results:\n{search_result}\n\nCrawled Websites:\n" + "\n\n".join(crawl_results)
                        else:
                            # Fallback: use regular search if no URLs found
                            logger.info("No URLs found in search, using search results only")
                            web_result = search_result
                        
                        # Mark both search and crawl as used
                        tool_used.extend(['search', 'crawl'])
                        
                        logger.info(f"Web search result length: {len(web_result) if web_result else 0}")
                        logger.info(f"Web search result preview: {web_result[:200] if web_result else 'None'}...")
                        if "Error" not in web_result:
                            tool_used.append(web_search_tool)
                            logger.info(f"Added {web_search_tool} to tools used")
                        else:
                            logger.warning(f"Web search error: {web_result}")
                        kb_response += f"\nWeb search result: {web_result}"
                    else:
                        logger.warning("No web search MCP tool found")
                        kb_response += "\nWeb search: Not available"

            # Step 3: Generate or refine response
            context = f"History: {history}\nTool Results: {kb_response}"
            if previous_solution and feedback:
                # Use MathFeedbackRefiner for refinement
                logger.info("Refining response using MathFeedbackRefiner")
                # Assume feedback is formatted as "Critic Feedback: ... \nHuman Feedback: ..."
                critic_feedback = feedback.split("Human Feedback:")[0].replace("Critic Feedback:", "").strip() if "Critic Feedback:" in feedback else ""
                human_feedback = feedback.split("Human Feedback:")[1].strip() if "Human Feedback:" in feedback else feedback
                refined = self.feedback_refiner(
                    initial_response=previous_solution,
                    human_feedback=human_feedback,
                    critic_feedback=critic_feedback,
                    query=user_input,
                    context=context
                )
                response = refined.refined_response
                tool_used.append("MathFeedbackRefiner")
            else:
                # Initial response
                messages = [
                    ("system", self.system_prompt),
                    ("user", f"History: {history}\nQuery: {user_input}\nTools: {tool_names}\nTool Results: {kb_response}")
                ]
                response = self.llm.invoke(messages).content
                if "Information not available" not in response:
                    tool_used.append("LLM")  # LLM used for final response composition

            # Step 4: No info fallback
            if ("No relevant info" in kb_response and
                ("Web search: Not available" in kb_response or not web_result)):
                response = "Information not available in the knowledge bases or reliable online sources. I cannot provide an accurate solution without verified data."
                tool_used = ["None"]  # Override to indicate no tools provided data

            # Log tools used
            logger.info(f"Tools used for query '{user_input}': {', '.join(tool_used) or 'None'}")

            # Update memory
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response)
            logger.info(f"Professor Agent processed query: {user_input}")
            return response, tool_used

        except Exception as e:
            logger.error(f"Error in Professor Agent: {e}")
            logger.exception(e)
            return "Error processing query.", tool_used

    def cleanup(self):
        """Close Qdrant clients to release file locks"""
        if hasattr(self, 'tools'):
            for name, tool in self.tools.items():
                if hasattr(tool, 'client') and tool.client is not None:
                    try:
                        tool.client.close()
                        logger.info(f"Closed Qdrant client for {name}")
                    except:
                        pass
        if self.mcp_client:
            try:
                asyncio.get_event_loop().run_until_complete(self.mcp_client.close())
            except:
                pass

if __name__ == "__main__":
    from backend.app.agents.GuardrailAgent import GuardrailAgent

    # Test case
    query = "what is the integral of x^2?"
    guardrail = GuardrailAgent()
    professor = ProfessorAgent()

    # Simulate a guardrail check
    guard_result = guardrail.call_llm(query)
    logger.info(f"Guardrail result for query '{query}': {guard_result}")

    # Initial response
    response = professor.call_llm(query, guard_result)
    print(f"\nQuery: {query}")
    print(f"Initial Response: {response}")

    # Simulate refinement (example feedback)
    feedback = "Critic Feedback: The explanation is correct but lacks a clear step for integration by substitution.\nHuman Feedback: Please add a step showing the power rule explicitly."
    refined_response = professor.call_llm(query, guard_result, previous_solution=response, feedback=feedback)
    print(f"\nRefined Response: {refined_response}")

    professor.cleanup()