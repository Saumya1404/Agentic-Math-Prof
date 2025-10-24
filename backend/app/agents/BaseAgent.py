from backend.app.core.logger import logger
from langchain_groq import ChatGroq
from backend.app.Memory.custom_memory import ConversationMemory,SummarizedMemory
from backend.app.config.settings import settings


class BaseAgent:
    """Base Agent class for all agents."""
    def  __init__(self,model:str = "llama3.3-70b-versatile",system_prompt: str = "",memory = None):
        self.model = model
        self.system_prompt = system_prompt

        self.llm = ChatGroq(model_name=self.model, api_key=settings.GROQ_API_KEY)
        logger.info(f"Initialized BaseAgent with model: {self.model}")

        self.memory = memory if memory else ConversationMemory(
            max_messages=10
        )


    def call_llm(self,user_input:str):
        try:
            logger.debug(f"User Input received: {user_input}")
            chat_history = self.memory.get_tuple_messages()
            messages = [("system",self.system_prompt)] + chat_history + [("user",user_input)]
            response = self.llm.invoke(messages)

            self.memory.add_message("user",user_input)
            self.memory.add_message("assistant",response.content)
            logger.info("Response generated successfully.")
            return response.content
        except Exception as e:
            logger.exception(f"Error in call_llm: {e}")
            return "Error occured while processing your request."
        
            