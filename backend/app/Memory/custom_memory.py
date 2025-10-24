from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def to_tuple(self) -> tuple:
        return (self.role, self.content)
    

class ConversationMemory:
    def __init__(self,max_messages: int = 100):
        self.max_messages = max_messages
        self.messages: List[Message] = []

    def add_message(self,role:str,content:str,metadata:Optional[Dict] = None):
        message = Message(role=role,content=content,metadata=metadata or {})
        self.messages.append(message)
        self._enforce_window()

    def _enforce_window(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_exchange(self,user_input:str,assistant_response:str):
        self.add_message(role="user",content=user_input)
        self.add_message(role="assistant",content=assistant_response)

    def get_messages(self)->List[Message]:
        return self.messages
    
    def get_tuple_messages(self)-> List[tuple]:
        return [msg.to_tuple() for msg in self.messages]
    
    def clear(self):
        self.messages = []

    def to_json(self)-> str:
        return json.dumps([msg.to_dict() for msg in self.messages],indent=2)
    
    def from_json(self,json_str:str):
        data = json.load(json_str)
        self.messages = [
            Message(
                role=item['role'],
                content=item['content'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                metadata=item.get('metadata', {})
            )
            for item in data
        ]
    

class SummarizedMemory(ConversationMemory):
    def __init__(self,max_messages:int = 10,llm=None,summary_message_role: str = "system"):
        super().__init__(max_messages=max_messages)
        self.llm = llm
        self.summary_message_role = summary_message_role

    @property
    def current_summary(self) -> str:
        if (self.messages and self.messages[0].metadata.get("summary")):
            return self.messages[0].content
        return ""

    def _enforce_window(self):
        if len(self.messages) > self.max_messages and self.llm:
            recent_count = self.max_messages //2
            if recent_count>len(self.messages):
                return
            
            split_idx = len(self.messages) - recent_count
            old_messages = self.messages[:split_idx]
            recent_messages = self.messages[split_idx:]
            new_summary = self._generate_summary(old_messages)
            if not new_summary:
                self.messages = recent_messages
                return
            summary_msg = Message(
                role=self.summary_message_role,
                content="[Previous conversation_summary]:"+new_summary,
                metadata={"summary": True}
            )
            self.messages = [summary_msg] + recent_messages

    def _generate_summary(self):
        if not self.llm or not self.messages:
            return ""
        
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in self.messages
        ])
        
        prompt = f"""Summarize the following conversation concisely, focusing on key mathematical concepts, problems discussed, and solutions provided:

                {conversation_text}

                Summary:"""
        
        try:
            response = self.llm.invoke([("user", prompt)])
            return response.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."
        
        
    def get_tuple_messages(self) -> List[tuple]:
        return [msg.to_tuple() for msg in self.messages]
    
    def get_tuple_messages_without_summary(self) -> List[Message]:
        return [msg.to_tuple() for msg in self.messages if not msg.metadata.get("summary", False)]
    
    def get_summary(self) -> str:
        return self.summary 
    
    def clear(self):
        super().clear()
        self.summary = ""
