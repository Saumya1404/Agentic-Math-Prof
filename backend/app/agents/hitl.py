import dspy
import os
from dspy import Signature, InputField, OutputField


groq_lm = dspy.LM(
    model="groq/llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)
dspy.settings.configure(lm=groq_lm)


feedback_examples = [] 

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