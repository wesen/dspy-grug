from dotenv import load_dotenv

load_dotenv()

import dspy
import instructor

wrkr = dspy.OpenAI(model="gpt-4o-mini", max_tokens=1000, model_type="chat")
bss = dspy.OpenAI(model="gpt-4o", max_tokens=1000, model_type="chat")

dspy.configure(lm=wrkr)

from typing import List, Any, Callable, Optional
from pydantic import BaseModel


class Plan(dspy.Signature):
    """Produce a step by step plan to perform the task.
    The plan needs to be in markdown format and should be broken down into big steps (with ## headings) and sub-steps beneath those.
    When thinking about your plan, be sure to think about the tools at your disposal and include them in your plan.
    
    Don't use headers for the Task/Reasoning/Proposed Plan delimiters.
    """

    task = dspy.InputField(prefix="Task", desc="The task to perform")
    context = dspy.InputField(
        format=str, desc="Any context that may be relevant to the task"
    )
    proposed_plan = dspy.OutputField(
        desc="The proposed, step by step plan to perform the task"
    )


class Worker(dspy.Module):
    def __init__(self, role: str, tools: List):
        self.role = role
        self.tools = tools
        self.tool_descriptions = "\n".join(
            [
                f"- {t.name}: {t.description}. To use this tool please provide: `{t.requires}`"
                for t in tools
            ]
        )
        self.plan = dspy.ChainOfThought(Plan)

    def forward(self, task: str):
        context = f"{self.role}\n{self.tool_descriptions}"
        input_args = dict(context=context, task=task)
        result = self.plan(**input_args)
        print(result.proposed_plan)


class Tool(BaseModel):
    name: str
    description: str
    requires: str
    func: Callable


test_tools = [
    Tool(
        name="phone",
        description="Call a phone number",
        requires="phone number",
        func=lambda x: "they've got time",
    ),
    Tool(
        name="local business lookup",
        description="Look up businesses by category",
        requires="business category",
        func=lambda x: "Bills landscaping: 415-555-5555",
    ),
]

# with dspy.context(lm=wrkr):
#     Worker("assistant", test_tools).forward("get this yard cleaned up.")
#     print("\nworker history")
#     print(wrkr.inspect_history(n=10))