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
        # module has its predictor
        self.plan = dspy.ChainOfThought(Plan)

    def forward(self, task: str):
        context = f"{self.role}\n{self.tool_descriptions}"
        input_args = dict(context=context, task=task)
        # delegate the forward step to the predictor, but does this then turn our Module into a predictor? 
        # What is the Module abstraction for, I still don't fully understand.
        # ...
        # Ahh, they are callable modules (but the call really just defers to forward, so why add that level of syntactic funk)
        # They are basically about composability, but where do I see the composability? Because it has named predictors?
        #
        # It does look like the telepropmters use the named predictors to do their stuff.
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

with dspy.context(lm=wrkr):
    Worker("assistant", test_tools).forward("get this yard cleaned up.")
    print("\nworker history")
    print(wrkr.inspect_history(n=10))

email_tool = Tool(
    name="email",
    description="Send and receive emails",
    requires="email_address",
    func=lambda x: f"Email sent to {x}"
)
schedule_meeting_tool = Tool(
    name="schedule meeting",
    description="Schedule meetings",
    requires="meeting_details",
    func=lambda x: f"Meeting scheduled on {x}"
)

cleaning_supplies_tool = Tool(
    name="cleaning supplies",
    description="List of cleaning supplies needed",
    requires="cleaning_area",
    func=lambda x: f"Need supplies for {x}"
)
maintenance_report_tool = Tool(
    name="maintenance report",
    description="Report maintenance issues",
    requires="issue_description",
    func=lambda x: f"There's too much work for one person. I need help!"
)

code_compiler_tool = Tool(
    name="code compiler",
    description="Compile code",
    requires="source_code",
    func=lambda x: "Code compiled successfully"
)
bug_tracker_tool = Tool(
    name="bug tracker",
    description="Track and report bugs",
    requires="bug_details",
    func=lambda x: f"Bug reported: {x}"
)

recipe_lookup_tool = Tool(
    name="recipe lookup",
    description="Look up recipes",
    requires="dish_name",
    func=lambda x: f"Recipe for {x} found"
)
kitchen_inventory_tool = Tool(
    name="kitchen inventory",
    description="Check kitchen inventory",
    requires="ingredient",
    func=lambda x: f"Inventory checked for {x}"
)

class Worker2(dspy.Module):
    def __init__(self, role: str, tools: List):
        self.role = role
        self.tools = tools
        self.tool_descriptions = "\n".join(
            [
                f"- {t.name}: {t.description}. To use this tool please provide: `{t.requires}`"
                for t in tools
            ]
        )
        self._plan = dspy.ChainOfThought(Plan)
        self._tool = dspy.ChainOfThought("task, context -> tool_name, tool_argument")

    def plan(self, task: str, feedback:Optional[str]=None):
        context = f"Your role: {self.role}\nTools are your disposal:\n{self.tool_descriptions}"
        if feedback:
            context += f"\nPrevious feedback on your prior plan: {feedback}"
        input_args = dict(context=context, task=task)
        result = self._plan(**input_args)
        return result.proposed_plan

    def execute(self, task:str, use_tool:bool):
        print(f"Executing task: {task} with tool: {use_tool}")
        if not use_tool:
            return f"{task} completed successfully."

        res = self._tool(task=task, context=self.tool_descriptions)
        t = res.tool_name
        if t in self.tools:
            complete = self.tools[t].func(res.tool_argument)
            return complete

        return "Not done."

workers = [
    Worker2("assistant", [email_tool, schedule_meeting_tool]),
    Worker2("janitor", [cleaning_supplies_tool, maintenance_report_tool]),
    Worker2("software engineer", [code_compiler_tool, bug_tracker_tool]),
    Worker2("cook", [recipe_lookup_tool, kitchen_inventory_tool])
]

## Using instructor to parse the responses of the LLM

from pydantic import Field
import instructor
from openai import OpenAI

_client = instructor.from_openai(OpenAI())

class SubTask(BaseModel):
    action: str
    assignee: str
    requires_tool: bool = Field(..., description="Whether the task requires a tool to be used.")

class Task(BaseModel):
    subtasks: List[SubTask]

    
class ParsedPlan(BaseModel):
    tasks: List[Task]

def get_plan(plan: str, context: str):
    return _client.chat.completions.create(
        response_model=ParsedPlan,
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You help parse markdown into a structured format."},
            {"role": "user", "content": f"Here is the context about the plan including the available tools: \n{context}.\n\nThe plan: \n{plan}"}
        ]
    )

class Boss(dspy.Module):