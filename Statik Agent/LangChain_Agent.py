from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from Prompt_Parser import parse_prompt
from Ansys_Geometry_Creator_with_Prompt import run_static_analysis
from Report_Generator import generate_pdf_report

def FEA_pipeline(prompt):
    parsed = parse_prompt(prompt)
    results = run_static_analysis(parsed)
    generate_pdf_report(results, parsed)
    return "Static Analysis completed. PDF report has been generated"

tools = [
    Tool(
        name= "StaticFEA",
        func= FEA_pipeline,
        description="Use this tool to perform static analysis of a column and generate a report from a natural language prompt."

    )
]

llm = Ollama(model="llama2")

prompt = PromptTemplate.from_template("""
You are an engineering assistant. Convert the following user request into a static analysis run.
Then generate a report using the tools provided.

Request: {user_prompt}
""")

chain = LLMChain(llm=llm, prompt=prompt)

user_input = input("💬 Prompt: ")
parsed = parse_prompt(user_input)
results = run_static_analysis(parsed)
generate_pdf_report(results, parsed)
print("✅ Report ready.")

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
#     handle_parsing_errors=True
# )

# if __name__ == "__main__":
#     while True:
#         user_input = input("\n Ask the Static Agent: ")
#         if user_input.lower() in ["exit", "quit", "stop"]:
#             break
#         response = agent.run(user_input)
#         print("\n Static Agent response:\n", response)

