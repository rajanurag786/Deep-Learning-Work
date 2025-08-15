from Prompt_Parser import parse_prompt
from Ansys_Geometry_Creator_with_Prompt import run_static_analysis
from Report_Generator import generate_pdf_report

# 1. Simulate a prompt (or read user input)
user_prompt = "Analyze a box with 0.4x0.4x0.6m, concrete, under 10000N load axially."

if __name__=="__main__":
    # 2. Parse the prompt into structured data
    parsed_data = parse_prompt(user_prompt)
    print(type(parsed_data))
    print(parsed_data["geometry"])

    # 3. Run the static analysis using Ansys
    results = run_static_analysis(parsed_data)

    # 4. Generate the PDF report
    generate_pdf_report(results, parsed_data)
