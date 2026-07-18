from Prompt_Parser import parse_prompt
from Ansys_Geometry_Creator_with_Prompt import run_static_analysis
from Report_Generator import generate_pdf_report
from Input_validator import validate_simulation_input

# 1. Simulate a prompt (or read user input)
user_prompt = "Analyze a box with 0.4x0.4x0.6m, concrete, under 10000N load axially."

parsed_data = {
    "geometry": [
        {
            "type": "box",
            "parameters": {
                "width": 0.4,
                "depth": 0.4,
                "height": 0.6,
                "position": [0.0, 0.0, 0.0]
            }
        },
        {
            "type": "box",
            "parameters": {
                "width": 0.4,
                "depth": 0.4,
                "height": 0.6,
                "position": [0.0, 0.0, 0.6]
            }
        }
    ],
    "material": "concrete",
    "load": {
        "magnitude_N": 10000
    }
}

if __name__=="__main__":
    # 2. Parse the prompt into structured data
    # parsed_data = parse_prompt(user_prompt)
    print("This file is for backend testing only.")
    print("To run the UI, use: streamlit run app.py")
