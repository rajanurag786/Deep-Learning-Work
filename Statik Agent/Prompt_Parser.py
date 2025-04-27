# import requests
# import json
#
# def parse_prompt(user_prompt):
#     system_instruction = (
#         "You are a structural engineering assistant. Read the user prompt carefully. "
#         "Extract only the information that is explicitly provided. Do not generate data or fill in graphs or image paths unless mentioned. "
#         "Return the result in JSON format with these fields:\n"
#         "{\n"
#         "  'geometry': { 'type': str, 'height_m': float, 'size_m': float },\n"
#         "  'load': { 'type': str, 'magnitude_kN': float },\n"
#         "  'analysis_type': str,\n"
#         "  'outputs': {\n"
#         "     'graphs': [ { 'title': str, 'x_label': str, 'y_label': str } ],\n"
#         "     'images': []\n"
#         "  }\n"
#         "}\n"
#         "If a field is not mentioned in the prompt, set it to null or skip it."
#     )
#
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json = {
#             "model": "llama2",
#             "prompt": f"{system_instruction}\n\n Prompt: {user_prompt}\n JSON:",
#             "stream": False
#         }
#     )
#
#     result = response.json()["response"]
#     parsed_dict = json.loads(result.strip())
#     return parsed_dict
#
# # Example
# user_prompt = "I want to analyze a column of height 3m and size 0.5m with 10kN force on it. The analysis should be static not dynamic and provide me the end report including the stress-strain, displacement graphs and images."
# parsed = parse_prompt(user_prompt)
# print(parsed)

import json
import requests
import re

def extract_and_clean_json(text):
    """
    Extracts JSON-like block from LLM response and converts to valid JSON.
    """
    # Strip text before the first {
    json_start = text.find('{')
    if json_start == -1:
        raise ValueError("No JSON block found.")
    json_str = text[json_start:]

    # Remove trailing text after last }
    json_end = json_str.rfind('}') + 1
    json_str = json_str[:json_end]

    # Replace single quotes with double quotes
    json_str = re.sub(r"'", '"', json_str)

    # Remove trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    return json.loads(json_str)

def parse_prompt(user_prompt):
    prompt_instruction = (
        "You are an engineering assistant. Extract clean JSON from a prompt. "
        "Detect multiple geometries and their relationships. Output format:\n"
        "{\n"
        "  'geometry': [\n"
        "    { 'type': 'cylinder' | 'box' | 'sphere', parameters... }, ...\n"
        "  ],\n"
        "  'material': str,\n"
        "  'load': { 'magnitude_N': float, 'direction': str },\n"
        "  'analysis_type': str\n"
        "}\n"
        "Geometry fields vary: e.g., cylinders have 'radius' and 'height'; boxes have 'width', 'depth', 'height'.\n"
        "If relationships are mentioned (e.g., 'on top of'), calculate relative Z-position.\n"
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": f"{prompt_instruction}\n\nPrompt: {user_prompt}\nJSON:",
            "stream": False
        }
    )
    raw = response.json()["response"]
    print("🧠 Raw LLM output:\n", raw)

    try:
        parsed = extract_and_clean_json(raw)
        return parsed
    except Exception as e:
        print("❌ JSON parsing failed:", e)
        raise e

