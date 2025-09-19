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
        "IMPORTANT: Convert all load magnitudes to Newtons (N). For example, '10 kN' should become 10000.\n"
        "Geometry fields vary: e.g., cylinders have 'radius' and 'height'; boxes have 'width', 'depth', 'height'.\n"
        "If relationships are mentioned (e.g., 'on top of'), calculate relative Z-position.\n"
    )
    def extract_and_clean_json(text):
        # Step 1: Remove anything before the first {
        json_start = text.find('{')
        if json_start == -1:
            raise ValueError("No JSON object found.")

        text = text[json_start:]

        # Step 2: Find the matching closing brace
        # extract up to the last closing brace to avoid extra trailing comments
        json_end = text.rfind('}') + 1
        text = text[:json_end]

        # Step 3: Replace single quotes with double quotes
        text = text.replace("'", '"')

        # Step 4: Remove trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Step 5: Parse it
        return json.loads(text)
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

