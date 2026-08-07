from google import genai
import json
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def search_medicine_online(medicine_name):

    prompt = f"""
Give information about the medicine '{medicine_name}'.

Return ONLY valid JSON in the format:

{{
    "medicine_name":"",
    "composition":"",
    "uses":"",
    "side_effects":"",
    "manufacturer":""
}}

If some information is unavailable, write "Unknown".
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")

        return json.loads(text)

    except Exception as e:
        print("Gemini error:", e)
        return None