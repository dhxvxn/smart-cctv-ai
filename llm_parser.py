import os
import json
from groq import Groq


class LLMParser:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def parse(self, query):

        prompt = f"""
Convert this CCTV query into structured JSON.

Query: "{query}"

Extract:
- object
- color
- event
- zone
- time

Return ONLY JSON.
"""

        models = [
            "llama-3.1-8b-instant"
        ]

        for model in models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.choices[0].message.content.strip()

                return json.loads(content)

            except Exception as e:
                print(f"⚠ Model {model} failed:", e)

        return {}