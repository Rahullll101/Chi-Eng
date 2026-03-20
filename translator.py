import os
import requests
import certifi   # ✅ ADD THIS

def translate_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "[Error: GROQ_API_KEY not found]"

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": """You are a professional technical documentation translator (Chinese → English).

                    STRICT RULES:
                    - ONLY output translated English text
                    - DO NOT add explanations or notes
                    - DO NOT change ANY numbers (e.g., 7000-7010 must remain EXACT)
                    - DO NOT change code, JSON, or technical values
                    - DO NOT fix or guess numbers — copy exactly
                    - Preserve formatting, indentation, and structure
                    - Keep proper nouns unchanged unless clearly translatable
                    - Maintain technical accuracy over fluency
                    """
                },
                {"role": "user", "content": text}
            ],
            "temperature": 0.1
        }

        response = requests.post(
            url,
            headers=headers,
            json=data,
            verify=certifi.where()   # 🔥 THIS LINE FIXES SSL
        )

        result = response.json()

        if "error" in result:
            return f"[Groq Error: {result['error']['message']}]"

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"[Translation Error: {str(e)}]"