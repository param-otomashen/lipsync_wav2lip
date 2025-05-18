import pyttsx3 

import os
import requests
engine=pyttsx3.init() 
voices = engine.getProperty('voices')
#female voice:
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0')
#slower voice speed:
engine.setProperty('rate', 125)

import openai
import os

TOGETHER_API_KEY = "tgp_v1_wGmsTiix2XJgaSPZ0jLt98JcYG7JKik2YfJEtuYLZZY" 

def ask_gemini(prompt):
    print(f"🔹 Sending prompt to Together.ai: {prompt}")
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {"role": "system", "content": "Be brief. Answer in less than 20 words."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 60,
            "temperature": 0.7
        }

        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()
        print("✅ Together.ai reply:", reply)
        return reply

    except Exception as e:
        print("❌ Together.ai API error:", e)
        return "Sorry, I couldn't generate a response."


def generate_tts(response):
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../wav2lip/examples/audio.wav")) 
    engine.save_to_file(response, output_path)   
    engine.runAndWait()
    return output_path
