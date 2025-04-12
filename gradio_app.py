import gradio as gr
import openai
import pyttsx3
import os
import speech_recognition as sr
import librosa
import numpy as np
import tempfile
import soundfile as sf
from transformers import pipeline

# Set your OpenAI API key
openai.api_key = os.environ.get("Your_open_AI_API_key")

# Initialize text-to-speech engine
engine = pyttsx3.init()
def customize_tts():
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Choose female voice
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
customize_tts()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Emotion detection using HuggingFace's pre-trained model
emotion_recognizer = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

def classify_emotion_using_huggingface(audio_data):
    # Use HuggingFace model for emotion classification
    emotion = emotion_recognizer(audio_data)
    return emotion[0]['label'].lower()  # Extract the emotion label

# Chat memory limit
MAX_HISTORY = 10
message_history = [
    {"role": "system", "content": "You are a helpful and conversational voice assistant."}
]

# Voice listener with retry logic
def listen_with_retry(max_attempts=3):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        attempts = 0
        while attempts < max_attempts:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")

                # Emotion detection using HuggingFace model
                audio_data = audio.get_wav_data()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    tmp_wav.write(audio_data)
                    tmp_wav_path = tmp_wav.name

                emotion = classify_emotion_using_huggingface(tmp_wav_path)

                print(f"Detected Emotion: {emotion}")
                return text, emotion
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                speak("I didn't catch that. Please try again.")
                attempts += 1
            except sr.RequestError:
                speak("Speech recognition service error.")
                return None, "neutral"
    return None, "neutral"

# Core GPT chat function
def chat_with_gpt(user_input, emotion):
    global message_history
    if not user_input:
        return "No input received.", ""

    message_history.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_history,
            request_timeout=10
        )

        reply = response['choices'][0]['message']['content']
        message_history.append({"role": "assistant", "content": reply})

        warning_msg = ""
        if len(message_history) > 2 * MAX_HISTORY + 1:
            message_history = [message_history[0]] + message_history[-2 * MAX_HISTORY:]
            warning_msg = "⚠️ Too many messages! Older ones were removed to stay sharp."

        # Adjust response based on emotion
        if emotion == "happy":
            reply = "😊 " + reply
        elif emotion == "sad":
            reply = "😢 " + reply
        elif emotion == "angry":
            reply = "😡 " + reply
        else:
            reply = reply

        speak(reply)
        return reply, warning_msg
    except Exception as e:
        error_text = f"Error: {str(e)}"
        speak("There was an error generating a response.")
        return error_text, "⚠️ Hmm, something went wrong while getting a response. Please try again."

# Trigger voice input
def voice_input_trigger():
    result, emotion = listen_with_retry()
    if result:
        return result, emotion
    return "😕 I couldn't understand you after a few tries. You can try again or type your message!", "neutral"

# Gradio UI
with gr.Blocks(title="AI Voice Assistant with Emotion Detection") as demo:
    gr.Markdown("## 🎙️ AI Voice Assistant with Emotion Detection + Text + Voice Input + Smart Memory")

    with gr.Row():
        voice_btn = gr.Button("🎤 Speak")
        text_input = gr.Textbox(label="Or Type Your Message")
    output = gr.Textbox(label="Assistant Response", lines=4)
    warnbox = gr.Textbox(label="Status", lines=1)

    voice_btn.click(fn=voice_input_trigger, outputs=[text_input, warnbox])
    text_input.change(fn=chat_with_gpt, inputs=[text_input, warnbox], outputs=[output, warnbox])
    voice_btn.click(fn=chat_with_gpt, inputs=[text_input, warnbox], outputs=[output, warnbox])

demo.launch()
