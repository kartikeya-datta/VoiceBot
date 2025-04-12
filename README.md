# VoiceBot
In this project I will be presenting a basic Voice assistant bot model using OpenAI GPT.


Download
# AI Voice Assistant with Emotion Detection

This project is an AI-powered voice assistant that integrates emotion detection from audio input and utilizes GPT-3.5 for generating conversational responses. The assistant supports both voice and text input, and it customizes its responses based on the detected emotions, such as happy, sad, or angry. The app is built using Gradio for the user interface, OpenAI's GPT-3.5 for chat responses, and HuggingFace's pre-trained models for emotion detection from voice.

## Features

- **Voice and Text Input**: Users can either speak or type their message to interact with the assistant.
- **Emotion Detection**: The assistant detects emotions like happiness, sadness, and anger from the user's voice input.
- **Personalized Responses**: Based on the detected emotion, the assistant customizes its responses, adding emoticons (e.g., 😊 for happy, 😢 for sad).
- **Hotword Activation**: Users can trigger the assistant by saying a specific hotword (e.g., "Hey Assistant").
- **Smart Memory**: The assistant remembers the recent conversation context to provide coherent and relevant responses.

## Setup Instructions

To run this project locally, follow these steps:

### 1. Clone the Repository

First, clone the project repository to your local machine.

```bash
git clone https://github.com/your-username/voice-assistant.git
cd voice-assistant
```

### 1. Clone the Repository
Install the necessary Python libraries by using the requirements.txt file provided.

```bash
pip install -r requirements.txt
```
### 3. Set Up OpenAI API Key
You need to set up your OpenAI API key to use GPT-3.5 for generating responses.

- Create an account on OpenAI.

- Obtain your API key from the OpenAI dashboard.

- Set the API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Alternatively, you can directly set the API key within the code (though it's not recommended for security reasons).

### 4. Run the Application
Run the application using Gradio. The application will start a web-based user interface for interacting with the voice assistant.

```bash
python gradio_app.py
```

Once the app is running, a URL will be provided where you can access the Gradio interface in your web browser.


### 5. Interact with the Assistant
- Voice Input: Click the "Speak" button or say the hotword (e.g., "Hey Assistant") to activate the assistant.

- Text Input: You can also type your message in the text box and receive a response.

### 6. How Emotion Detection Works
The assistant detects emotions from your voice using a pre-trained HuggingFace model (wav2vec2-lg-xlsr-en-speech-emotion-recognition). When you speak, the assistant captures the audio, sends it to the HuggingFace model, and classifies the emotion from the voice tone (e.g., happy, sad, angry). Based on the detected emotion, the assistant customizes its response.

### 7. Hotword Activation
The hotword activation feature uses a voice listener that continuously listens for a specific hotword (e.g., "Hey Assistant"). Once the hotword is detected, the assistant becomes active and ready to respond. You can say the hotword multiple times to trigger the assistant.

Technologies Used
- Gradio: For creating the user interface and handling the communication between the frontend and backend.

- OpenAI GPT-3.5: For generating intelligent conversational responses.

- HuggingFace Transformers: For emotion recognition from audio using pre-trained models.

- SpeechRecognition: For capturing and converting speech to text.

- pyttsx3: For converting the assistant's text responses to speech.

- librosa & soundfile: For handling audio files and processing the speech data.

Troubleshooting
- If the assistant doesn't respond to the hotword, ensure your microphone is correctly configured and sensitive enough to detect audio input.

- If there are issues with speech recognition, make sure you have an active internet connection and that your microphone is properly set up.

- For emotion detection issues, ensure that the HuggingFace model is accessible and loaded correctly.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Feel free to contribute or raise issues on this project. If you have any questions, please open an issue or contact me directly. Thanks for going through my work 😁.