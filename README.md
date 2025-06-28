# Audio Transcription Service

This is a simple Flask application that performs audio transcription using
[OpenAI Whisper](https://github.com/openai/whisper) and speaker diarization
using [pyannote.audio](https://github.com/pyannote/pyannote-audio). The web
interface allows uploading audio files via drag and drop and displays the
transcript with speaker labels in a dark themed UI.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Hugging Face access token in the environment:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open `http://localhost:5000` in your browser and upload an audio file.

The resulting transcript can be downloaded as a text file.
