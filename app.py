import os
import tempfile
from flask import Flask, request, jsonify, render_template
import whisper
from pyannote.audio import Pipeline

app = Flask(__name__)

# Load models once at startup for efficiency
whisper_model = whisper.load_model("base")

def load_diarization_pipeline():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN environment variable not set")
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=token,
    )

diarization_pipeline = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global diarization_pipeline
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    # Transcription
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    segments = result.get('segments', [])

    # Diarization
    if diarization_pipeline is None:
        diarization_pipeline = load_diarization_pipeline()
    diarization = diarization_pipeline(audio_path)

    # Assign speaker labels to transcription segments
    transcript_lines = []
    for segment in segments:
        start = segment['start']
        speaker = 'Speaker ?'
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= start <= turn.end:
                speaker = label
                break
        line = f"{speaker}: {segment['text'].strip()}"
        transcript_lines.append(line)

    os.unlink(audio_path)
    return jsonify({'text': '\n'.join(transcript_lines)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
