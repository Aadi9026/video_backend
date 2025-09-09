# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import subprocess, uuid, os
import whisper
from transformers import pipeline
import cv2
import torch
from decord import VideoReader, cpu
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification

# -------------------------
# Setup
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# AI Models
# -------------------------
whisper_model = whisper.load_model("base")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

# Video Action Recognition
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# -------------------------
# Utilities
# -------------------------
def save_file(file: UploadFile):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

def cut_clip(video_path: str, start: str, end: str, output_name: str):
    output_path = os.path.join(OUTPUT_DIR, output_name)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ss", start,
        "-to", end,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
        "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def extract_frames(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    video = vr.get_batch(indices).asnumpy()
    return list(video)

def classify_action(video_path):
    frames = extract_frames(video_path, num_frames=16)
    inputs = feature_extractor(list(frames), return_tensors="pt")
    with torch.no_grad():
        outputs = video_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    label = video_model.config.id2label[predicted_class]
    return label

def add_captions(video_path: str, language="en"):
    result = whisper_model.transcribe(video_path, task="transcribe")
    transcript = result["text"]

    srt_path = video_path.replace(".mp4", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("1\n00:00:00,000 --> 00:00:10,000\n")
        f.write(transcript + "\n")

    output_path = video_path.replace(".mp4", "_subtitled.mp4")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_path,
        "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path, transcript

# -------------------------
# Scene Detection
# -------------------------
def detect_scenes(video_path: str, scene_type: str, max_clips: int = 2):
    detected_scenes = []

    # (1) Emotion/Text based
    whisper_result = whisper_model.transcribe(video_path, task="transcribe")
    text = whisper_result["text"]
    emotions = emotion_classifier(text)
    if scene_type.lower() in str(emotions).lower():
        detected_scenes.append({"start": "00:01:00", "end": "00:01:20"})

    # (2) Video Action Recognition
    action = classify_action(video_path).lower()
    if scene_type.lower() in action:
        detected_scenes.append({"start": "00:02:00", "end": "00:02:20"})

    return detected_scenes[:max_clips]

# -------------------------
# API Routes
# -------------------------
@app.post("/process_video/")
async def process_video(
    file: UploadFile,
    scene_type: str = Form(...),
    clip_length: int = Form(20),
    max_clips: int = Form(2),
    add_captions_flag: bool = Form(False),
    translate_to: str = Form(None),
    add_music: bool = Form(False)
):
    video_path = save_file(file)
    scenes = detect_scenes(video_path, scene_type, max_clips)

    output_files = []
    transcripts = []

    for i, scene in enumerate(scenes):
        output_name = f"{scene_type}_{i}.mp4"
        clip_path = cut_clip(video_path, scene["start"], scene["end"], output_name)

        if add_captions_flag:
            clip_path, transcript = add_captions(clip_path)
            transcripts.append(transcript)

            if translate_to:
                translation = translator(transcript, max_length=512)
                transcripts[-1] = translation[0]['translation_text']

        output_files.append(clip_path)

    return {
        "clips": output_files,
        "captions": transcripts if transcripts else None,
        "status": "success"
    }

@app.get("/")
async def root():
    return {"message": "AI Video Scene Detection API (Whisper + VideoMAE) 🚀"}
