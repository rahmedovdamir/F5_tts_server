import os
import time
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import uuid
import queue
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from typing import Optional
import torchaudio
import soundfile as sf
from pydub import AudioSegment, silence
import re
from importlib.resources import files
import sys
import logging
import io
import magic
from pydantic import BaseModel
from cached_path import cached_path
from f5_tts.infer.utils_infer import transcribe

# Add F5-TTS root directory to path so we can import modules
sys.path.append("/workspace/F5-TTS")

from f5_tts.api import F5TTS

logging.basicConfig(level=logging.INFO)

SERVER_ROOT = Path(__file__).resolve().parent
TRT_RESULT_PREFIX = "__F5_TRT_RESULT__"
TTS_BACKEND = os.getenv("F5_TTS_BACKEND", "trt").lower()
TRT_NFE_STEP = int(os.getenv("F5_TRT_NFE_STEP", "14"))
TRT_PYTHON = os.getenv(
    "F5_TRT_PYTHON", "/home/rdr/f5tts-trtllm-py310/bin/python"
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = None
model_lock = threading.Lock()


def get_pytorch_model():
    """Load the legacy backend only when it is explicitly used as a fallback."""
    global model
    with model_lock:
        if model is None:
            model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=str(cached_path("hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base_accent_tune/model_last_inference.safetensors")),
                vocab_file=str(cached_path("hf://Misha24-10/F5-TTS_RUSSIAN/F5TTS_v1_Base/vocab.txt")),
                device=device,
                ode_method="euler",
                use_ema=True,
            )
    return model

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)


class TRTWorker:
    """A persistent bridge to the TensorRT virtualenv."""

    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.output_queue = None
        self.reader_thread = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        env = os.environ.copy()
        mpi_dir = str(SERVER_ROOT / "runtime_libs/openmpi")
        cuda_dir = "/usr/local/cuda/lib64"
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(
            part for part in (mpi_dir, cuda_dir, current_ld_path) if part
        )
        env["F5_TRT_NFE_STEP"] = str(TRT_NFE_STEP)
        self.process = subprocess.Popen(
            [TRT_PYTHON, str(SERVER_ROOT / "trt_worker.py")],
            cwd=str(SERVER_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.output_queue = queue.Queue()

        def read_output():
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line)

        self.reader_thread = threading.Thread(target=read_output, daemon=True)
        self.reader_thread.start()
        response = self._read_response(timeout=90)
        if not response.get("ready"):
            raise RuntimeError(f"TRT worker failed to start: {response}")
        logging.info("TensorRT worker ready (NFE=%s)", TRT_NFE_STEP)

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.output_queue = None
        self.reader_thread = None

    def _read_response(self, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is None or self.output_queue is None:
                raise RuntimeError("TRT worker is not running")
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"TRT worker exited with code {self.process.returncode}"
                )
            try:
                line = self.output_queue.get(
                    timeout=min(1.0, max(0.0, deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if line.startswith(TRT_RESULT_PREFIX):
                return json.loads(line[len(TRT_RESULT_PREFIX):])
            if line.strip():
                logging.debug("TRT worker: %s", line.rstrip())
        raise TimeoutError("Timed out waiting for TRT worker")

    def synthesize(self, **request):
        with self.lock:
            try:
                self.start()
                assert self.process is not None and self.process.stdin is not None
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self._read_response()
            except (BrokenPipeError, RuntimeError):
                self.stop()
                self.start()
                assert self.process is not None and self.process.stdin is not None
                self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                response = self._read_response()
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Unknown TRT worker error"))
            return response


trt_worker = TRTWorker()

# Copy the English reference audio to resources if it doesn't exist
resources_dir = 'resources'
os.makedirs(resources_dir, exist_ok=True)
default_ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
default_ref_text = "Some call me nature, others call me mother nature."

if not os.path.exists(f"{resources_dir}/default_en.wav"):
    import shutil
    shutil.copy2(default_ref_audio, f"{resources_dir}/default_en.wav")

os.makedirs("resources", exist_ok=True)

def convert_to_wav(input_path, output_path):
    """Convert any audio format to WAV using pydub."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(24000)  # Set to F5-TTS expected sample rate
    audio.export(output_path, format='wav')

def split_text_into_sentences(text):
    """Split text into sentences using regex."""
    # Split on common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty sentences and extra whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def detect_leading_silence(audio, silence_threshold=-42, chunk_size=10):
    """Detect silence at the beginning of the audio."""
    trim_ms = 0
    while audio[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
        trim_ms += chunk_size
    return trim_ms

def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from the beginning and end of the audio."""
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim:duration - end_trim]

class UploadAudioRequest(BaseModel):
    audio_file_label: str

def prepare_reference(voice: str, reference_file: str) -> tuple[str, str]:
    """Clip silence and transcribe the reference audio, caching by content hash.

    Called from /upload_audio/ so the cache is warm before a voice is ever used
    for synthesis, and from /synthesize_speech/ as a fallback so it still works
    for voices uploaded before this caching was added.
    """
    if voice == "default_en":
        return reference_file, default_ref_text

    reference_key = hashlib.sha256(Path(reference_file).read_bytes()).hexdigest()[:16]
    temp_short_ref = f'{output_dir}/ref_{reference_key}.wav'
    transcript_path = f'{output_dir}/ref_{reference_key}.txt'

    if os.path.exists(temp_short_ref) and os.path.exists(transcript_path):
        return temp_short_ref, Path(transcript_path).read_text(encoding="utf-8")

    aseg = AudioSegment.from_file(reference_file)

    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
            logging.info("Audio is over 15s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 15000:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                logging.info("Audio is over 15s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # 3. if no proper silence found for clipping
    if len(aseg) > 15000:
        aseg = aseg[:15000]
        logging.info("Audio is over 15s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_short_ref, format='wav')

    # Transcription is cached because it is identical for every request
    # using an unchanged voice file.
    ref_text = transcribe(temp_short_ref)
    Path(transcript_path).write_text(ref_text, encoding="utf-8")
    logging.info(f'Reference text transcribed from first 14s: {ref_text}')

    return temp_short_ref, ref_text

@app.on_event("startup")
async def startup_event():
    if TTS_BACKEND == "trt":
        await asyncio.to_thread(trt_worker.start)
    else:
        await asyncio.to_thread(get_pytorch_model)


@app.on_event("shutdown")
async def shutdown_event():
    trt_worker.stop()

@app.get("/base_tts/")
async def base_tts(text: str, speed: Optional[float] = 1.0):
    """
    Perform text-to-speech conversion using only the base speaker.
    """
    try:
        # Use the default English voice
        return await synthesize_speech(text=text, voice="default_en", speed=speed)
    except Exception as e:
        logging.error(f"Error in base_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/change_voice/")
async def change_voice(reference_speaker: str = Form(...), file: UploadFile = File(...)):
    """
    Change the voice of an existing audio file.
    """
    try:
        logging.info(f'changing voice to {reference_speaker}...')

        contents = await file.read()
        
        # Save the input audio temporarily
        input_path = f'{output_dir}/input_audio.wav'
        with open(input_path, 'wb') as f:
            f.write(contents)

        # Find the reference audio file
        matching_files = [file for file in os.listdir("resources") if file.startswith(str(reference_speaker))]
        if not matching_files:
            raise HTTPException(status_code=400, detail="No matching reference speaker found.")
        
        reference_file = f'resources/{matching_files[0]}'
        
        # Convert reference file to WAV if it's not already
        if not reference_file.lower().endswith('.wav'):
            ref_wav_path = f'{output_dir}/ref_converted.wav'
            convert_to_wav(reference_file, ref_wav_path)
            reference_file = ref_wav_path
        
        # For voice conversion, we'll use the same text for both reference and generation
        # This helps maintain the timing and prosody
        text = transcribe(input_path)
        save_path = f'{output_dir}/output_converted.wav'

        generation_start = time.time()
        wav, sr, _ = get_pytorch_model().infer(
            ref_file=reference_file,
            ref_text=text,
            gen_text=text,
            file_wave=save_path
        )
        generation_time = time.time() - generation_start
        audio_duration = len(wav) / sr
        logging.info(f"Generation completed in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")

        result = FileResponse(
            save_path,
            media_type="audio/wav",
            background=BackgroundTask(os.unlink, save_path),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_audio/")
async def upload_audio(audio_file_label: str = Form(...), file: UploadFile = File(...)):
    """
    Upload an audio file for later use as the reference audio.
    """
    try:
        contents = await file.read()

        allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
        max_file_size = 5 * 1024 * 1024  # 5MB

        if not file.filename.split('.')[-1] in allowed_extensions:
            return {"error": "Invalid file type. Allowed types are: wav, mp3, flac, ogg"}

        if len(contents) > max_file_size:
            return {"error": "File size is over limit. Max size is 5MB."}

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)

        if 'audio' not in file_format:
            return {"error": "Invalid file content."}

        file_extension = file.filename.split('.')[-1]
        stored_file_name = f"{audio_file_label}.{file_extension}"

        with open(f"resources/{stored_file_name}", "wb") as f:
            f.write(contents)

        # Also create a WAV version for F5-TTS
        wav_path = f"resources/{audio_file_label}.wav"
        convert_to_wav(f"resources/{stored_file_name}", wav_path)

        # Warm the clip/transcript cache now so the first real /synthesize_speech/
        # call for this voice doesn't pay for silence trimming and Whisper.
        prepare_reference(audio_file_label, wav_path)

        return {"message": f"File {file.filename} uploaded successfully with label {audio_file_label}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/synthesize_speech/")
async def synthesize_speech(
        text: str,
        voice: str,
        speed: Optional[float] = 1.0,
):
    """
    Synthesize speech from text using a specified voice and style.
    """
    start_time = time.time()
    try:
        logging.info(f'Generating speech for {voice}')

        # First try to find a WAV version
        matching_files = [f for f in os.listdir("resources") if f.startswith(voice) and f.lower().endswith('.wav')]
        
        # If no WAV found, try other formats and convert
        if not matching_files:
            matching_files = [f for f in os.listdir("resources") if f.startswith(voice)]
            if not matching_files:
                raise HTTPException(status_code=400, detail="No matching voice found.")
            
            # Convert to WAV
            input_file = f'resources/{matching_files[0]}'
            wav_path = f'{output_dir}/ref_converted.wav'
            convert_to_wav(input_file, wav_path)
            reference_file = wav_path
        else:
            reference_file = f'resources/{matching_files[0]}'

        # Normally already warmed by /upload_audio/; this is a fallback for voices
        # uploaded before that caching existed, or non-WAV formats.
        reference_file, ref_text = prepare_reference(voice, reference_file)


        save_path = str(
            (SERVER_ROOT / output_dir / f"output_synthesized_{uuid.uuid4().hex}.wav").resolve()
        )
        
        # TensorRT runs in a persistent worker from its own virtualenv. The old
        # PyTorch path remains available as an explicit fallback.
        generation_start = time.time()
        backend_used = TTS_BACKEND
        if TTS_BACKEND == "trt":
            try:
                trt_result = await asyncio.to_thread(
                    trt_worker.synthesize,
                    reference_audio=os.path.abspath(reference_file),
                    ref_text=ref_text,
                    gen_text=text,
                    speed=speed,
                    seed=9527,
                    nfe_step=TRT_NFE_STEP,
                    output=save_path,
                )
                sr = trt_result["sample_rate"]
                audio_duration = trt_result["samples"] / sr
                logging.info(
                    "TRT stages: preprocess=%.3fs trt=%.3fs decode=%.3fs chunks=%s",
                    trt_result["preprocess_seconds"],
                    trt_result["trt_seconds"],
                    trt_result["decode_seconds"],
                    trt_result["chunks"],
                )
            except Exception:
                if os.getenv("F5_TRT_FALLBACK", "1") != "1":
                    raise
                logging.exception("TensorRT inference failed; falling back to PyTorch")
                backend_used = "pytorch-fallback"
                wav, sr, _ = get_pytorch_model().infer(
                    ref_file=reference_file,
                    ref_text=ref_text,
                    gen_text=text,
                    speed=speed,
                    nfe_step=32,
                    cfg_strength=2.0,
                    file_wave=save_path
                )
                audio_duration = len(wav) / sr
        else:
            wav, sr, _ = get_pytorch_model().infer(
                ref_file=reference_file,
                ref_text=ref_text,
                gen_text=text,
                speed=speed,
                nfe_step=32,
                cfg_strength=2.0,
                file_wave=save_path
            )
            audio_duration = len(wav) / sr
        generation_time = time.time() - generation_start
        logging.info(f"Generation completed using {backend_used} in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")

        result = FileResponse(
            save_path,
            media_type="audio/wav",
            background=BackgroundTask(os.unlink, save_path),
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        result.headers["X-Elapsed-Time"] = str(elapsed_time)
        result.headers["X-Device-Used"] = device
        result.headers["X-TTS-Backend"] = backend_used
        result.headers["X-NFE-Step"] = str(TRT_NFE_STEP if backend_used == "trt" else 32)

        # Add CORS headers
        result.headers["Access-Control-Allow-Origin"] = "*"
        result.headers["Access-Control-Allow-Credentials"] = "true"
        result.headers["Access-Control-Allow-Headers"] = "Origin, Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token, locale"
        result.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
