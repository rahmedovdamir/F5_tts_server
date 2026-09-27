"""Persistent TensorRT-LLM worker used by server.py.

The HTTP server and TensorRT-LLM intentionally live in separate virtualenvs.
Requests and responses are newline-delimited JSON; TensorRT's regular stdout
logging is ignored by the parent until it sees RESULT_PREFIX.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torchaudio


FASTER_ROOT = Path(os.environ.get("F5_FASTER_ROOT", "/home/rdr/F5_TTS_Faster"))
ENGINE_DIR = Path(
    os.environ.get(
        "F5_TRT_ENGINE_DIR", str(FASTER_ROOT / "ckpts/misha_engine_4060_fast")
    )
)
NFE_STEP = int(os.environ.get("F5_TRT_NFE_STEP", "14"))
MIN_AUDIO_SECONDS = float(os.environ.get("F5_TRT_MIN_AUDIO_SECONDS", "0.8"))
RESULT_PREFIX = "__F5_TRT_RESULT__"

os.chdir(FASTER_ROOT)
sys.path.insert(0, str(FASTER_ROOT / "export_trtllm"))
import sample as runtime  # noqa: E402


runtime.args = argparse.Namespace(
    gpus_per_node=1,
    tllm_model_dir=str(ENGINE_DIR),
)
runtime.tensorrt_llm.logger.set_level("error")

with (ENGINE_DIR / "config.json").open(encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)


class PersistentSynthesizer:
    def __init__(self) -> None:
        self.models = [runtime.F5TTS(CONFIG, debug_mode=False, device="cuda:0")]

    def _ensure_models(self, count: int) -> None:
        while len(self.models) < count:
            self.models.append(runtime.F5TTS(CONFIG, debug_mode=False, device="cuda:0"))

    def synthesize(self, request: dict) -> dict:
        started = time.perf_counter()
        reference_audio = str(Path(request["reference_audio"]).resolve())
        ref_text = request["ref_text"]
        gen_text = request["gen_text"]
        output = str(Path(request["output"]).resolve())
        speed = float(request.get("speed", 1.0))
        seed = int(request.get("seed", 9527))
        nfe_step = int(request.get("nfe_step", NFE_STEP))

        runtime.SPEED = speed
        runtime.NFE_STEP = nfe_step
        torch.manual_seed(seed)
        reference_audio, ref_text = runtime.preprocess_ref_audio_text(
            reference_audio, ref_text
        )

        ref_audio, ref_sr = torchaudio.load(reference_audio)
        ref_seconds = ref_audio.shape[-1] / ref_sr
        max_chars = max(
            1,
            int(
                len(ref_text.encode("utf-8"))
                / ref_seconds
                * (22 - ref_seconds)
                * speed
            ),
        )
        chunks = runtime.chunk_text(gen_text, max_chars=max_chars)
        if not chunks:
            raise ValueError("Generation text is empty")
        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text += " "

        self._ensure_models(len(chunks))
        prepared = []
        preprocess_started = time.perf_counter()
        for chunk_index, chunk in enumerate(chunks, start=1):
            audio, text_ids, max_duration, initial_noise = runtime.get_input(
                reference_audio, ref_text, chunk, seed
            )
            # Very short phrases (for example "Алло.") can be assigned fewer
            # than 30 mel frames. Vocos then produces an almost silent clip.
            # Give short replies enough frames to form a complete syllable.
            ref_frames = audio.shape[-1] // runtime.HOP_LENGTH
            min_generated_frames = int(
                MIN_AUDIO_SECONDS * runtime.SAMPLE_RATE / runtime.HOP_LENGTH
            )
            min_duration = ref_frames + min_generated_frames
            if int(max_duration) < min_duration:
                max_duration = np.array(min_duration, dtype=np.int64)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                initial_noise = torch.randn(
                    (1, min_duration, 100), generator=generator
                ).numpy().astype(np.float32)
            values = runtime.preprocess(audio, text_ids, max_duration, initial_noise)
            prepared.append((chunk_index, values))
        preprocess_seconds = time.perf_counter() - preprocess_started

        def run_chunk(model, item):
            chunk_index, values = item
            (
                noise,
                cond,
                cond_drop,
                time_expand,
                rope_cos,
                rope_sin,
                delta_t,
                ref_signal_len,
                rms,
            ) = values
            denoised = model.forward(
                torch.from_numpy(noise).cuda(),
                torch.from_numpy(cond).cuda(),
                torch.from_numpy(cond_drop).cuda(),
                time_expand.cuda(),
                torch.from_numpy(rope_cos).cuda(),
                torch.from_numpy(rope_sin).cuda(),
                delta_t.cuda(),
            )
            return chunk_index, denoised, ref_signal_len, rms

        trt_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(run_chunk, model, item)
                for model, item in zip(self.models, prepared)
            ]
            generated = [future.result() for future in futures]
        torch.cuda.synchronize()
        trt_seconds = time.perf_counter() - trt_started

        decode_started = time.perf_counter()
        waves = []
        for _, denoised, ref_signal_len, rms in sorted(generated):
            waves.append(
                runtime.decode(
                    denoised.cpu().numpy().astype(np.float32), ref_signal_len, rms
                )
            )
        final_wave = runtime.combine_with_crossfade(waves)
        peak = float(np.max(np.abs(final_wave)))
        if peak > 0.99:
            final_wave *= 0.99 / peak
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        # Telephony clients commonly reject IEEE-float WAV even though it is a
        # valid WAV file. Match the legacy F5 server and emit signed PCM16.
        torchaudio.save(
            output,
            torch.from_numpy(final_wave).unsqueeze(0),
            runtime.SAMPLE_RATE,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        decode_seconds = time.perf_counter() - decode_started

        return {
            "ok": True,
            "output": output,
            "sample_rate": runtime.SAMPLE_RATE,
            "samples": len(final_wave),
            "chunks": len(chunks),
            "nfe_step": nfe_step,
            "preprocess_seconds": preprocess_seconds,
            "trt_seconds": trt_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": time.perf_counter() - started,
        }


def emit(payload: dict) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    synthesizer = PersistentSynthesizer()
    emit({"ok": True, "ready": True, "nfe_step": NFE_STEP})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            emit(synthesizer.synthesize(request))
        except Exception as exc:
            emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
