# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Gradio studio for bulk mixed-language Qwen3-TTS generation.
"""

import argparse
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from .. import Qwen3TTSModel


_TAG_LANGUAGE_ALIASES = {
    "EN": "English",
    "ENG": "English",
    "ENGLISH": "English",
    "ES": "Spanish",
    "SPA": "Spanish",
    "SPANISH": "Spanish",
    "DE": "German",
    "GER": "German",
    "GERMAN": "German",
    "FR": "French",
    "FRE": "French",
    "FRENCH": "French",
    "IT": "Italian",
    "ITA": "Italian",
    "ITALIAN": "Italian",
    "PT": "Portuguese",
    "POR": "Portuguese",
    "PORTUGUESE": "Portuguese",
    "RU": "Russian",
    "RUS": "Russian",
    "RUSSIAN": "Russian",
    "JA": "Japanese",
    "JP": "Japanese",
    "JPN": "Japanese",
    "JAPANESE": "Japanese",
    "KO": "Korean",
    "KR": "Korean",
    "KOR": "Korean",
    "KOREAN": "Korean",
    "ZH": "Chinese",
    "CN": "Chinese",
    "CHINESE": "Chinese",
}
_TAG_RE = re.compile(r"\[(/?)([A-Za-z]{2,12})\]")


def _title_case_display(s: str) -> str:
    s = (s or "").strip().replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    return display, {d: r for d, r in zip(display, items)}


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio studio for Qwen3 TTS models.\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --dtype float16 --no-flash-attn\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --port 8000 --ip 127.0.0.1\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    parser.add_argument("checkpoint_pos", nargs="?", default=None, help="Model checkpoint path or Hugging Face repo id.")
    parser.add_argument("-c", "--checkpoint", default=None, help="Model checkpoint path or Hugging Face repo id.")
    parser.add_argument("--device", default="cuda:0", help="Device for device_map, e.g. cpu, cuda, cuda:0.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model.",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2.",
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind IP.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link.",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Gradio queue concurrency.")
    parser.add_argument("--ssl-certfile", default=None, help="Path to SSL certificate file.")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file.")
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificates.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k.")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p.")
    parser.add_argument("--subtalker-temperature", type=float, default=None, help="Subtalker temperature.")
    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise SystemExit(0)
    return ckpt


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        return _normalize_audio(audio["data"]), sr
    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    return sr, np.asarray(wav, dtype=np.float32)


def _detect_model_kind(tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def _clean_text_chunk(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _append_segment(segments: List[Dict[str, str]], language: str, text: str) -> None:
    text = _clean_text_chunk(text)
    if not text:
        return
    if segments and re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        segments[-1]["text"] = f"{segments[-1]['text']}{text}".strip()
        return
    leading_punctuation = re.match(r"^([^\w\s]+)\s+(.*)$", text, flags=re.UNICODE)
    if segments and leading_punctuation:
        segments[-1]["text"] = f"{segments[-1]['text']}{leading_punctuation.group(1)}".strip()
        text = leading_punctuation.group(2).strip()
        if not text:
            return
    if segments and segments[-1]["language"] == language:
        segments[-1]["text"] = f"{segments[-1]['text']} {text}".strip()
    else:
        segments.append({"language": language, "text": text})


def _split_long_text(text: str, max_chars: int) -> List[str]:
    text = _clean_text_chunk(text)
    if not text:
        return []
    max_chars = max(80, int(max_chars or 650))
    pieces = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    chunks: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            flush_current()
            word_chunk = ""
            for word in piece.split():
                candidate = f"{word_chunk} {word}".strip()
                if word_chunk and len(candidate) > max_chars:
                    chunks.append(word_chunk)
                    word_chunk = word
                else:
                    word_chunk = candidate
            if word_chunk:
                chunks.append(word_chunk)
            continue
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            flush_current()
            current = piece
        else:
            current = candidate
    flush_current()
    return chunks


def _parse_tagged_script(script: str, base_language: str, max_chars: int) -> List[Dict[str, str]]:
    base_language = base_language or "English"
    current_language = base_language
    segments: List[Dict[str, str]] = []
    pos = 0

    for match in _TAG_RE.finditer(script or ""):
        _append_segment(segments, current_language, script[pos:match.start()])
        is_closing, raw_tag = match.groups()
        tagged_language = _TAG_LANGUAGE_ALIASES.get(raw_tag.upper())
        if tagged_language:
            if is_closing:
                current_language = base_language
            elif current_language.lower() == tagged_language.lower():
                current_language = base_language
            else:
                current_language = tagged_language
        pos = match.end()

    _append_segment(segments, current_language, (script or "")[pos:])

    split_segments: List[Dict[str, str]] = []
    for seg in segments:
        for chunk in _split_long_text(seg["text"], max_chars):
            split_segments.append({"language": seg["language"], "text": chunk})
    return split_segments


def _preview_segments(segments: List[Dict[str, str]], limit: int = 100) -> str:
    if not segments:
        return "No script segments found."
    lines = []
    for i, seg in enumerate(segments[:limit], start=1):
        text = seg["text"]
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        lines.append(f"{i:02d}. [{seg['language']}] {text}")
    if len(segments) > limit:
        lines.append(f"... {len(segments) - limit} more segment(s)")
    return "\n".join(lines)


def _read_uploaded_text(file_obj: Any) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, dict):
        path = file_obj.get("name") or file_obj.get("path")
    else:
        path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
    if not path or not os.path.exists(path):
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _script_from_inputs(pasted_text: str, file_obj: Any) -> str:
    uploaded_text = _read_uploaded_text(file_obj)
    pasted_text = pasted_text or ""
    if uploaded_text.strip() and pasted_text.strip():
        return f"{pasted_text.strip()}\n\n{uploaded_text.strip()}"
    if uploaded_text.strip():
        return uploaded_text
    return pasted_text


def _concat_wavs(wavs: List[np.ndarray], sr: int, silence_ms: int) -> np.ndarray:
    if not wavs:
        return np.zeros(0, dtype=np.float32)
    silence_len = max(0, int(sr * int(silence_ms or 0) / 1000))
    silence = np.zeros(silence_len, dtype=np.float32)
    pieces: List[np.ndarray] = []
    for i, wav in enumerate(wavs):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1).astype(np.float32)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 1.0:
            wav = wav / peak
        if i and silence_len:
            pieces.append(silence)
        pieces.append(wav)
    return np.clip(np.concatenate(pieces), -1.0, 1.0).astype(np.float32)


def _write_temp_wav(wav: np.ndarray, sr: int) -> str:
    fd, out_path = tempfile.mkstemp(prefix="qwen_tts_script_", suffix=".wav")
    os.close(fd)
    sf.write(out_path, wav, sr)
    return out_path


def build_demo(tts: Qwen3TTSModel, ckpt: str, gen_kwargs_default: Dict[str, Any]) -> gr.Blocks:
    model_kind = _detect_model_kind(tts)

    supported_langs_raw = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()

    supported_spks_raw = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_spks_raw = tts.model.get_supported_speakers()

    lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])
    spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])
    lang_lookup = {str(v).lower(): v for v in (supported_langs_raw or [])}
    lang_lookup.update({str(k).lower(): v for k, v in lang_map.items()})

    def resolve_language(name: str) -> str:
        if not name:
            return "Auto"
        return lang_lookup.get(str(name).lower(), name)

    def default_choice(choices: List[str], preferred: List[str], fallback: str = "") -> str:
        by_lower = {str(x).lower(): x for x in choices}
        for item in preferred:
            if item.lower() in by_lower:
                return by_lower[item.lower()]
        return choices[0] if choices else fallback

    default_language = default_choice(lang_choices_disp, ["English", "Auto"], "English")
    default_speaker = default_choice(spk_choices_disp, ["Ryan", "Aiden", "Vivian"], spk_choices_disp[0] if spk_choices_disp else "")

    def gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;} textarea {font-family: ui-monospace, SFMono-Regular, Consolas, monospace;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(
            f"""
# Language Learning TTS Studio
**Checkpoint:** `{ckpt}`  
**Model Type:** `{model_kind}`
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                script_in = gr.Textbox(
                    label="Script",
                    lines=16,
                    value="Today we are practicing [ES]buenos dias[ES]. It means good morning. Repeat after me: [ES]buenos dias[ES].",
                    placeholder="Paste a script here. Use [ES]hola[ES], [DE]guten Morgen[DE], [FR]bonjour[FR], etc.",
                )
                script_file = gr.File(label="Optional .txt script upload", file_types=[".txt"])
                with gr.Row():
                    base_lang_in = gr.Dropdown(
                        label="Default language",
                        choices=lang_choices_disp or ["English", "Auto"],
                        value=default_language,
                        interactive=True,
                    )
                    max_chars_in = gr.Number(label="Max characters per segment", value=650, precision=0)
                    silence_in = gr.Number(label="Pause between segments (ms)", value=180, precision=0)
                with gr.Row():
                    preview_btn = gr.Button("Preview Segments")
                    generate_btn = gr.Button("Generate Script Audio", variant="primary")

            with gr.Column(scale=2):
                if model_kind == "custom_voice":
                    speaker_in = gr.Dropdown(
                        label="Speaker",
                        choices=spk_choices_disp,
                        value=default_speaker,
                        interactive=True,
                    )
                    instruct_in = gr.Textbox(
                        label="Optional style instruction",
                        lines=3,
                        placeholder="Example: friendly teacher voice, clear pronunciation, natural pace.",
                    )
                    design_in = gr.State("")
                    ref_audio = gr.State(None)
                    ref_text = gr.State("")
                    xvec_only = gr.State(False)
                elif model_kind == "voice_design":
                    design_in = gr.Textbox(
                        label="Voice design instruction",
                        lines=4,
                        value="Friendly language teacher voice, clear pronunciation, natural pace.",
                    )
                    speaker_in = gr.State("")
                    instruct_in = gr.State("")
                    ref_audio = gr.State(None)
                    ref_text = gr.State("")
                    xvec_only = gr.State(False)
                else:
                    ref_audio = gr.Audio(label="Reference audio", type="numpy")
                    ref_text = gr.Textbox(
                        label="Reference transcript",
                        lines=3,
                        placeholder="Required unless x-vector only is enabled.",
                    )
                    xvec_only = gr.Checkbox(label="Use x-vector only", value=False)
                    speaker_in = gr.State("")
                    instruct_in = gr.State("")
                    design_in = gr.State("")

                preview_out = gr.Textbox(label="Segment preview", lines=12)
                status_out = gr.Textbox(label="Status", lines=4)
                audio_out = gr.Audio(label="Generated audio", type="numpy")
                file_out = gr.File(label="Download WAV")

        def preview_script(script_text: str, file_obj, base_lang_disp: str, max_chars):
            script = _script_from_inputs(script_text, file_obj)
            if not script.strip():
                return "No script text found."
            base_language = resolve_language(lang_map.get(base_lang_disp, base_lang_disp))
            segments = _parse_tagged_script(script, base_language, int(max_chars or 650))
            return _preview_segments(segments)

        def generate_script(
            script_text: str,
            file_obj,
            base_lang_disp: str,
            max_chars,
            silence_ms,
            speaker_disp: str,
            instruct: str,
            design: str,
            ref_aud,
            ref_txt: str,
            use_xvec: bool,
        ):
            try:
                script = _script_from_inputs(script_text, file_obj)
                if not script.strip():
                    return None, None, "No script segments found.", "Paste a script or upload a .txt file."

                base_language = resolve_language(lang_map.get(base_lang_disp, base_lang_disp))
                segments = _parse_tagged_script(script, base_language, int(max_chars or 650))
                if not segments:
                    return None, None, "No script segments found.", "No script segments found."

                kwargs = gen_common_kwargs()
                wavs_out: List[np.ndarray] = []
                sr_out: Optional[int] = None
                voice_clone_prompt = None

                if model_kind == "custom_voice":
                    if not speaker_disp:
                        return None, None, _preview_segments(segments), "Choose a speaker."
                    speaker = spk_map.get(speaker_disp, speaker_disp)
                elif model_kind == "voice_design":
                    if not design or not design.strip():
                        return None, None, _preview_segments(segments), "Voice design instruction is required."
                else:
                    at = _audio_to_tuple(ref_aud)
                    if at is None:
                        return None, None, _preview_segments(segments), "Reference audio is required for Base voice clone models."
                    if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                        return None, None, _preview_segments(segments), "Reference transcript is required unless x-vector only is enabled."
                    voice_clone_prompt = tts.create_voice_clone_prompt(
                        ref_audio=at,
                        ref_text=(ref_txt.strip() if ref_txt else None),
                        x_vector_only_mode=bool(use_xvec),
                    )

                for seg in segments:
                    language = resolve_language(seg["language"])
                    text = seg["text"]
                    if model_kind == "custom_voice":
                        wavs, sr = tts.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker,
                            instruct=(instruct or "").strip() or None,
                            **kwargs,
                        )
                    elif model_kind == "voice_design":
                        wavs, sr = tts.generate_voice_design(
                            text=text,
                            language=language,
                            instruct=design.strip(),
                            **kwargs,
                        )
                    else:
                        wavs, sr = tts.generate_voice_clone(
                            text=text,
                            language=language,
                            voice_clone_prompt=voice_clone_prompt,
                            **kwargs,
                        )
                    wavs_out.append(wavs[0])
                    sr_out = sr

                assert sr_out is not None
                final_wav = _concat_wavs(wavs_out, sr_out, int(silence_ms or 0))
                wav_path = _write_temp_wav(final_wav, sr_out)
                status = f"Finished. Generated {len(segments)} segment(s) into one WAV."
                return _wav_to_gradio_audio(final_wav, sr_out), wav_path, _preview_segments(segments), status
            except Exception as e:
                return None, None, "", f"{type(e).__name__}: {e}"

        common_inputs = [
            script_in,
            script_file,
            base_lang_in,
            max_chars_in,
            silence_in,
            speaker_in,
            instruct_in,
            design_in,
            ref_audio,
            ref_text,
            xvec_only,
        ]
        preview_btn.click(preview_script, inputs=[script_in, script_file, base_lang_in, max_chars_in], outputs=[preview_out])
        generate_btn.click(generate_script, inputs=common_inputs, outputs=[audio_out, file_out, preview_out, status_out])

        gr.Markdown(
            """
**Use note**  
Tagged spans are generated as separate segments and stitched into one WAV. Use consent-based reference audio for cloning, and review pronunciation before publishing language-learning material.
"""
        )

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)
    dtype = _dtype_from_str(args.dtype)
    attn_impl = "flash_attention_2" if args.flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    demo = build_demo(tts, ckpt, _collect_gen_kwargs(args))
    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
