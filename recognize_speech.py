import argparse
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Queue
from typing import Iterator

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

import config


SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512
SAMPLE_WIDTH_BYTES = 2


@dataclass
class SpeechConfig:
    model_path: str = config.STT_PATH
    record_backend: str = "parecord"
    mic_device: str | None = None
    stt_device: str = "cpu"
    compute_type: str = "default"       # "default": "float16" / "int8", etc.
    language: str | None = None
    beam_size: int = 5
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 1500      # after speech starts, VAD waits for about vad_min_silence_ms of silence before deciding the utterance is finished and sending it to STT.
    vad_speech_pad_ms: int = 100
    pre_roll_ms: int = 300              # prevent VAD missing the start of speech by including some audio from before VAD triggers. This is typically set to a value slightly longer than vad_min_silence_ms to account for the fact that VAD may trigger in the middle of a silence period.
    min_speech_ms: int = 300            # discard speech segments shorter than this duration, which are likely to be VAD errors
    max_speech_seconds: float = 20.0    # forcibly finalize speech segments longer than this duration
    show_level: bool = True
    level_interval_ms: int = 200
    verbose: bool = True


class SpeechRecognizer:
    def __init__(self, speech_config: SpeechConfig | None = None):
        self.config = speech_config or SpeechConfig()
        self._vad_iterator: VADIterator | None = None
        self._stt_model: WhisperModel | None = None
        self._last_level_time = 0.0

    def _load_models(self) -> None:
        if self._vad_iterator is None:
            vad_model = load_silero_vad()
            self._vad_iterator = VADIterator(
                vad_model,
                threshold=self.config.vad_threshold,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=self.config.vad_min_silence_ms,
                speech_pad_ms=self.config.vad_speech_pad_ms,
            )

        if self._stt_model is None:
            if not os.path.isdir(self.config.model_path):
                raise FileNotFoundError(
                    f"STT model directory does not exist: {self.config.model_path}"
                )
            self._log(
                "Loading faster-whisper model "
                f"from {self.config.model_path} "
                f"with device={self.config.stt_device}, "
                f"compute_type={self.config.compute_type}"
            )
            self._stt_model = WhisperModel(
                self.config.model_path,
                device=self.config.stt_device,
                compute_type=self.config.compute_type,
            )

    def listen(self) -> Iterator[str]:
        self._load_models()
        assert self._vad_iterator is not None

        recorder = self._start_recorder()
        chunk_bytes = VAD_CHUNK_SAMPLES * SAMPLE_WIDTH_BYTES
        pre_roll_chunks = max(
            1,
            int((self.config.pre_roll_ms / 1000) * SAMPLE_RATE / VAD_CHUNK_SAMPLES),
        )
        pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
        speech_chunks: list[np.ndarray] = []
        recording = False

        self._log(
            f"Listening with {self.config.record_backend}. "
            "Speak after VAD is ready; press Ctrl+C to stop."
        )

        try:
            while True:
                raw_audio = recorder.stdout.read(chunk_bytes) if recorder.stdout else b""
                if len(raw_audio) < chunk_bytes:
                    return_code = recorder.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            f"{self.config.record_backend} stopped with exit code {return_code}"
                        )
                    continue

                chunk = self._pcm16_to_float32(raw_audio)
                was_recording = recording
                vad_event = self._vad_iterator(torch.from_numpy(chunk))
                self._print_level(chunk, recording)

                if was_recording:
                    speech_chunks.append(chunk)
                else:
                    pre_roll.append(chunk)

                if vad_event and "start" in vad_event and not recording:
                    recording = True
                    speech_chunks = list(pre_roll)
                    pre_roll.clear()
                    self._log("Speech started")

                if not recording:
                    continue

                speech_seconds = (
                    sum(len(part) for part in speech_chunks) / SAMPLE_RATE
                    if speech_chunks
                    else 0.0
                )
                should_finalize = bool(vad_event and "end" in vad_event)
                should_finalize = (
                    should_finalize or speech_seconds >= self.config.max_speech_seconds
                )

                if not should_finalize:
                    continue

                audio = np.concatenate(speech_chunks) if speech_chunks else np.array([])
                speech_chunks = []
                recording = False
                self._vad_iterator.reset_states()

                if len(audio) < self._samples_from_ms(self.config.min_speech_ms):
                    self._log("Speech skipped because it was too short")
                    continue

                text = self.transcribe(audio)
                if text:
                    yield text

        finally:
            self._stop_recorder(recorder)

    def transcribe(self, audio: np.ndarray) -> str:
        self._load_models()
        assert self._stt_model is not None

        # A small amount of trailing silence helps Whisper finish short commands.
        trailing_silence = np.zeros(self._samples_from_ms(250), dtype=np.float32)
        audio = np.concatenate([audio.astype(np.float32), trailing_silence])

        segments, _ = self._stt_model.transcribe(
            audio,
            beam_size=self.config.beam_size,
            language=self.config.language,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        text = " ".join(text_parts).strip()
        self._log(f"Recognized: {text}" if text else "No text recognized")
        return text

    def _start_recorder(self) -> subprocess.Popen:
        if self.config.record_backend == "parecord":
            command = [
                "parecord",
                "--raw",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--format=s16le",
            ]
            if self.config.mic_device:
                command.append(f"--device={self.config.mic_device}")
        elif self.config.record_backend == "arecord":
            command = [
                "arecord",
                "-q",
                "-D",
                self.config.mic_device or "default",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                "1",
            ]
        else:
            raise ValueError(f"Unsupported record backend: {self.config.record_backend}")

        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=None if self.config.verbose else subprocess.DEVNULL,
            bufsize=0,
        )

    @staticmethod
    def _stop_recorder(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    def _pcm16_to_float32(raw_audio: bytes) -> np.ndarray:
        return np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _samples_from_ms(milliseconds: int) -> int:
        return int((milliseconds / 1000) * SAMPLE_RATE)

    def _print_level(self, audio: np.ndarray, recording: bool) -> None:
        if not self.config.show_level:
            return

        now = time.monotonic()
        interval_seconds = self.config.level_interval_ms / 1000
        if now - self._last_level_time < interval_seconds:
            return
        self._last_level_time = now

        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        dbfs = 20 * math.log10(max(rms, 1e-8))
        normalized = min(1.0, max(0.0, (dbfs + 60.0) / 60.0))
        filled = int(normalized * 24)
        meter = "#" * filled + "-" * (24 - filled)
        state = "speech" if recording else "idle"
        print(
            f"\r[recognize_speech] level {dbfs:6.1f} dBFS [{meter}] {state}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def _log(self, message: str) -> None:
        if self.config.verbose:
            if self.config.show_level:
                print(file=sys.stderr, flush=True)
            print(f"[recognize_speech] {message}", file=sys.stderr, flush=True)


class SpeechTextServer:
    def __init__(self, host: str, port: int, verbose: bool = False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self._clients: list[Queue[str | None]] = []
        self._clients_lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = threading.Event()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._serve, daemon=True)
        thread.start()
        self._ready.wait(timeout=3)
        return thread

    def publish(self, text: str) -> None:
        with self._clients_lock:
            clients = list(self._clients)

        if not clients:
            self._log(f"No speech clients connected; dropped: {text}")
            return

        for client in clients:
            client.put(text)

    def close(self) -> None:
        self._closed.set()
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            client.put(None)

    def _serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(0.5)
            self._ready.set()
            self._log(f"Speech text server listening on {self.host}:{self.port}")

            while not self._closed.is_set():
                try:
                    client_socket, address = server.accept()
                except socket.timeout:
                    continue

                self._log(f"Speech client connected from {address[0]}:{address[1]}")
                client_queue: Queue[str | None] = Queue()
                with self._clients_lock:
                    self._clients.append(client_queue)
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_queue),
                    daemon=True,
                ).start()

    def _handle_client(
        self,
        client_socket: socket.socket,
        client_queue: Queue[str | None],
    ) -> None:
        try:
            with client_socket:
                while not self._closed.is_set():
                    text = client_queue.get()
                    if text is None:
                        break
                    client_socket.sendall(f"{text}\n".encode("utf-8"))
        except OSError as exc:
            self._log(f"Speech client disconnected: {exc}")
        finally:
            with self._clients_lock:
                if client_queue in self._clients:
                    self._clients.remove(client_queue)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[recognize_speech] {message}", file=sys.stderr, flush=True)


def list_mics() -> int:
    print("PulseAudio/PipeWire sources:", file=sys.stderr)
    pulse_sources = subprocess.run(["pactl", "list", "short", "sources"], check=False)
    print("\nALSA capture hardware devices:", file=sys.stderr)
    alsa_hardware = subprocess.run(["arecord", "-l"], check=False)
    return pulse_sources.returncode or alsa_hardware.returncode


def parse_args() -> argparse.Namespace:
    defaults = SpeechConfig()
    parser = argparse.ArgumentParser(
        description="Continuously listen with Silero VAD and transcribe speech with faster-whisper."
    )
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument(
        "--record-backend",
        choices=["parecord", "arecord"],
        default=defaults.record_backend,
        help="Audio capture backend. parecord is recommended for Bluetooth mics.",
    )
    parser.add_argument(
        "--mic-device",
        default=defaults.mic_device,
        help=(
            "Input device/source. For parecord use a pactl source name. "
            "For arecord use an ALSA device such as 'default' or 'hw:1,0'. "
            "Omit this to use the system default input."
        ),
    )
    parser.add_argument("--stt-device", default=defaults.stt_device, help="faster-whisper device.")
    parser.add_argument("--compute-type", default=defaults.compute_type, help="faster-whisper compute type.")
    parser.add_argument("--language", help="Optional language code such as 'en' or 'ko'.")
    parser.add_argument("--beam-size", type=int, default=defaults.beam_size)
    parser.add_argument("--vad-threshold", type=float, default=defaults.vad_threshold)
    parser.add_argument("--vad-min-silence-ms", type=int, default=defaults.vad_min_silence_ms)
    parser.add_argument("--max-speech-seconds", type=float, default=defaults.max_speech_seconds)
    parser.add_argument(
        "--show-level",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_level,
        help="Show a live microphone level meter while listening.",
    )
    parser.add_argument(
        "--level-interval-ms",
        type=int,
        default=defaults.level_interval_ms,
        help="Refresh interval for --show-level.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run a localhost speech text server for inference_stt.py.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=defaults.verbose,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_mics:
        return list_mics()

    recognizer = SpeechRecognizer(
        SpeechConfig(
            model_path=args.model_path,
            record_backend=args.record_backend,
            mic_device=args.mic_device,
            stt_device=args.stt_device,
            compute_type=args.compute_type,
            language=args.language,
            beam_size=args.beam_size,
            vad_threshold=args.vad_threshold,
            vad_min_silence_ms=args.vad_min_silence_ms,
            max_speech_seconds=args.max_speech_seconds,
            show_level=args.show_level,
            level_interval_ms=args.level_interval_ms,
            verbose=args.verbose,
        )
    )

    server = None
    try:
        if args.server:
            server = SpeechTextServer(args.host, args.port, verbose=args.verbose)
            server.start()

        for text in recognizer.listen():
            if args.server:
                assert server is not None
                server.publish(text)
            else:
                print(text, flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.SubprocessError as exc:
        print(f"Audio capture failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
