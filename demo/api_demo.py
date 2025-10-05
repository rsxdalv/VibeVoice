import argparse
import base64
import time
from pathlib import Path
from typing import Any

from gradio_client import Client


DEFAULT_PARAMS = {
	"num_speakers": 2,
	"script": (
		"Speaker 0: Welcome back to the VibeVoice quick demo!\n"
		"Speaker 1: Today we'll give a speedy status update on the pipeline.\n"
		"Speaker 0: The new streaming API is humming along and the poll loop feels snappy.\n"
		"Speaker 1: Next on the roadmap is polishing the voice selector and presets."
	),
	"param_2": "en-Alice_woman",
	"param_3": "en-Carter_man",
	"param_4": "en-Frank_man",
	"param_5": "en-Maya_woman",
	"param_6": 1.3,
}


def decode_payload_to_file(payload: dict, destination: Path) -> Path:
	audio_bytes = base64.b64decode(payload["base64_wav"])
	destination.parent.mkdir(parents=True, exist_ok=True)
	destination.write_bytes(audio_bytes)
	return destination


def run_full_generation(client: Client, output_dir: Path, params: dict[str, Any]) -> tuple[bytes, dict[str, int | str], str]:
	print("Requesting full podcast audio...")
	payload, generation_log = client.predict(
		api_name="/generate_podcast_full",
		**params,
	)

	if not isinstance(payload, dict) or "base64_wav" not in payload:
		raise ValueError("Full-generation API returned an unexpected payload format.")

	audio_bytes = base64.b64decode(payload["base64_wav"])
	metadata: dict[str, int | str] = {
		"sample_rate": int(payload.get("sample_rate", 0) or 0),
		"num_samples": int(payload.get("num_samples", 0) or 0),
	}

	output_dir.mkdir(parents=True, exist_ok=True)
	destination = output_dir / "full_generation.wav"
	destination.write_bytes(audio_bytes)
	metadata["file_path"] = str(destination)
	metadata["byte_length"] = len(audio_bytes)

	print(f"Saved complete audio to: {destination}")
	print("Generation log:\n", generation_log)

	return audio_bytes, metadata, generation_log


def run_streaming_session(client: Client, output_dir: Path, poll_interval: float, params: dict[str, Any]) -> None:
	print("Starting streaming session...")
	session_id = client.predict(api_name="/stream_start", **params)
	print(f"Session ID: {session_id}")

	chunk_dir = output_dir / "stream_chunks"
	chunk_index = 0
	final_path = output_dir / "final_stream.wav"
	last_log = ""

	while True:
		response = client.predict(session_id, api_name="/stream_poll")
		status = response.get("status")
		updates = response.get("updates", [])
		last_log = response.get("log", last_log)

		for update in updates:
			u_type = update.get("type")
			payload = update.get("payload")
			if not payload:
				continue

			if u_type == "chunk":
				chunk_index += 1
				chunk_path = chunk_dir / f"chunk_{chunk_index:03d}.wav"
				decode_payload_to_file(payload, chunk_path)
				print(f"Received chunk {chunk_index} ({payload['num_samples']} samples)")
			elif u_type == "final":
				decode_payload_to_file(payload, final_path)
				print(f"Received final audio -> {final_path}")

		if status in {"completed", "failed", "not_found"}:
			if status == "failed":
				print("Session failed:", response.get("error"))
			break

		if not updates:
			time.sleep(poll_interval)

	print("Streaming session status:", status)
	if status != "completed":
		stop_response = client.predict(session_id, api_name="/stream_stop")
		print("Stop response:", stop_response)

	if last_log:
		print("\nFinal generation log:\n", last_log)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Demo client for VibeVoice APIs")
	parser.add_argument(
		"--server-url",
		default="http://127.0.0.1:7860/",
		help="Base URL where the Gradio server is running",
	)
	parser.add_argument(
		"--mode",
		choices=["full", "stream", "both"],
		default="full",
		help="Which demo to run",
	)
	parser.add_argument(
		"--speakers",
		nargs="+",
		help="Optional list of speaker preset keys (up to 4, order matters)",
	)
	parser.add_argument(
		"--num-speakers",
		type=int,
		help="Override number of speakers (1-4). Defaults to inferred value when specifying speakers.",
	)
	parser.add_argument(
		"--cfg-scale",
		type=float,
		help="Override CFG scale (param_6)",
	)
	parser.add_argument(
		"--script-text",
		help="Inline script text. Takes precedence over --script-file when provided.",
	)
	parser.add_argument(
		"--script-file",
		type=Path,
		help="Optional path to a text file containing the dialogue script",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("demo_outputs"),
		help="Directory to store streamed WAV chunks",
	)
	parser.add_argument(
		"--poll-interval",
		type=float,
		default=2.0,
		help="Seconds to wait between streaming polls when idle",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	client = Client(args.server_url)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	request_params: dict[str, Any] = dict(DEFAULT_PARAMS)
	voice_param_keys = ["param_2", "param_3", "param_4", "param_5"]

	if args.script_text is not None:
		script_text = args.script_text.replace("\\n", "\n")
		if not script_text.strip():
			raise ValueError("Provided --script-text is empty after trimming.")
		req_lines = len([line for line in script_text.splitlines() if line.strip()])
		request_params["script"] = script_text
		print(
			f"Loaded inline script ({req_lines} non-empty lines, {len(script_text)} characters)"
		)

	elif args.script_file:
		if not args.script_file.exists():
			raise FileNotFoundError(f"Script file not found: {args.script_file}")
		script_text = args.script_file.read_text(encoding="utf-8")
		if not script_text.strip():
			raise ValueError(f"Script file '{args.script_file}' is empty")
		req_lines = len(script_text.splitlines())
		request_params["script"] = script_text
		print(
			f"Loaded custom script from {args.script_file} "
			f"({req_lines} lines, {len(script_text)} characters)"
		)

	if args.cfg_scale is not None:
		request_params["param_6"] = args.cfg_scale
		print(f"CFG scale override: {args.cfg_scale}")

	if args.speakers:
		if len(args.speakers) > len(voice_param_keys):
			raise ValueError("Specify at most 4 speakers.")
		for idx, key in enumerate(voice_param_keys):
			if idx < len(args.speakers):
				request_params[key] = args.speakers[idx]
			else:
				break
		request_params["num_speakers"] = len(args.speakers)
		print(f"Using speakers: {', '.join(args.speakers)}")

	if args.num_speakers is not None:
		if not (1 <= args.num_speakers <= 4):
			raise ValueError("--num-speakers must be between 1 and 4.")
		if args.speakers and args.num_speakers != len(args.speakers):
			raise ValueError("--num-speakers must match the number of values passed to --speakers.")
		request_params["num_speakers"] = args.num_speakers
		print(f"Number of speakers override: {args.num_speakers}")

	if args.mode in {"full", "both"}:
		audio_bytes, metadata, _ = run_full_generation(client, args.output_dir, request_params)
		sample_rate = metadata.get("sample_rate")
		print(
			f"Retrieved {len(audio_bytes)} raw bytes"
			+ (f" at {sample_rate} Hz" if sample_rate else "")
		)

	if args.mode in {"stream", "both"}:
		run_streaming_session(client, args.output_dir, args.poll_interval, request_params)


if __name__ == "__main__":
	main()