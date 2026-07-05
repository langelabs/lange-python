"""Multi-domain LLM benchmark runner."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

from openai import OpenAI


class BenchmarkTestResult(TypedDict):
    """Serialized result for a single benchmark test."""

    test: str
    time: float
    result: str


class BenchmarkResult(TypedDict):
    """Serialized result for one benchmark run."""

    name: str
    version: str
    overall_time: float
    base_url: str
    model: str
    tests: list[BenchmarkTestResult]


class LLMBenchmark:
    """Run a multi-domain benchmark against an OpenAI-compatible LLM server."""

    def __init__(
        self,
        base_url: str,
        name: str,
        version: str,
        model_id: str,
    ) -> None:
        """Create a benchmark runner.

        :param base_url: Base URL for the OpenAI-compatible API.
        :param name: Human-readable benchmark target name.
        :param version: Version label for the benchmark target.
        :param model_id: Model identifier passed to the chat completion API.
        """
        self.base_url = base_url
        self.name = name
        self.version = version
        self.model_id = model_id

        benchmark_dir = Path(__file__).resolve().parent
        self.data_dir = benchmark_dir / "data"
        self.results_dir = benchmark_dir / "results"
        self.result_path = self.results_dir / f"{name}-{version}.json"

    def __call__(self) -> BenchmarkResult:
        """Execute all benchmark tests and persist a JSON result file.

        :return: Serialized benchmark result.
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OPENAI_API_KEY", "benchmark"),
        )
        start_time = time.perf_counter()
        tests = [
            self._run_timed_test("chat", lambda: self._run_chat_test(client)),
            self._run_timed_test("email", lambda: self._run_email_test(client)),
            self._run_timed_test("image", lambda: self._run_image_test(client)),
            self._run_timed_test("audio", lambda: self._run_audio_test(client)),
            self._run_timed_test(
                "manifest",
                lambda: self._run_manifest_test(client),
            ),
        ]
        result: BenchmarkResult = {
            "name": self.name,
            "version": self.version,
            "overall_time": time.perf_counter() - start_time,
            "base_url": self.base_url,
            "model": self.model_id,
            "tests": tests,
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result

    def _run_timed_test(
        self,
        test_name: str,
        run_test: Callable[[], str],
    ) -> BenchmarkTestResult:
        """Run one benchmark test and capture runtime plus failures.

        :param test_name: Name stored in the serialized result.
        :param run_test: Callable that executes the benchmark request.
        :return: Serialized single-test result.
        """
        print(f"Starting benchmark test: {test_name}")
        start_time = time.perf_counter()

        try:
            result = run_test()
        except Exception as exc:  # noqa: BLE001 - benchmark must continue.
            result = f"ERROR: {exc}"
        finally:
            print(f"Finished benchmark test: {test_name}")

        return {
            "test": test_name,
            "time": time.perf_counter() - start_time,
            "result": result,
        }

    def _run_chat_test(self, client: OpenAI) -> str:
        """Ask the model to generate the next assistant message in a chat.

        :param client: OpenAI client used for the request.
        :return: Assistant response text.
        """
        chat_data = json.loads(
            (self.data_dir / "chat.json").read_text(encoding="utf-8")
        )
        response = client.chat.completions.create(
            model=self.model_id,
            messages=chat_data["messages"],
        )
        return self._extract_completion_text(response)

    def _run_email_test(self, client: OpenAI) -> str:
        """Extract structured event facts from the email fixture.

        :param client: OpenAI client used for the request.
        :return: JSON text returned by the model.
        """
        email_html = (self.data_dir / "email.html").read_text(encoding="utf-8")
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract event facts from HTML email. Return one "
                        "compact JSON object. Required keys: event_title, "
                        "date, time, location, format, speaker, organizer, "
                        "confirmation_deadline. Use null only if the value is "
                        "not present. Do not return an empty object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract the event facts from this HTML email:\n\n"
                        f"{email_html}"
                    ),
                },
            ],
        )
        return self._extract_completion_text(response)

    def _run_image_test(self, client: OpenAI) -> str:
        """Ask the model to describe the benchmark image fixture.

        :param client: OpenAI client used for the request.
        :return: Assistant response text.
        """
        image_data_url = self._read_file_as_data_url(
            self.data_dir / "where_is.png",
            mime_type="image/png",
        )
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image concisely.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
        )
        return self._extract_completion_text(response)

    def _run_audio_test(self, client: OpenAI) -> str:
        """Ask the model to describe the benchmark audio fixture.

        :param client: OpenAI client used for the request.
        :return: Assistant response text.
        """
        audio_data = self._read_file_as_base64(self.data_dir / "birds.mp3")
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this audio concisely.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": "mp3",
                            },
                        },
                    ],
                }
            ],
        )
        return self._extract_completion_text(response)

    def _run_manifest_test(self, client: OpenAI) -> str:
        """Ask the model to summarize the manifesto fixture.

        :param client: OpenAI client used for the request.
        :return: Assistant response text.
        """
        manifest = (self.data_dir / "manifest.txt").read_text(encoding="utf-8")
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the provided manifesto concisely.",
                },
                {
                    "role": "user",
                    "content": manifest,
                },
            ],
        )
        return self._extract_completion_text(response)

    @staticmethod
    def _read_file_as_base64(path: Path) -> str:
        """Read a file and return its base64-encoded contents.

        :param path: File path to encode.
        :return: Base64-encoded file contents.
        """
        return base64.b64encode(path.read_bytes()).decode("ascii")

    @classmethod
    def _read_file_as_data_url(cls, path: Path, mime_type: str) -> str:
        """Read a file and return a data URL.

        :param path: File path to encode.
        :param mime_type: MIME type for the data URL.
        :return: Data URL containing the file contents.
        """
        return f"data:{mime_type};base64,{cls._read_file_as_base64(path)}"

    @staticmethod
    def _extract_completion_text(response: Any) -> str:
        """Extract assistant text from an OpenAI chat completion response.

        :param response: OpenAI SDK chat completion response.
        :return: Assistant message content.
        """
        content = response.choices[0].message.content

        if content is None:
            return ""

        return str(content)
