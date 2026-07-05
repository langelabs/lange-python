"""Benchmark instance for the manual Gemma 4 MLX VLM inference server."""

from benchmark import LLMBenchmark


if __name__ == "__main__":
    benchmark = LLMBenchmark(
        base_url="http://127.0.0.1:8080/v1",
        name="gemma-4-12b-it-mlx-vlm",
        version="2",
        model_id="mlx-community/gemma-4-12B-it-8bit",
    )
    result = benchmark()
    print(f"Saved benchmark result to {benchmark.result_path}")
    print(f"Completed {len(result['tests'])} tests in {result['overall_time']:.3f}s.")
