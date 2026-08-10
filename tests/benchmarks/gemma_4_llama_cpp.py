"""Benchmark instance for the manual Gemma 4 MLX VLM inference server."""

from benchmark import LLMBenchmark


if __name__ == "__main__":
    benchmark = LLMBenchmark(
        base_url="http://127.0.0.1:8080/v1",
        name="gemma-4-12b-it-llama_cpp",
        version="0",
        model_id="LL_C_0",
    )
    result = benchmark()
    print(f"Saved benchmark result to {benchmark.result_path}")
    print(f"Completed {len(result['tests'])} tests in {result['overall_time']:.3f}s.")
