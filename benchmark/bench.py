import api_bench
import fir_bench
import iir_bench

if __name__ == "__main__":
    with open("bench.out", "w") as f:
        print("Starting benchmarks...", file=f)
        print("FIR Benchmark:", file=f)
        fir_bench.start(f)
        print("\nIIR Benchmark:", file=f)
        iir_bench.start(f)
        print("\nAPI Benchmark:", file=f)
        api_bench.start(f)
