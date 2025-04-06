import api_bench
import fir_bench
import iir_bench

if __name__ == "__main__":
    print("Starting benchmarks...")
    print("FIR Benchmark:")
    fir_bench.start()
    print("\nIIR Benchmark:")
    iir_bench.start()
    print("\nAPI Benchmark:")
    api_bench.start()
