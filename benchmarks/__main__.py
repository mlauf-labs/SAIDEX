"""Entry point for running benchmarks as a module.

This allows both:
  python -m benchmarks                    # recommended
  python -m benchmarks.run_benchmarks     # alternative
"""

from .run_benchmarks import main

if __name__ == "__main__":
    main()
