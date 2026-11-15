# Compression-Research

[![CI](https://github.com/kapildhinwa2010-dev/Compression-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Compression-Research/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Compression-Research/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Compression-Research/actions/workflows/codeql.yml)

Compression trade-off experiments: rank sweeps, bit-depth sweeps, Pareto front.

## What it does

Beyond the fixed four-method benchmark, this module sweeps the two main
compression knobs and reports the trade-off tables:

- rank_sweep: SVD error and ratio per rank
- bits_sweep: quantization error and ratio per bit depth
- pareto_front: the operating points no other point beats in both error and ratio

    from compression_research import rank_sweep, pareto_front

    rows = rank_sweep(W, ranks=(1, 2, 4, 8))
    print(pareto_front(rows))

## API

- svd_compress / quantize / fourier_compress / hybrid_compress / benchmark
- rank_sweep(matrix, ranks) / bits_sweep(matrix, bits_list)
- pareto_front(points)

## Honest results and limitations

- Ratios are parameter-count estimates, not measured serialized sizes.
- The Pareto front only considers error and ratio; it knows nothing about
  downstream accuracy, speed or hardware behaviour.
- Sweeps use fixed candidate grids; the true optimum between two grid
  points is never evaluated.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
