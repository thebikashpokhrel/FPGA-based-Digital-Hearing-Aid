#ifndef FILTER_BANK_H
#define FILTER_BANK_H

#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_fir.h"

// Use fixed-point types that match Q1.15 and Q2.30 formats
typedef ap_fixed<16, 1>  data_t;    // Q1.15 (1 integer bit -> sign + 15 frac)
typedef ap_fixed<16, 1>  coeff_t;   // coefficient same Q1.15
typedef ap_fixed<32, 2>  fir_out_t; // Q2.30 (2 integer bits + 30 frac) for accumulation

const int NUM_BANDS = 8;

// Top-level function declaration
void filter_bank(
    hls::stream<data_t>& input_left,
    hls::stream<data_t>& input_right,
    hls::stream<data_t>& output_left,
    hls::stream<data_t>& output_right,
    int num_samples
);

#endif
