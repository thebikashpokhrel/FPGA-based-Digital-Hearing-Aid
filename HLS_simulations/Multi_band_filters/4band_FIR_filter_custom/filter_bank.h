#ifndef FILTER_BANK_H
#define FILTER_BANK_H

#include "ap_fixed.h"
#include "hls_stream.h"

typedef ap_fixed<16, 1> data_t;
typedef ap_fixed<16, 1> coeff_t;
typedef ap_fixed<32, 2> out_t;

const int NUM_BANDS = 4;
static const int coeff_counts[NUM_BANDS] = {163, 131, 67, 23};
const int MAX_COEFFS = 163;
const int BAND_TO_BOOST = 3;
const double TARGET_GAIN = 10.0;

void filter_bank_fourbands(
    hls::stream<data_t> &input_left,
    hls::stream<data_t> &input_right,
    hls::stream<data_t> &output_left,
    hls::stream<data_t> &output_right,
    int num_samples);

#endif