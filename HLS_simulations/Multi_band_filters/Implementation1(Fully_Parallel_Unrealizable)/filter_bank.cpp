#include "filter_bank.h"

// Configuration structures for each band
// coeff_vec is declared as double because hls::FIR expects const double*

struct config_band0 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 163;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band1 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 163;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band2 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 147;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band3 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 75;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band4 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 75;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band5 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 75;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band6 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 39;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

struct config_band7 : hls::ip_fir::params_t {
    static const unsigned num_coeffs = 75;
    static const double coeff_vec[num_coeffs];
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 15;
    static const unsigned output_width = 32;
    static const unsigned output_fractional_bits = 30;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 15;
    static const unsigned input_length = 8192;
    static const unsigned output_length = 8192;
    static const unsigned coeff_structure = hls::ip_fir::symmetric;
    static const unsigned quantization = hls::ip_fir::quantize_only;
    static const unsigned output_rounding_mode =
        hls::ip_fir::convergent_rounding_to_even;
};

// initialize coefficient arrays using numeric lists from included headers

const double config_band0::coeff_vec[config_band0::num_coeffs] = {
#include "coeffs_band1_init.h"
};

const double config_band1::coeff_vec[config_band1::num_coeffs] = {
#include "coeffs_band2_init.h"
};

const double config_band2::coeff_vec[config_band2::num_coeffs] = {
#include "coeffs_band3_init.h"
};

const double config_band3::coeff_vec[config_band3::num_coeffs] = {
#include "coeffs_band4_init.h"
};

const double config_band4::coeff_vec[config_band4::num_coeffs] = {
#include "coeffs_band5_init.h"
};

const double config_band5::coeff_vec[config_band5::num_coeffs] = {
#include "coeffs_band6_init.h"
};

const double config_band6::coeff_vec[config_band6::num_coeffs] = {
#include "coeffs_band7_init.h"
};

const double config_band7::coeff_vec[config_band7::num_coeffs] = {
#include "coeffs_band8_init.h"
};

// Top-level filter bank - uses data_t and fir_out_t streams
void filter_bank(
    hls::stream<data_t>& input_left,
    hls::stream<data_t>& input_right,
    hls::stream<data_t>& output_left,
    hls::stream<data_t>& output_right,
    int num_samples) {

    #pragma HLS INTERFACE axis port = input_left
    #pragma HLS INTERFACE axis port = input_right
    #pragma HLS INTERFACE axis port = output_left
    #pragma HLS INTERFACE axis port = output_right
    #pragma HLS INTERFACE s_axilite port = num_samples bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    const int BAND_TO_BOOST = 5;  // 5th band to apply gain
    const double BOOST_LINEAR = 30.0; // Gain of 30

    // compile-time fixed-point gains per band
    ap_fixed<64, 6> band_gain[NUM_BANDS];
    for (int b = 0; b < NUM_BANDS; ++b) {
        if (b == BAND_TO_BOOST)
            band_gain[b] = (ap_fixed<64, 6>)BOOST_LINEAR;
        else
            band_gain[b] = (ap_fixed<64, 6>)1.0;
    }

    // Streams for each band
    hls::stream<data_t> band_in_left[NUM_BANDS];
    hls::stream<fir_out_t> band_out_left[NUM_BANDS];
    hls::stream<data_t> band_in_right[NUM_BANDS];
    hls::stream<fir_out_t> band_out_right[NUM_BANDS];

    // Demux inputs to all bands
    for (int i = 0; i < num_samples; i++) {
    #pragma HLS PIPELINE II=1
        data_t l = input_left.read();
        data_t r = input_right.read();
        for (int j = 0; j < NUM_BANDS; j++) {
        #pragma HLS UNROLL
            band_in_left[j].write(l);
            band_in_right[j].write(r);
        }
    }

    // FIR filter instances
    static hls::FIR<config_band0> fir_left_0,  fir_right_0;
    static hls::FIR<config_band1> fir_left_1,  fir_right_1;
    static hls::FIR<config_band2> fir_left_2,  fir_right_2;
    static hls::FIR<config_band3> fir_left_3,  fir_right_3;
    static hls::FIR<config_band4> fir_left_4,  fir_right_4;
    static hls::FIR<config_band5> fir_left_5,  fir_right_5;
    static hls::FIR<config_band6> fir_left_6,  fir_right_6;
    static hls::FIR<config_band7> fir_left_7,  fir_right_7;

    // Run FIRs
    fir_left_0.run(band_in_left[0], band_out_left[0]);
    fir_left_1.run(band_in_left[1], band_out_left[1]);
    fir_left_2.run(band_in_left[2], band_out_left[2]);
    fir_left_3.run(band_in_left[3], band_out_left[3]);
    fir_left_4.run(band_in_left[4], band_out_left[4]);
    fir_left_5.run(band_in_left[5], band_out_left[5]);
    fir_left_6.run(band_in_left[6], band_out_left[6]);
    fir_left_7.run(band_in_left[7], band_out_left[7]);

    fir_right_0.run(band_in_right[0], band_out_right[0]);
    fir_right_1.run(band_in_right[1], band_out_right[1]);
    fir_right_2.run(band_in_right[2], band_out_right[2]);
    fir_right_3.run(band_in_right[3], band_out_right[3]);
    fir_right_4.run(band_in_right[4], band_out_right[4]);
    fir_right_5.run(band_in_right[5], band_out_right[5]);
    fir_right_6.run(band_in_right[6], band_out_right[6]);
    fir_right_7.run(band_in_right[7], band_out_right[7]);

    // Sum all bands and write outputs
    for (int i = 0; i < num_samples; i++) {
    #pragma HLS PIPELINE II=1
        ap_fixed<64, 6> sum_l = 0;
        ap_fixed<64, 6> sum_r = 0;

        for (int b = 0; b < NUM_BANDS; b++) {
        #pragma HLS UNROLL
            ap_fixed<64, 6> out_l = (ap_fixed<64, 6>)band_out_left[b].read();
            ap_fixed<64, 6> out_r = (ap_fixed<64, 6>)band_out_right[b].read();

            // Apply per-band gain
            sum_l += out_l * band_gain[b];
            sum_r += out_r * band_gain[b];
        }

        // Saturate/clamp to data_t range
        const ap_fixed<64, 6> max_val = (ap_fixed<64, 6>)0.99996;
        const ap_fixed<64, 6> min_val = (ap_fixed<64, 6>)-1.0;

        if (sum_l > max_val) sum_l = max_val;
        if (sum_l < min_val) sum_l = min_val;
        if (sum_r > max_val) sum_r = max_val;
        if (sum_r < min_val) sum_r = min_val;

        output_left.write((data_t)sum_l);
        output_right.write((data_t)sum_r);
    }
}