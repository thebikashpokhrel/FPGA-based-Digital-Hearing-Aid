#include "filter_bank.h"

typedef ap_fixed<64,20> acc_t; 

// coefficient storage for all bands
const ap_fixed<16,1> band_coeffs[NUM_BANDS][MAX_COEFFS] = {
    {
#include "coeffs_band1_init.h"
    },
    {
#include "coeffs_band2_init.h"
    },
    {
#include "coeffs_band3_init.h"
    },
    {
#include "coeffs_band4_init.h"
    }
};

void filter_bank_fourbands(
    hls::stream<data_t>& input_left,
    hls::stream<data_t>& input_right,
    hls::stream<data_t>& output_left,
    hls::stream<data_t>& output_right,
    int num_samples)
{
    #pragma HLS INTERFACE axis port=input_left
    #pragma HLS INTERFACE axis port=input_right
    #pragma HLS INTERFACE axis port=output_left
    #pragma HLS INTERFACE axis port=output_right
    #pragma HLS INTERFACE s_axilite port=num_samples bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // #pragma HLS ALLOCATION instances=mul limit=16 operation
    
    // Band gains
    const ap_fixed<18,4> BOOST_LINEAR = 10.0;
    const ap_fixed<18,4> band_gain[NUM_BANDS] = {
        1.0, 1.0, BOOST_LINEAR, 1.0
    };

    #pragma HLS ARRAY_PARTITION variable=band_gain complete
    
    const int band_taps[NUM_BANDS] = {163, 131, 67,23};

    #pragma HLS ARRAY_PARTITION variable=band_taps complete
    
    // Delay buffers - complete partitioning on band dimension only
    static data_t delay_left[NUM_BANDS][MAX_COEFFS];
    static data_t delay_right[NUM_BANDS][MAX_COEFFS];
    static int write_ptr[NUM_BANDS] = {0};
    
    #pragma HLS ARRAY_PARTITION variable=delay_left complete dim=1
    #pragma HLS ARRAY_PARTITION variable=delay_right complete dim=1
    #pragma HLS BIND_STORAGE variable=delay_left type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=delay_right type=ram_2p impl=bram
    
    // NO partitioning on dim=2 to avoid memory port explosion
    
    const acc_t max_val = 0.99996;
    const acc_t min_val = -1.0;
    
    SAMPLE_LOOP: for (int s = 0; s < num_samples; ++s) {
        #pragma HLS LOOP_TRIPCOUNT min=1024 max=50000
        
        data_t in_l = input_left.read();
        data_t in_r = input_right.read();
        
        acc_t sum_l = 0;
        acc_t sum_r = 0;
        
        // Process all bands sequentially no inner pipeline
        BAND_LOOP: for (int b = 0; b < NUM_BANDS; ++b) {
            int wp = write_ptr[b];
            int n_taps = band_taps[b];
            
            // Write new sample to circular buffer
            delay_left[b][wp] = in_l;
            delay_right[b][wp] = in_r;
            
            acc_t acc_l = 0;
            acc_t acc_r = 0;
            
            // NO UNROLLING to save DSPs
            MAC_LOOP: for (int i = 0; i < n_taps; ++i) {
                // Calculate read index for circular buffer
                int idx = wp - i;
                if (idx < 0) idx += MAX_COEFFS;
                
                // Single MAC per iteration - sequential through taps
                acc_l += (acc_t)delay_left[b][idx] * (acc_t)band_coeffs[b][i];
                acc_r += (acc_t)delay_right[b][idx] * (acc_t)band_coeffs[b][i];
            }
            
            // Advance circular pointer
            wp = wp + 1;
            if (wp >= MAX_COEFFS) wp = 0;
            write_ptr[b] = wp;
            
            // Apply gain and accumulate
            acc_t gain = (acc_t)band_gain[b];
            sum_l += acc_l * gain;
            sum_r += acc_r * gain;
        }
        
        // Clamp output
        if (sum_l > max_val) sum_l = max_val;
        else if (sum_l < min_val) sum_l = min_val;
        
        if (sum_r > max_val) sum_r = max_val;
        else if (sum_r < min_val) sum_r = min_val;
        
        output_left.write((data_t)sum_l);
        output_right.write((data_t)sum_r);
    }
}