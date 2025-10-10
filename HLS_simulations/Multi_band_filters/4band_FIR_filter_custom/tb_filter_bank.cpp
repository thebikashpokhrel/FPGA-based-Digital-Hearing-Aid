#include "filter_bank.h"
#include "hls_stream.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#if defined(_WIN32)
  #include <direct.h>
#endif

static const int BATCH = 8192;

static bool load_samples_decimals(const char* filename, std::vector<data_t>& left, std::vector<data_t>& right) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        std::printf("Cannot open input file: %s\n", filename);
        return false;
    }
    double l, r;
    while (fscanf(fp, "%lf %lf", &l, &r) == 2) {
        left.push_back((data_t)l);
        right.push_back((data_t)r);
    }
    fclose(fp);
    return !left.empty();
}

static void ensure_dir_for_file(const char* filepath) {
    std::string s(filepath);
    size_t pos = s.find_last_of("/\\");
    if (pos == std::string::npos) return;
    std::string dir = s.substr(0, pos);
#if defined(_WIN32)
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
}

int main(int argc, char** argv) {
    const char* input_file  = (argc > 1) ? argv[1] : "samples_stereo.txt";
    const char* output_file = (argc > 2) ? argv[2] : "filtered_stereo.txt";

    std::vector<data_t> left, right;
    if (!load_samples_decimals(input_file, left, right)) {
        std::printf("No input samples loaded from %s\n", input_file);
        return 2;
    }

    // int total_samples = (int)left.size();
    int total_samples = 30000;
    std::printf("Loaded %d stereo samples from %s\n", total_samples, input_file);

    // Print a few middle samples to verify loading correctness
    if (total_samples > 10) {
        int mid = total_samples / 2;
        std::printf("Middle input samples around index %d:\n", mid);
        for (int i = mid - 3; i <= mid + 3; ++i) {
            if (i >= 0 && i < total_samples)
                std::printf("Index %6d: Left = %.15f, Right = %.15f\n", 
                            i, (double)left[i], (double)right[i]);
        }
    } else {
        std::printf("Input too short (%d samples) to show middle samples.\n", total_samples);
    }

    std::vector<data_t> out_left;
    std::vector<data_t> out_right;
    out_left.reserve(total_samples);
    out_right.reserve(total_samples);

    int offset = 0;
    while (offset < total_samples) {
        hls::stream<data_t> in_l, in_r, out_l, out_r;

        for (int i = 0; i < BATCH; ++i) {
            int idx = offset + i;
            if (idx < total_samples) {
                in_l.write(left[idx]);
                in_r.write(right[idx]);
            } else {
                in_l.write((data_t)0);
                in_r.write((data_t)0);
            }
        }

        filter_bank_fourbands(in_l, in_r, out_l, out_r, BATCH);

        for (int i = 0; i < BATCH; ++i) {
            data_t ol = out_l.read();
            data_t orr = out_r.read();
            int idx = offset + i;
            if (idx < total_samples) {
                out_left.push_back(ol);
                out_right.push_back(orr);
            }
        }

        offset += BATCH;
        std::printf("Processed up to sample %d / %d\n", std::min(offset, total_samples), total_samples);
    }

    if ((int)out_left.size() != total_samples || (int)out_right.size() != total_samples) {
        std::printf("WARNING: output sample count (%d,%d) != input sample count (%d,%d)\n",
            (int)out_left.size(), (int)out_right.size(), (int)total_samples, (int)total_samples);
    }

    ensure_dir_for_file(output_file);

    FILE* fo = fopen(output_file, "w");
    if (!fo) {
        std::printf("ERROR: Cannot open output file %s\n", output_file);
        return 3;
    }

    for (int i = 0; i < (int)out_left.size(); ++i) {
        fprintf(fo, "%.15f %.15f\n", (double)out_left[i], (double)out_right[i]);
    }
    fclose(fo);

    std::printf("First 8 output sample pairs (left right):\n");
    for (int i = 0; i < 8 && i < (int)out_left.size(); ++i) {
        std::printf("%2d: %.15f %.15f\n", i, (double)out_left[i], (double)out_right[i]);
    }

    std::printf("Filtered output written to %s (samples: %d)\n", output_file, (int)out_left.size());
    return 0;
}