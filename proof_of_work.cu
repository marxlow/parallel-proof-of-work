#include <iostream>
#include <bitset>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <ctime>

#include "hash.h" // hash.h must be in the current directory
using namespace std;

__device__ std::bitset<352> prepend_bit_device;

// Helper functions
const char* hex_char_to_bin(char c) {
    switch(toupper(c)) {
        case '0': return "0000";
        case '1': return "0001";
        case '2': return "0010";
        case '3': return "0011";
        case '4': return "0100";
        case '5': return "0101";
        case '6': return "0110";
        case '7': return "0111";
        case '8': return "1000";
        case '9': return "1001";
        case 'A': return "1010";
        case 'B': return "1011";
        case 'C': return "1100";
        case 'D': return "1101";
        case 'E': return "1110";
        case 'F': return "1111";
    }
    return "0000";
}

std::string hex_str_to_bin_str(const std::string& hex)
{
    // TODO use a loop from <algorithm> or smth
    std::string bin;
    for(unsigned i = 0; i != hex.length(); ++i)
       bin += hex_char_to_bin(hex[i]);
    return bin;
}

// Parallel code/functions
__global__ void find_nonce(std::bitset<352>prepend_bit) {
    printf("Thread ID: %d\n", threadIdx.x);
    // Step 1: Calculate the value of X which is 416 bit in length and defined as:
    // (416, 384] t --> Unix timestamp (seconds since UNIX epoch), unsigned 32-bit number.
    // [383, 128] previous_digest --> 256 bits given as input
    // [127, 64] id -->  NUSNET ID "E0003049" in char representation
    // [63, 0] nonce --> unsigned 64-bit number. Can be in the range of [2^64 - 1, 0]. This is what we have to find
    // Each thread will be calculating 1 nonce.
    long nonce = blockIdx.x * blockDim.x + threadIdx.x;
    std::bitset<64> nonce_bit(nonce);
    std::bitset<416> x(prepend_bit_device + nonce_bit);
    // Step 2: Hash SHA256(X)
    //__device__ void sha256(uint8_t hash[32], const uint8_t * input, size_t len);
    uint8_t hash_res[32]; 
    hash.sha256(hash_res, x, sizeof(x));

    // Step 3: Get first 64-bits of the digest SHA256(X)
    
    // Step 4: Compare with "n" to see if it can be accepted
}

int main(int argc, char **argv) {
    
    // Initialize x with 416 bits of 0s
    std::bitset<416> x;
    
    printf("~~~~~~~~~~ Calculating proof of work ~~~~~~~~~~ \n");
    printf("> Reading input file.... '%s'\n", argv[1]);
    std::ifstream file(argv[1]);
    if (file.is_open()) {
        // Read odd lines (1, 3, 5...) as digest
        // Read even lines(2, 4, 6...) as n in decimal form
        std::string previous_digest;
        while (getline(file, previous_digest)) {
            std::string n_decimal;
            getline(file, n_decimal);
            
            printf("> Pre-processing digest & decimal\n");
            // Convert Unix timestamp of time now (seconds since UNIX epoch) to  bit array
            std::time_t time_now = std::time(0); // unsigned(?) 32-bit integer
            std::bitset<32> t_bit(time_now);

            // Convert previous_digest in hex format(Each hex value corresponds to 4-bits) to bit array
            std::string previous_digest_binary_string = hex_str_to_bin_str(previous_digest);
            std::bitset<256> previous_digest_bit(string(previous_digest_binary_string.c_str()));
            
            // Convert NUSNET ID to bit array
            std::string nus_net_id = "E0003049";
            std::string nus_net_id_binary;
            for(int i = 0; i < nus_net_id.length(); i ++) {
                std::bitset<8> char_in_bit(nus_net_id[i]);
                nus_net_id_binary += char_in_bit.to_string();
            }
            std::bitset<64> id_bit(string(nus_net_id_binary.c_str()));
            
            // Concatenate "[t_bit] [previous_digest_bit] [id_bit]"
            std::bitset<352>prepend_bit(
                t_bit.to_string() + 
                previous_digest_bit.to_string() +
                id_bit.to_string()
            );
            rc = cudaMemcpyToSymbol(prepend_bit, prepend_bit_device, sizeof(prepend_bit_device));
            if (rc != cudaSuccess) {
                printf("Could not copy to device. Reason %s\n", cudaGetErrorString(rc));
            }
            // cout << t_bit << "\n";
            // cout << previous_digest_bit << "\n";
            // cout << id_bit << "\n";
            // cout << prepend_bit << "\n";
            printf("> Pre-processing done\n");
            printf("> Passing 352-bitset to threads to find nonce");
            // find_nonce<<<1, 1>>>(); // (num_thread_blocks, num_threads/block)
            // cudaDeviceSynchronize(); // Waits for all CUDA threads to complete.
        }
        file.close();
    }
    printf("~~~~~~~~~~ Done ~~~~~~~~~~ \n");
    return 0;
}