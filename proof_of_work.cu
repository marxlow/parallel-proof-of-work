#include <iostream>
#include <bitset>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <ctime>
#include <cmath>

#include "hash.h" // hash.h must be in the current directory
using namespace std;

__device__ uint8_t device_prepend_byte_array[44];
__device__ unsigned long long n;
// These 2 variables will only be set once an answer is found by a thread.
// Copy these values back into host.
__device__ long long nonce_answer;
__device__ uint8_t digest_answer[32];
__device__ bool found_res;


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
__global__ void find_nonce() {
    printf("Thread ID: %d\n", threadIdx.x);
    printf("The value here: %d\n", device_prepend_byte_array[0]);
    found_res = false;
    // Step 1: Calculate the value of X which is 416 bit in length and defined as:
    // (416, 384] t --> Unix timestamp (seconds since UNIX epoch), unsigned 32-bit number.
    // [383, 128] previous_digest --> 256 bits given as input
    // [127, 64] id -->  NUSNET ID "E0003049" in char representation
    // [63, 0] nonce --> unsigned 64-bit number. Can be in the range of [2^64 - 1, 0]. This is what we have to find
    long long nonce = blockDim.x * blockIdx.x + threadIdx.x;
    uint8_t nonce_bytes[8];
    int mask = 0xFF;
    // Reversed due to big endianness
    nonce_bytes[7] = (int)(nonce & mask);
    nonce_bytes[6] = (int)((nonce>>8) & mask);
    nonce_bytes[5] = (int)((nonce>>16) & mask);
    nonce_bytes[4] = (int)((nonce>>24) & mask);
    nonce_bytes[3] = (int)((nonce>>32) & mask);
    nonce_bytes[2] = (int)((nonce>>40) & mask);
    nonce_bytes[1] = (int)((nonce>>48) & mask);
    nonce_bytes[0] = (int)((nonce>>56) & mask);
    uint8_t x[52];
    for (int i = 0 ; i < 44; i++) {
        x[i] = device_prepend_byte_array[i];
    }
    for (int i = 44; i < 52; i++) {
        x[i] = nonce_bytes[i - 44];
    }
    // Step 2: Hash SHA256(X)
    //__device__ void hash.sha256(uint8_t hash[32], const uint8_t * input, size_t len);
    uint8_t hash_res[32]; 
    sha256(hash_res, x, sizeof(x));

    // Step 3: Get first 64-bits of the digest SHA256(X)
    unsigned long long digest = 0; 
    unsigned long long hash_res_long = (unsigned long long)(hash_res[0]);
    digest += hash_res_long<<56;
    hash_res_long = (unsigned long long)(hash_res[1]);
    digest += hash_res_long<<48;
    hash_res_long = (unsigned long long)(hash_res[2]);
    digest += hash_res_long<<40;
    hash_res_long = (unsigned long long)(hash_res[3]);
    digest += hash_res_long<<32;
    hash_res_long = (unsigned long long)(hash_res[4]);
    digest += hash_res_long<<24;  
    hash_res_long = (unsigned long long)(hash_res[5]);
    digest += hash_res_long<<16;  
    hash_res_long = (unsigned long long)(hash_res[6]);
    digest += hash_res_long<<8;  
    hash_res_long = (unsigned long long)(hash_res[7]);
    digest += hash_res_long;  
    printf("digest = %lu\n ", digest);
    
    // Step 4: Compare with "n" to see if it can be accepted
    // TODO(lowjiansheng): pass these values back to host
    if (digest < n) {
        found_res = true;
        nonce_answer = nonce;
        digest_answer = hash_res;
    }
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

            // Convert bitset to a uint_8t array
            uint8_t prepend_array[44];
            for (int i = 0; i < prepend_bit.size() - 7; i += 8) {
                int byte_value = 
                    prepend_bit.test(i) * pow(2, 7)+
                    prepend_bit.test(i+1) * pow(2, 6)+
                    prepend_bit.test(i+2) * pow(2, 5)+
                    prepend_bit.test(i+3) * pow(2, 4)+
                    prepend_bit.test(i+4) * pow(2, 3)+
                    prepend_bit.test(i+5) * pow(2, 2)+
                    prepend_bit.test(i+6) * pow(2, 1)+
                    prepend_bit.test(i+7);
                uint8_t byte_value_uint8 = byte_value;
                int index = i / 8;
                prepend_array[index] = byte_value_uint8;

            }
            printf("> Pre-processing done\n");
            cudaError_t rc = cudaMemcpyToSymbol(device_prepend_byte_array, &prepend_array, sizeof(device_prepend_byte_array));
            if (rc != cudaSuccess) {
                printf("Could not copy to device. Reason %s\n", cudaGetErrorString(rc));
            }
            printf("> Passing 352-bitset to threads to find nonce\n");
            // TODO(lowjiansheng): Find some way to check if a thread has found the answer. Once found terminate and return.
            find_nonce<<<1, 2>>>(); // (num_thread_blocks, num_threads/block)
            cudaDeviceSynchronize(); // Waits for all CUDA threads to complete.
        }
        file.close();
    }
    printf("~~~~~~~~~~ Done ~~~~~~~~~~ \n");
    return 0;
}