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


__device__ __managed__ uint8_t unified_prepend_byte_array[44];
// These  variables will only be set once an answer is found by a thread.
__device__ __managed__ unsigned long long unified_nonce_answer;
__device__ __managed__ uint8_t unified_digest_answer[32];
__device__ __managed__ bool unified_found_res;

__device__ __managed__ int num_thread_failure;
__device__ bool found_res = false;

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
__device__ void get_x(uint8_t x[52], unsigned long long nonce) {
    // (416, 384] t --> Unix timestamp (seconds since UNIX epoch), unsigned 32-bit number.
    // [383, 128] previous_digest --> 256 bits given as input
    // [127, 64] id -->  NUSNET ID "E0003049" in char representation
    // [63, 0] nonce --> unsigned 64-bit number. Can be in the range of [2^64 - 1, 0]. This is what we have to find
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
    for (int i = 44; i < 52; i++) {
        x[i] = nonce_bytes[i - 44];
    }
}

// From Lab
void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));
    }

}

__device__ unsigned long long hash_to_decimal(uint8_t hash_res[32]) {
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

    return digest;
}


__global__ void find_nonce(unsigned long long thread_search_space, unsigned long long n_decimal, bool runBlock) {
    // printf("Thread idx: %d | Block dim: %d | Block idx: %d\n", threadIdx.x, blockDim.x, blockIdx.x);
    // Return if res is already found to skip work
    if (found_res) {
        return;
    }

    // Initialization
    uint8_t x[52];
    for (int i = 0; i < 44; i ++) {
        x[i] = unified_prepend_byte_array[i];
    }
    unsigned long long thread_start_index;
    if (runBlock) {
        thread_start_index = (blockDim.x * blockIdx.x + threadIdx.x) * thread_search_space;
    } else {
        thread_start_index = threadIdx.x + (blockDim.x * blockIdx.x);
    }

    // Iterate through search space of each thread
    for (unsigned long long i = 0; i < thread_search_space; i++) {
        if (!found_res) {
            // Step 1: Form x with nonce which is thread_start_index + i. 
            unsigned long long next_index;
            if (runBlock) {
                next_index = thread_start_index + i;
            } else {
                next_index = thread_start_index + (i * blockDim.x * blockIdx.x);
            }
            get_x(x, next_index);
            
            // Step 2: Hash SHA256(X)
            uint8_t hash_res[32];
            sha256(hash_res, x, sizeof(x));

            // Step 3: Get first 64-bits of the digest SHA256(X)
            unsigned long long digest = hash_to_decimal(hash_res);

            // Step 4: Compare with "n" to see if it can be accepted
            if (digest < n_decimal && !found_res) {
                found_res = true;
                unified_found_res= true;
                unified_nonce_answer = next_index;
                for (int i = 0; i < 32; i ++) {
                    unified_digest_answer[i] = hash_res[i];
                }
                // Thread early exit
                return;
            }
        } else {
            return;
        }
    }

    // note: Will only reach here when no nonce is found for this thread at the end of search space.
    atomicAdd(&num_thread_failure, 1);
    printf("1 thread has failed.\n");
    return;
}

int main(int argc, char **argv) {
    // Initialize x with 416 bits of 0s
    std::bitset<416> x;

    printf("> Reading input file: '%s'...\n", argv[1]);
    
    std::ifstream file(argv[1]);
    if (file.is_open()) {
        // Read odd lines (1, 3, 5...) as digest
        // Read even lines(2, 4, 6...) as n in decimal form
        std::string previous_digest;
        while (getline(file, previous_digest)) {
            printf("\n\n~~~~~~~~~~ Calculating proof of work ~~~~~~~~~~ \n");
            std::string n_decimal_string;
            getline(file, n_decimal_string);
            
            printf("> Pre-processing digest & decimal...\n");
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

            // Convert string o usigned long long for "n"
            char* endptr = NULL;
            unsigned long long n_decimal = strtoll(n_decimal_string.c_str(), &endptr, 10);
            num_thread_failure = 0;
            printf("~~~~~~~~~~ Pre-processing done ~~~~~~~~~~ \n");

            // Copy over values to unified memory for CPU & GPU
            printf("> Copying over values from CPU to unified memory (shared & global)...\n");

            // Copy prepend_byte_array
            for (int i = 0; i < 44; i ++) {
                unified_prepend_byte_array[i] = prepend_array[i];
            }

            // Set unified_found_res to be false initially. Will be set to true after nonce is found
            unified_found_res = false;
            printf("~~~~~~~~~~ Copying data to unified memory done ~~~~~~~~~~ \n");

            // Initialize GPU parameters
            int num_blocks_per_grid = 256; // Each block will search (2^64 / 16) = 2^60 range
            int num_threads_per_block = 128; // Each thread will search (2^60 / 128) = 2^53 range
            bool runBlock = false;
            unsigned long long thread_search_space = pow(2, 64) / (num_blocks_per_grid * num_threads_per_block);
            
            // Run GPU code
            printf("> Executing parallel code to find nonce below n_decimal: %llu ...\n", n_decimal);
            find_nonce<<<num_blocks_per_grid, num_threads_per_block>>>(thread_search_space, n_decimal, runBlock);

            // Wait for all threads to end & log the time
            cudaDeviceSynchronize(); 
            check_cuda_errors();
            
            // Output data
            if (unified_found_res) {
                printf("~~~~~~~~~~ Found nonce ~~~~~~~~~~\n");
                printf("Output: \n");
                printf("%s\n", nus_net_id.c_str());
                printf("%ld\n", time_now);
                printf("%llu\n", unified_nonce_answer);
                char buffer [65];
                buffer[64] = 0;
                for(int j = 0; j < 32; j++) {
                    sprintf(&buffer[2*j], "%02X", unified_digest_answer[j]);
                }
                printf("%s\n", buffer); // Credit from https://stackoverflow.com/questions/19371845/using-cout-to-print-the-entire-contents-of-a-character-array
            } else {
                printf("~~~~~~~~~~ No nonce found ~~~~~~~~~~\n");
            }
        }
        file.close();
    }
    printf("\n~~~~~~~~~~ Done ~~~~~~~~~~\n");
    return 0;
}

// CPU busy wait.
/*
while (!unified_found_res) {
    check_cuda_errors();
    // No threads found the nonce.
    if (num_thread_failure == num_blocks_per_grid * num_threads_per_block){
        break;
    }
    // Timeout
    if (((float)(clock() - parallel_time)/CLOCKS_PER_SEC) > 900) {
        printf("TIMEOUT.\n");
        break; 
    } 
};*/