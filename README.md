# parallel-proof-of-work

Create Alias:
alias run='nvcc -arch=sm_52 -rdc=true -o poc hash.cu proof_of_work.cu; nvprof ./poc test.in'
