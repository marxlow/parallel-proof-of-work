# parallel-proof-of-work

Create Alias to compile and run the program. The program will read in the test file from "test.in":
alias run='nvcc -arch=sm_52 -rdc=true -o poc hash.cu proof_of_work.cu; nvprof ./poc test.in'

Open terminal in directory and run:
`run`

- Running the program will generate a "test.out" file which logs all the output of the program.
- Setting the boolean "runBlock" to be false will run the program with cyclic distribution.
- Setting the boolean "runBlock" to be true will run the program with block distribution (default).
