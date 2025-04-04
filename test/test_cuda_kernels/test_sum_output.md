注：由于向量化版本改变了输入，benchmark 无法正确分析其性能，因此仅给出数值检验结果
```
I0404 14:07:40.319449 49144 test_main.cpp:16] Start Test...
[==========] Running 10 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 6 tests from SumVerify
[ RUN      ] SumVerify.Cpu
N: 1000001
[       OK ] SumVerify.Cpu (901 ms)
[ RUN      ] SumVerify.Cublas
N: 1000001
[       OK ] SumVerify.Cublas (653 ms)
[ RUN      ] SumVerify.globalMemF32
N: 1000001
[       OK ] SumVerify.globalMemF32 (626 ms)
[ RUN      ] SumVerify.SharedMemF32
N: 1000001
[       OK ] SumVerify.SharedMemF32 (609 ms)
[ RUN      ] SumVerify.BlockAllF32
N: 1000001
[       OK ] SumVerify.BlockAllF32 (595 ms)
[ RUN      ] SumVerify.BlockAllF32x4
N: 1000001
[       OK ] SumVerify.BlockAllF32x4 (722 ms)
[----------] 6 tests from SumVerify (4110 ms total)

[----------] 4 tests from SumProfile
[ RUN      ] SumProfile.Cpu
Lower performance(cpu):
Average time: 2.496553 ms
[       OK ] SumProfile.Cpu (2797 ms)
[ RUN      ] SumProfile.Cublas
Upper performance(cublas):
Average time: 0.969277 ms
[       OK ] SumProfile.Cublas (1218 ms)
[ RUN      ] SumProfile.SharedMemF32
Shared memory result:
Average time: 1.695689 ms
Speedup(lower): 1.47x
Speedup(upper): 0.57x
[       OK ] SumProfile.SharedMemF32 (1912 ms)
[ RUN      ] SumProfile.BlockAllF32
Warp shuffle result:
Average time: 1.650831 ms
Speedup(lower): 1.51x
Speedup(upper): 0.59x
[       OK ] SumProfile.BlockAllF32 (1862 ms)
[----------] 4 tests from SumProfile (7791 ms total)

[----------] Global test environment tear-down
[==========] 10 tests from 2 test suites ran. (11902 ms total)
[  PASSED  ] 10 tests.
```