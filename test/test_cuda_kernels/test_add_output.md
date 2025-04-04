注：由于向量化版本改变了输入，benchmark 无法正确分析其性能，因此仅给出数值检验结果
``` 
root@83fac93f116f:/workspace/cuda/kernel_samples/build/test# /workspace/cuda/kernel_samples/build/test/test
I0404 14:13:17.296108 55840 test_main.cpp:16] Start Test...
[==========] Running 7 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 4 tests from AddVerify
[ RUN      ] AddVerify.Cpu
N: 1000001
[       OK ] AddVerify.Cpu (2085 ms)
[ RUN      ] AddVerify.Cublas
N: 1000001
[       OK ] AddVerify.Cublas (1799 ms)
[ RUN      ] AddVerify.AddF32
N: 1000001
[       OK ] AddVerify.AddF32 (1796 ms)
[ RUN      ] AddVerify.AddF32x4
N: 1000001
[       OK ] AddVerify.AddF32x4 (2679 ms)
[----------] 4 tests from AddVerify (8362 ms total)

[----------] 3 tests from AddProfile
[ RUN      ] AddProfile.Cpu
N: 1000001
Lower performance(cpu):
Average time: 3.954393 ms
[       OK ] AddProfile.Cpu (4497 ms)
[ RUN      ] AddProfile.Cublas
N: 1000001
Upper performance(cublas):
Average time: 0.636034 ms
[       OK ] AddProfile.Cublas (1017 ms)
[ RUN      ] AddProfile.AddF32
N: 1000001
Add f32 performance:
Average time: 0.631674 ms
Speedup(lower): 6.26x
Speedup(upper): 1.01x
[       OK ] AddProfile.AddF32 (833 ms)
[----------] 3 tests from AddProfile (6348 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 2 test suites ran. (14711 ms total)
[  PASSED  ] 7 tests.
```