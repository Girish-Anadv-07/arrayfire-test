Debugging ArrayFire Issues {#debugging}
===============================================================================

Using Environment Variables
---------------------------

 * [`AF_PRINT_ERRORS=1`](configuring_environment.htm#af_print_errors) : Makes exception's messages more helpful
 * [`AF_TRACE=all`](configuring_environment.htm#af_trace): Print ArrayFire message stream to console
 * [`AF_JIT_KERNEL_TRACE=stdout`](configuring_environment.htm#af_jit_kernel_trace): Writes out source code generated by ArrayFire's JIT to the specified target
 * [`AF_OPENCL_SHOW_BUILD_INFO=1`](configuring_environment.htm#af_opencl_show_build_info): Print OpenCL kernel build log to console


Tips in Language Bindings
-------------------------

### C++

* `af_print_mem_info("message", -1);`: Print table of memory used by ArrayFire on the active GPU

### Python

* `arrayfire.device.print_mem_info("message")`: Print table of memory used by ArrayFire on the active GPU



Further Reading
---------------

See the [ArrayFire README](https://github.com/arrayfire/arrayfire) for support information.