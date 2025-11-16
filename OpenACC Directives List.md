## 🏷️ OpenACC Directives and API Calls

OpenACC uses **directives** (Fortran: `!$acc`, C/C++: `#pragma acc`) followed by a construct name and optional clauses .

| Construct Name | C/C++ Pragma | Purpose | Key Clauses Mentioned |
| :--- | :--- | :--- | :--- |
| **Kernels** | `#pragma acc kernels` | Executes the enclosed loops as one or more separate kernels on the GPU [cit. | `if()`, `async()`, Data Clauses (e.g., `copy`, `create`. |
| **Data** | `#pragma acc data` | Manages the movement and residency of data between the host (CPU) and accelerato. | `if()`, `async()`, Data Clauses (e.g., `copy`, `create`) [cit. |
| **Update** | `#pragma acc update` | Used to explicitly move existing data copies between the host and device *after* one copy has been modifie. | `host(list)`, `device(list)`, `async()`, `if(). |
| **Parallel** | `#pragma acc parallel` | Explicitly creates a parallel region, mapping execution units (gangs/workers) directly to the parallel structure. | `num_gangs()`, `num_workers()`, `vector_length()`, `private()`, `reduction()`, Data Clause. |
| **Loop** | `#pragma acc loop` | Provides detailed control over how the subsequent loop is parallelized within a `parallel` or `kernels` region. | `collapse(n)`, `seq`, `private()`, `reduction()`, `gang`, `worker`, `vector`, `independent` [cit. |
| **C | `#pragma acc cache` | Caches data in the software-managed data cache (typically GPU shared . | (No clauses explicitly listed for this construct). |
| **Host  | `#pragma acc host_data` | Makes the address of device data available on the hos. | (No clauses explicitly listed for this construct). |
| ** | `#pragma acc wait` | Waits for asynchronous GPU activity to complet. | (No clauses explicitly listed for this construct). |
| **Dec | `#pragma acc declare` | Specifies that data should be allocated in device memory for the duration of a subprogram's executio. | (No clauses explicitly listed for this construct). |

***

## 💾 OpenACC Data Clauses

These clauses are used with `data`, `kernels`, and `parallel` constructs to manage data movement.

| Clause Name | Action |
| :--- | :--- |
| **`copy(list)`** | Allocates memory on the GPU, copies data **from host to GPU** upon entering the region, and copies data back **to the host** upon exiting. |
| **`copyin(list)`** | Allocates memory on the GPU and copies data **from host to GPU** upon entering the region. |
| **`copyout(list)`** | Allocates memory on the GPU and copies data **to the host** upon exiting the region. |
| **`create(list)`** | Allocates memory on the GPU but **does not copy** data from the host. |
| **`present(list)`** | Asserts that data is **already resident** on the GPU from a containing data region. |
| **`deviceptr(list)`** | Specifies that the pointer variables in the list already hold **device addresses** (instead of host addresses). |

***

## ⚙️ Runtime Library Routines (C API)

These are explicit C functions that allow control over the device and asynchronous operations.

* `acc_get_num_devices()`
* `acc_set_device_type()`
* `acc_get_device_type()`
* `acc_set_device_num()` 
* `acc_get_device_num()` 
* `acc_async_test()` 
* `acc_async_wait()` 
* `acc_async_wait_all()` 
* `acc_async_test_all()` 
* `acc_shutdown()` 
* `acc_malloc()` 
* `acc_free()` 