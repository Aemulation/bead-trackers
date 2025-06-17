import threading
import concurrent.futures
import multiprocessing
import time
import numpy as np


number_of_frames_per_buffer = 100
threads = []

image_size = (2016, 2560)

source = np.ones((number_of_frames_per_buffer, *image_size), dtype=np.uint16)
destination = np.zeros((number_of_frames_per_buffer, *image_size), dtype=np.uint16)

num_iters = 10
num_warmup = 10


def copy_image(destination_id):
    np.copyto(destination[destination_id], source[destination_id])


# multithreaded_results = []
# for _ in range(num_iters + num_warmup):
#     start = time.perf_counter()
#     for i in range(number_of_frames_per_buffer):
#         thread = threading.Thread(target=copy_image, args=(i,))
#         thread.start()
#         threads.append(thread)
#
#     for thread in threads:
#         thread.join()
#
#     end = time.perf_counter()
#     print(f"multi threaded elapsed: {(end - start) * 1_000}ms ")
#     multithreaded_results.append(end - start)
# for _ in range(num_warmup):
#     multithreaded_results.pop(0)
# multithreaded_results = np.array(multithreaded_results)

threadpool_results = []
for _ in range(num_iters + num_warmup):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(5) as executor:
        for i in range(number_of_frames_per_buffer):
            executor.submit(copy_image, i)

    end = time.perf_counter()
    print(f"threadpool elapsed: {(end - start) * 1_000}ms ")
    threadpool_results.append(end - start)
for _ in range(num_warmup):
    threadpool_results.pop(0)
threadpool_results = np.array(threadpool_results)

multiprocessing_results = []
for _ in range(num_iters + num_warmup):
    start = time.perf_counter()
    with multiprocessing.Pool(5) as pool:
        pool.map(copy_image, range(number_of_frames_per_buffer))

    end = time.perf_counter()
    print(f"multiprocessing elapsed: {(end - start) * 1_000}ms ")
    multiprocessing_results.append(end - start)
for _ in range(num_warmup):
    multiprocessing_results.pop(0)
multiprocessing_results = np.array(multiprocessing_results)

singlethreaded_results = []
for _ in range(num_iters + num_warmup):
    start = time.perf_counter()

    for i in range(number_of_frames_per_buffer):
        copy_image(i)

    end = time.perf_counter()
    print(f"single threaded elapsed: {(end - start) * 1_000}ms ")
    singlethreaded_results.append(end - start)
for _ in range(num_warmup):
    singlethreaded_results.pop(0)
singlethreaded_results = np.array(singlethreaded_results)


print()
print()
print()
print(f"SINGLETHREADED MEAN:  {np.mean(singlethreaded_results) * 1_000}")
print(f"SINGLETHREADED STD:   {np.std(singlethreaded_results) * 1_000}")
print()
print(f"MULTITHREADED MEAN:   {np.mean(multithreaded_results) * 1_000}")
print(f"MULTITHREADED STD:    {np.std(multithreaded_results) * 1_000}")
print()
print(f"THREADPOOL MEAN:      {np.mean(threadpool_results) * 1_000}")
print(f"THREADPOOL STD:       {np.std(threadpool_results) * 1_000}")
print()
print(f"MULTIPROCESSING MEAN: {np.mean(multiprocessing_results) * 1_000}")
print(f"MULTIPROCESSING STD:  {np.std(multiprocessing_results) * 1_000}")
