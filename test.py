import time

def compare_add_mult_speed(num_operations=1000000):
    """
    比较计算机在相同计算量下加法和乘法的速度。

    Args:
        num_operations: 执行的加法和乘法操作的数量。

    Returns:
        None. 打印加法和乘法所用的时间，并指出哪个更快。
    """

    # 加法测试
    start_time = time.time()
    sum_result = 0
    for _ in range(num_operations):
        sum_result += 1  # 简单的加法操作
    add_time = time.time() - start_time

    # 乘法测试
    start_time = time.time()
    mult_result = 1
    for _ in range(num_operations):
        mult_result *= 2  # 简单的乘法操作
    mult_time = time.time() - start_time

    print(f"执行 {num_operations} 次加法耗时: {add_time:.6f} 秒")
    print(f"执行 {num_operations} 次乘法耗时: {mult_time:.6f} 秒")

    if add_time < mult_time:
        print("在相同的计算量下，加法通常比乘法更快。")
    elif mult_time < add_time:
        print("在相同的计算量下，乘法通常比加法更快 (这种情况比较少见)。")
    else:
        print("在相同的计算量下，加法和乘法的速度大致相同。")

if __name__ == "__main__":
    compare_add_mult_speed()
    compare_add_mult_speed(num_operations=10000000) # 增加操作数量进行测试