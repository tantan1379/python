"""
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and
return its sum.
Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach,
which is more subtle.

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
"""


# Violence
# Time Complexity: O(n^2)
# Space Complexity:O(1)
def method_1(nums):
    maxsum = 0
    for i in range(len(nums)):
        subsum = 0
        for j in range(i, len(nums)):
            subsum += nums[j]
            if subsum > maxsum:
                maxsum = max(subsum, maxsum)
    return maxsum

# Dynamic planning
# Time Complexity: O(n)
# Space Complexity: O(1)
def method_2(nums):
    """
    :param nums: List[int]
    :return: int
    """
    subsum = 0
    maxsum = nums[0]
    for i in range(len(nums)):
        subsum = max(subsum + nums[i], nums[i])
        maxsum = max(subsum, maxsum)
    return maxsum

# Greedy Method
# Time Complexity: O(n)
# Space Complexity: O(1)
def method_3(nums):
    """
    :param nums: List[int]
    :return: int
    """
    subsum = nums[0]  # 如果subsum赋值为0，可以将i由0->len(nums)遍历
    maxsum = nums[0]
    for i in range(1, len(nums)):
        if subsum < 0:  # 如果子列和为负先清零，如果后清零可能会出现subsum[0]为负的情况，而subsum[1]为正，maxsum却无法更新到正值的情况
            subsum = 0
        subsum += nums[i]
        maxsum = max(subsum, maxsum)
    return maxsum
    # for i in range(1, len(nums)):
    #     nums[i] = nums[i] + max(nums[i - 1], 0)
    # return max(nums)


if __name__ == "__main__":
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    result1 = method_1(arr)
    result2 = method_2(arr)
    result3 = method_3(arr)
    print(result1)
    print(result2)
    print(result3)
