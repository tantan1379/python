#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: 1_TwoSum.py
# author: twh
# time: 2020/11/27 12:41
"""
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
"""


class Solution(object):
    def method1(self, nums, target):
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]

    def method2(self, nums, target):
        # Hashmap solution
        # Time complexity: O(n)
        # Space complexity: O(n)
        mapping = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in mapping.keys():
                return [mapping[diff], i]
            else:
                mapping[nums[i]] = i


F = Solution()
arr = [-1, 3, 4, 5, 2, 9]
a, b = F.method1(arr, 8)
c, d = F.method2(arr, 8)
print(a, b)
print(c, d)
