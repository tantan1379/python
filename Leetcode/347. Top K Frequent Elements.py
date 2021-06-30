# TODO doing
'''
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]
'''
from utils.heapsort import heapSort


def topKFrequent(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """
    hashmap = {}

    for num in nums:
        if(num in hashmap.keys()):
            hashmap[num] += 1
        else:
            hashmap[num] = 1
    
    
    # for key,value in hashmap.items():
    #     if(arr[value]==[]):
    #         arr[value] = []
    #     arr[value].append(key)

    # print(arr)

topKFrequent([7,6,1,1,1,2,2,2,2,5,3],0)
    
