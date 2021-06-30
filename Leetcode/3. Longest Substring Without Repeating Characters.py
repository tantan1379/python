'''
Given a string s, find the length of the longest substring without repeating characters.
'''
# 暴力解法 Time O(n^2) Space O(n)
def lengthOfLongestSubstring(s):
    maxlen = 0
    for i in range(len(s)):
        curlen = 0
        lookup = set()
        for j in range(i, len(s)):
            if s[j] in lookup:
                break
            lookup.add(s[j])
            curlen += 1
                
        maxlen = max(curlen, maxlen)
    return maxlen


# 滑动窗口 Time O(n) Space O(gamma) gamma为字符可能出现的个数
# 思路：枚举以每一个字符开头的最大非重复子字符串。
# 每次移动窗口都去除哈希集中当前的前一个字符，并从上一个最大非重复子字符串的最后一个字符开始向后扩展
def lengthOfLongestSubstring_(s):
    if len(s)<2:
        return len(s)
    occ = set() # 哈希集合，记录每个字符是否出现过
    rk, ans = -1, 1 # rk为右指针
    for i in range(len(s)): # i为左指针
        if i != 0:
            occ.remove(s[i - 1]) # 左指针向右移动一格，移除一个字符
        while rk + 1 < len(s) and s[rk + 1] not in occ: 
        # [while]判断：（1）右指针是否越界（2）窗口的下一个字符是否出现重复字符 
        # 如果为True，则：（1）右指针向右移动一位 （2）添加下一个字符到哈希集中
            occ.add(s[rk + 1])
            rk += 1
        # 第 i 到 rk 个字符是一个极长的无重复字符子串
        ans = max(ans, rk - i + 1)
    return ans


if __name__ == "__main__":
    print(lengthOfLongestSubstring("abcabcbb"))
