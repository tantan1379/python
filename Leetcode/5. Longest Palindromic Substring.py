'''
Given a string s, return the longest palindromic substring in s. 回文字符串
'''
# Violent O(n^3)
def isPalindrome(s):
    for i in range(len(s)//2):
        if(s[i] != s[len(s)-i-1]):
            return False
    return True


def longestPalindrome_v(s):
    max_len = 0
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            sub_s = s[i:j]
            if(isPalindrome(sub_s) and max_len < len(sub_s)):
                max_len = len(sub_s)
                ans = sub_s[:]
    return ans


# ----------------------------------------------------------------
# Dynamic planning O(n^2)
def longestPalindrome(s):
    n = len(s)
    if n <= 1:
        return s
    max_len = 1
    begin = 0
    dp = [[False]*n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True

    for L in range(2, n+1):
        for i in range(n):
            j = i+L-1
            if j>=n:
                break
            if s[i]==s[j]:
                if j-i<3:
                    dp[i][j]=True
                else:
                    dp[i][j]=dp[i+1][j-1]
            if dp[i][j] and j-i+1>max_len:
                maxlen = j-i+1
                begin = i
    return s[begin:begin+maxlen]


if __name__ == "__main__":
    print(longestPalindrome("1423214"))
