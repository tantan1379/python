'''
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
'''

def isValid(s):
    if len(s)%2 ==1:
        return False
    stack = list()
    pairs = {
        "}":"{",
        ")":"(",
        "]":"["
    }
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1]!=pairs[ch]:
                return False
            else:
                stack.pop()
        else:
            stack.append(ch)
    return not stack


print(isValid("{{()}[]}"))