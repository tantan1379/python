#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/10/24 12:34


# pinyin(“生僻字”)，即可得到拼音列表[['shēng'], ['pì'], ['zì']]
from pypinyin import pinyin


# 判断该单词是否为中文
def is_chinese(word):
    for ch in word:
        # 通常汉字都在'\u4e00' 和 '\u9fff之间，但是生僻字如“㙓” 位于拓展库 '\u3400' 到 '\u4DB5'之间
        if ('\u4e00' <= ch <= '\u9fff') or ('\u3400' <= ch <= '\u4DB5'):
            return True
    return False


# 输入任意行的文本，返回带拼音并上下对齐的文本
def formatting(text):
    # 去除字符串中空格
    no_space_text = ''.join(text.split(" "))
    raw_list = []
    temp = ""
    flag=0
    for i in range(len(no_space_text)):
        if not is_chinese(no_space_text[i]):
             temp+=no_space_text[i]
        else:
            if temp!="":
                raw_list.append(temp)
                temp=""
            raw_list.append(no_space_text[i])

    print(raw_list)
    # 生成拼音列表
    pylist = pinyin(no_space_text)

    # 将拼音列表和字符串拼接对齐
    text_list = []
    pinyin_list = []

    for i in range(len(raw_list)):
        if is_chinese(raw_list[i]):
            pinyin_list.append(pylist[i][0].ljust(6, ' '))
            text_list.append(raw_list[i].ljust(5, ' '))
        else:
            pinyin_list.append(pylist[i][0])
            text_list.append(raw_list[i])

    pinyin_str = ''.join(pinyin_list)
    text_str = ''.join(text_list)

    pinyin_line_list = pinyin_str.split('\n')
    text_line_list = text_str.split("\n")
    final_result = ""
    for i in range(len(pinyin_line_list)):
        final_result += pinyin_line_list[i] + '\n' + text_line_list[i] + "\n"
    return final_result


def main():
    txt_name = "test"
    # 读写txt文档时，encoding="UTF-8-sig"是为了去除utf-8带的BOM头'\ufeff'
    with open("%s.txt" % txt_name, encoding="UTF-8-sig") as f:
        content = f.read()
        input_str = content
    # print('input_str:',input_str)
    result = formatting(input_str)
    print(result)
    with open("%s_pinyin.txt" % txt_name, 'w', encoding='UTF-8-sig') as m:
        m.write(result)


if __name__ == '__main__':
    main()
