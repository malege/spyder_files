# 编程练习
# # 输入数字n和k, 输出由1-n组成的所有排列中按大小排序后的第k个数
# import random
# import numpy as np
# def njc(n):
#     jc = 1
#     for i in range(1,n+1):
#         jc*=i
#     return jc
# def knumfind(n,k):
#     n_ = njc(n)
#     n_1 = njc(n-1)
#     lll = []
#     for i in range(1,n+1):
#         a = list(np.arange(1, n + 1))
#         a.remove(i)
#         ll = []
#         s = []
#         while len(ll) < n_1:
#             l = [i]
#             while len(l)!= n:
#                 x = random.choice(a)
#                 if x not in l:
#                     l.append(x)
#             if l not in s:
#                 s.append(l)
#                 su = 0
#                 for j in range(0,n):
#                     su = su + l[j]*10**(n-j-1)
#                 if su not in ll:
#                     ll.append(su)
#         lll.append(ll)
#     lll = np.array(lll).reshape(n_,)
#     lll = sorted(lll)
#     return lll
# if __name__ == "__main__":
#     print('input n: ')
#     n = int(input())
#     print('input k: ')
#     k = int(input())
#
#     lll = knumfind(n,k)
#     print(lll)
#     print(lll[k])



# 字符串解码
# import re
# def decode(stri):
#     ss = ''
#     d = re.findall('\d', stri)
#     s = re.findall('[a-z]', stri)
#     h = d + s
#     for x in stri:
#         if x not in h:
#             return "!error"
#     i = 0
#     while i < len(stri):
#         while stri[i] in s:
#             ss =  ss + stri[i]
#             i =i + 1
#             if i > len(stri)-1:
#                 break
#         if i > len(stri) - 1:
#             break
#         l = []
#         while stri[i] in d:
#             l.append(stri[i])
#             i =i + 1
#             if i > len(stri)-1:
#                 break
#         n=0
#         for j in range(len(l)):
#             n =  n + int(l[j])*10**(len(l)-j-1)
#         if n > 2:
#             ss = ss + stri[i]*n
#             i =i + 1
#         else:
#             return "!error"
#     return ss
# if __name__ == "__main__":
#     while(1):
#         st = input()
#         ss = decode(st)
#         print(ss)


# 求一个非负整数对应的二进制数中一的个数
# def numOf1(n):
#     num = 0
#     nn = bin(n)
#     # print(nn)
#     for x in nn:
#         if x=='1':
#             num +=1
#     print(num)
# def numOf_1(n):
#     num = 0
#     while(n):
#         num += 1
#         n = n&(n-1)
#     print(num)
# numOf_1(255)