# -*- coding: utf-8 -*-
#########函数装饰器学习#############
# def funA(fn):
#     print("C语言中文网")
#     fn() # 执行传入的fn参数
#     print("http://c.biancheng.net")
#     return "装饰器函数的返回值"
# @funA

# def funB():
#     print("学习 Python")  
# print(funB) 



# def funA(fn):
#     # 定义一个嵌套函数
#     def say(arc):
#         print("Python教程:",arc)
#     return say
# @funA
# def funB(arc):
#     print("funB():", arc)
#     print("Hello World!")
#     print('chougoushi!')
# funB("http://c.biancheng.net/python")



# import pdb
# class A(object):
#  def __init__(self):
#   self.n = 10
 
#  def minus(self, m):
#   self.n -= m
 
 
# class B(A):
#  def __init__(self):
#   self.n = 7
 
#  def minus(self, m):
#   super(B,self).minus(m)
#   self.n -= 2
# # b=B()
# # b.minus(3)
# # print(b.n)


# class C(A):
#  def __init__(self):
#   self.n = 12
 
#  def minus(self, m):
#   super(C,self).minus(m)
#   self.n -= 5
 
 
# class D(B, C):
#  def __init__(self):
#   self.n = 15
 
#  def minus(self, m):
#   super(D,self).minus(m)
#   self.n -= 2
 
# d=D()
# d.minus(2)
# print(d.n)

# import sys
# class Parent:
#   Value = "Hi, Parent value"
  
#   def fun(self):
#     print("This is from Parent")
  
  
# # 定义子类，继承父类
# class Child(Parent):
#     Value = "Hi, Child value"
  
#     def fun(self):
#         super().fun()
#         print("This is from Child")
  
  
# c = Child()
# c.fun()
# print(c.Value)

# import numpy as np
# arr = np.arange(12).reshape(3,4)
# print(arr)
# print(arr.flags)