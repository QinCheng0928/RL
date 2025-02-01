class A:
    def __init__(self):
        self.value = 0
    def printf(self):
        print("the value is :",self.value)
    def updata(self,val):
        self.value = val

def change(b:A):
    b.value = 10

# 类的实例化对象作为参数传递是 引用传递
if __name__ == "__main__":
    a = A() 
    a.printf()
    change(a)
    a.printf()