class Vehicle:
    # 构造函数，实例化对象时自动调用
    def __init__(self,value):
        self.value = value
        print("Vehicle value is",self.value)
        
    def update(self,new_value):
        self.value = new_value
        print("Vehicle new value is",self.value)

"""
输出：
Vehicle value is 100
Vehicle value is 150
Vehicle new value is 200
Vehicle new value is 250
"""   
if __name__ == "__main__":
    # 实例化对象
    
    # 自动调用__init__，输出Vehicle value is 100
    vehicle1 = Vehicle(100)
    
    # 自动调用__init__，输出Vehicle value is 150
    vehicle2 = Vehicle(150)

    
    # 调用updata方法
    # 使用不同实例化对象调用函数时self不是同一个对象
        
    
    # 输出Vehicle new value is 200
    vehicle1.update(200)# 等价于调用函数 updata(vehicle1,200)，即vehicle1会作为参数传递给self，self就表示vehicle1
    
    
    # 输出Vehicle new value is 250
    vehicle2.update(250)# 等价于调用函数 updata(vehicle2,250)，即vehicle2会作为参数传递给self，self就表示vehicle2
    
