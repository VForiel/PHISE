import numba as nb

class Test:
    def __init__(self):
        self.a = 2
        self.b = 3

    def sum(self):
        return Test.sum_jit(self.a, self.b)
    
    @staticmethod
    @nb.njit
    def sum_jit(x, y):
        return x + y
    
test = Test()
result = test.sum()
print(f"The sum is: {result}")