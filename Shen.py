import random

e=2.718281828459

class Vec(list):
    '''
    矢量，一个元素只有数字的列表，可以进行按位加减乘等操作
    '''
    def __add__(self, other):
        return Vec([self[i] + other[i] for i in range(len(self))])

    def __sub__(self, other):
        return Vec([self[i] - other[i] for i in range(len(self))])

    def __mul__(self, other):
        return Vec([self[i] * other[i] for i in range(len(self))])

    def __iadd__(self, other):
        return Vec(self+other)

    def __pow__(self, power:float, modulo=None):
        return Vec([self[i] ** power for i in range(len(self))])

    def batchprocess(self,func,*args):
        return Vec([func(self[i],*args) for i in range(len(self))])

class Ten():
    '''
    张量，一个带梯度的Vec，梯度是另一个同维Vec
    '''
    def __init__(self,lis,op=None):
        '''
        创建Ten
        :param lis: list or Vec
        :param op: 创建该张量的运算符，反向传播时用。无需输入
        '''
        self.data=Vec(lis)
        self.grad=Vec([0 for i in range(len(lis))])
        self.op=op

    def __repr__(self):
        return f"<data:{[i for i in self.data]},grad:{[i for i in self.grad]}>"

    def __add__(self, other):
        o=Add()
        return o.compute(self,other)

    def __sub__(self, other):
        o=Sub()
        return o.compute(self,other)

    def __mul__(self, other):
        o=Mul()
        return o.compute(self,other)

    def __pow__(self, power, modulo=None):
        o=Pownum()
        return o.compute(self,power)

    def sum(self):
        o=Sum()
        return o.compute(self)

    def relu(self):
        o=Relu()
        return o.compute(self)

    def exp(self):
        o=Exp()
        return o.compute(self)

    def softmax(self):
        a=self.exp()
        b=Ten.connect([a.sum()**-1 for i in range(len(a.data))])
        return a*b

    def zerograd(self):
        '''
        将自身梯度设为0
        :return: None
        '''
        self.grad = Vec([0 for i in self.grad])

    def onegrad(self):
        '''
        将自身梯度设为1
        :return: None
        '''
        self.grad = Vec([1 for i in self.grad])

    def graddescent(self,k):
        '''
        梯度下降，在反向传播积累梯度后使用
        :param k: float 步长
        :return: None
        '''
        self.data-=self.grad.batchprocess(lambda x:x*k)

    def back(self):
        '''
        反向传播
        :return: None
        '''
        self.grad=Vec([1 for i in self.grad])
        oplist=[self.op]
        for i in oplist:
            if i is None:
                continue
            if type(i.inp) is list: # 若运算符输入为list
                for ea in i.inp:
                    if ea.op not in oplist:
                        oplist.append(ea.op)    # 把每个输入的运算符(去掉重复的)加入表中
            else:
                oplist.append(i.inp.op)
            i.diriv()

    @classmethod
    def connect(cls,x):
        '''
        将一列Ten和为一个Ten
        :param x: list[Ten,Ten...]
        :return: Ten
        '''
        o=Connect()
        return o.compute(x)

class Operator():
    '''
    运算符类，所有对Ten操作的运算符都需要继承此类，并重写compute和diriv方法
    '''
    computelist=[]

    def __init__(self):
        Operator.computelist.append(self)
        self.inp=[]
        self.out=[]

    def compute(self,*args):
        '''
        进行运算。每一个运算符均需重写运算过程，填写self.inp和self.out，创建新的Ten并返回
        self.inp: Ten or list
        self.out: Ten
        :return: Ten
        '''
        pass

    def diriv(self):
        '''
        进行微分。对self.inp.grad进行处理，通常使用+=用于积累梯度
        :return: None
        '''
        pass

    @classmethod
    def back(cls):
        '''
        全局梯度计算。从后向前对每一个运算符使用diriv
        使用前，请先把损失函数的结果的梯度设为1
        使用后，会自动把computelist的内容删除，请注意
        :return:None
        '''
        Operator.computelist.reverse()
        for o in Operator.computelist:
            o.diriv()
        Operator.computelist=[]

class Add(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a:Ten, b:Ten):
        self.inp=[a,b]
        c = Ten(a.data + b.data,self)
        self.out=c
        return c

    def diriv(self):
        self.inp[0].grad += self.out.grad
        self.inp[1].grad += self.out.grad

class Sub(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a:Ten, b:Ten):
        self.inp=[a,b]
        c = Ten(a.data - b.data,self)
        self.out=c
        return c

    def diriv(self):
        self.inp[0].grad += self.out.grad
        self.inp[1].grad -= self.out.grad

class Mul(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a: Ten, b: Ten):
        self.inp = [a, b]
        c = Ten(a.data * b.data,self)
        self.out = c
        return c

    def diriv(self):
        self.inp[0].grad += self.inp[1].data * self.out.grad
        self.inp[1].grad += self.inp[0].data * self.out.grad

class Sum(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([sum(a.data)],self)
        self.out=c
        return c

    def diriv(self):
        for i in range(len(self.inp.grad)):
            self.inp.grad[i]+=self.out.grad[0]

class Connect(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,x:list):
        self.inp=x
        c=Ten([],self)
        for i in range(len(x)):
            c.data.extend(x[i].data)
            c.grad.extend(x[i].grad)
        self.out=c
        return c

    def diriv(self):
        seg=self.out.grad
        for i in range(len(self.inp)):
            self.inp[i].grad += Vec(seg[:len(self.inp[i].grad)])
            seg= seg[len(self.inp[i].grad):]

class Relu(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([max(i,0) for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        for i in range(len(self.inp.data)):
            if self.inp.data[i]>=0:
                self.inp.grad[i]+=self.out.grad[i]

class Pownum(Operator):
    '''
    幂运算
    '''
    def __init__(self):
        super().__init__()

    def compute(self,a,num):
        '''
        幂运算。指数只能为数字
        :param a:Ten
        :param num: float
        :return: Ten
        '''
        self.inp=a
        self.num=num
        c=Ten([i**num for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad=Vec([self.num * self.inp.data[i] ** (self.num - 1) * self.out.grad[i] for i in range(len((self.inp.grad)))])

class Exp(Operator):
    '''
    自然指数函数
    '''
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([e**i for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad=self.out.data*self.out.grad

class Layer:
    '''
    数据层。所有要保存数据的类都需要继承此类
    '''
    layerlist=[]    # 装着所有要保存数据的实例的表
    isload=False    # 是否处于读取状态
    pointer=0   # 读取数据用的指针

    def __init__(self,*args):
        '''
        当处于读取状态时，继承了Layer的实例会按顺序读取layerlist中的内容
        一般情况下，继承它的类的init中需最后调用super().init()，防止数据被覆盖
        '''
        if Layer.isload and len(Layer.layerlist)!=Layer.pointer:
            self.load(Layer.layerlist[Layer.pointer])
            Layer.layerlist[Layer.pointer]=self
            Layer.pointer+=1
        else:
            Layer.layerlist.append(self)

    def save(self):
        '''
        将自身转为str的形式。所有继承了Layer的类需重写此方法
        :return: str
        '''
        pass

    def load(self,*args):
        '''
        从str中读取数据到self
        :param args: str
        :return: None
        '''
        pass

    @classmethod
    def saveall(cls, name):
        '''
        存储所有layerlist中的实例
        :param name: str 保存的文件名
        :return: None
        '''
        with open(name,"w") as f:
            for i in Layer.layerlist:
                f.write(i.save()+"\n")

    @classmethod
    def loadall(cls, name):
        '''
        从文件中读取保存的内容，放到layerlist
        :param name: str 保存的文件名
        :return: None
        '''
        Layer.isload=True
        with open(name,"r") as f:
            Layer.layerlist=f.readlines()

class Linear(Layer):
    def __init__(self,inpsize,outsize,bias=True):
        '''
        全连接层
        :param inpsize: int 作为输入的Ten的维度
        :param outsize: int 作为输出的Ten的维度
        :param bias: bool 是否加上偏置
        '''
        super().__init__()
        if not Layer.isload:
            self.w=[Ten([random.gauss(0,0.01) for i in range(inpsize)]) for i in range(outsize)]
            self.bias=bias
            if bias:
                self.b=[Ten([random.gauss(0,0.01)]) for i in range(outsize)]

    def __call__(self,a):
        '''
        进行运算
        :param a:Ten
        :return: Ten
        '''
        if self.bias:
            o=[(self.w[i]*a).sum()+self.b[i] for i in range(len((self.w)))]
        else:
            o=[(self.w[i]*a).sum() for i in range(len((self.w)))]
        o=Ten.connect(o)
        return o

    def grad_descent_zero(self,k):
        '''
        进行梯度下降，并清空梯度
        :param k: 步长
        :return: None
        '''
        for i in range(len(self.w)):
            self.w[i].graddescent(k)
            self.b[i].graddescent(k)
            self.w[i].zerograd()
            self.b[i].zerograd()

    def save(self):
        t=str([i.data for i in self.w])
        if self.bias:
            t+="/"+str([i.data for i in self.b])
        return t

    def load(self,t):
        t=t.split("/")
        w=eval(t[0])
        self.w=[Ten(i) for i in w]
        if len(t)==2:
            self.bias=True
            b=eval(t[1])
            self.b=[Ten(i) for i in b]
        else:
            self.bias=False

class Ten2(Ten,Layer):
    '''
    与Ten一样，但是可以被保存
    '''
    def __init__(self,lis):
        Ten.__init__(self,lis)
        Layer.__init__(self)

    def save(self):
        t=str(self.data)
        return t

    def load(self,t):
        self.data=eval(t)

def test():
    x=Ten([1])
    y=Ten([1])
    z=Ten([1])

    for i in range(1000):
        s1=((x*y+z)-Ten([40]))**2
        s2=((x*z+y)-Ten([51]))**2
        s3=((x+y+z)-Ten([19]))**2
        if s1.data[0]<0.01 and s2.data[0]<0.01 and s3.data[0]<0.01:
            break
        s1.back()
        s2.back()
        s3.back()
        # s1.grad=Vec([1])
        # s2.grad = Vec([1])
        # s3.grad = Vec([1])
        # Operator.back()
        # Operator.computelist=[]
        x.data -= x.grad * Vec([0.002])
        y.data -= y.grad * Vec([0.002])
        z.data -= z.grad * Vec([0.002])
        print(f"x{x},y{y},z{z}")
        x.zerograd()
        y.zerograd()
        z.zerograd()




