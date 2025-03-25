from wuli import *
from Shen import *
import random,os,time

class env:
    def __init__(self):
        Phy.biao=[]
        self.car=Phy(1,[0,0,0],[0,0,0],r=3,color="red")
        self.stick=Phy(0.1,[random.uniform(-5,5),0,0],[0,100,0],r=2)

    def go(self,act,t=0.01,time=15):
        for i in range(time):
            self.car.resilience(k=1000,other=self.stick)
            self.stick.force([0,-2,0])
            if act:
                self.car.force([100,0,0])
            else:
                self.car.force([-100,0,0])
            self.car.a[1]=0
            Phy.run(t)

    def getstat(self):
        return Ten([self.car.p[0] / 10, self.car.v[0] / 5,
                    self.stick.p[0] / 10, self.stick.p[1] / 10, self.stick.v[0] / 5, self.stick.v[1] / 5])

    def isend(self):
        return self.stick.p[1]<0

def reward(stat):
    return 1 if stat.data[3]>9.5 and abs(stat.data[0])<5 else 0

def test(stat):
    return 0 if stat.data[0]>stat.data[2] else 1

def mase(x,y):
    a=((x-y)**2).sum()
    if a.data[0]<1:
        return a
    else:
        return a**0.5

def clip(x,maxx,minx):
    if x.data[0]>maxx:
        return Ten([maxx])
    elif x.data[0]<minx:
        return Ten([minx])
    else:
        return x

def minten(x,y):
    if x.data[0]<=y.data[0]:
        return x
    else:
        return y

def entropy(x):
    return Ten([-1]).expand(len(x))*x*x.log()

class model:
    def __init__(self):
        self.f1=Linear(6,128)
        self.f2=Linear(128,2)

    def forward(self,x):
        x=self.f1(x).relu()
        x=self.f2(x).softmax()
        return x

    def choice(self,x):
        v = self.forward(x).data
        a=v.index(random.choices(v,v)[0])
        Operator.clean()
        return a

    def optimize(self,k=0.01):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
    
    def dcopy(self):
        c=model()
        c.f1=self.f1.dcopy()
        c.f2=self.f2.dcopy()
        return c

class modelv:
    def __init__(self):
        self.f1=Linear(6,128)
        self.f2=Linear(128,1)

    def forward(self,x):
        x=self.f1(x).relu()
        x=self.f2(x)
        return x

    def optimize(self,k=0.01):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)

    def dcopy(self):
        c=modelv()
        c.f1=self.f1.dcopy()
        c.f2=self.f2.dcopy()
        return c

def show(m):
    e=env()
    Phy.tready()
    r=0
    for i in range(2000):
        act=m.choice(e.getstat())
        e.go(act)
        Phy.tplay()
        #print(reward(e.getstat()))
        r+=reward(e.getstat())
        if e.isend():
            break
        time.sleep(0.03)
    print(r)

def evalm(m,n=1):
    r=0
    for j in range(n):
        e = env()
        for i in range(2000):
            act = m.choice(e.getstat())
            e.go(act)
            r+=reward(e.getstat())
            if e.stick.p[1]<0:
                break
    return r/n

class memory:
    def __init__(self,maxsize=10):
        self.memo=[]
        self.maxsize=maxsize
    
    def experience(self,m,times=3,n=500):
        for t in range(times):
            e=env()
            exp=[]
            ar=0
            for i in range(n):
                s=e.getstat()
                out=m.forward(s).data
                a=out.index(random.choices(out,out)[0])
                p=out[a]
                Operator.clean()
                e.go(a)
                st=e.getstat()
                r=reward(s)
                ar+=r
                exp.append([s,a,r,st,0,p])
                if i/ar>1.5: #e.isend():
                    exp[-1][4]=1
                    break
            self.memo.append(exp)
        if len(self.memo)>self.maxsize:
            self.memo=self.memo[-self.maxsize:]
        return ar/times
    

def train(m, mv, memo, times=1,discount=0.99,lamb=0.99,eps=0.2,he=1):
    ar=memo.experience(m,times)
    for exp in memo.memo:
        aloss=0
        alossv=0
        ad=[]
        gae=0
        for i in range(len(exp)-1,-1,-1):
            v=mv.forward(exp[i][0]).data[0]
            v2=mv.forward(exp[i][3]).data[0]
            tdd=exp[i][2]+discount*v2*(1-exp[i][4])-v
            gae=tdd+lamb*discount*gae*(1-exp[i][4])
            ad.append(gae)
            Operator.clean()
        ad.reverse()
        
        # mean=sum(ad)/len(ad)
        # std=sum([(a-mean)**2 for a in ad])**0.5
        # for i in range(len(exp)):
        #     ad[i]=(ad[i]-mean)/std
            
        for i in range(len(exp)):
            p=m.forward(exp[i][0])
            pc=p.cut(exp[i][1],exp[i][1]+1)
            v=mv.forward(exp[i][0])
            # v2=mv.forward(exp[i][3])
            # tdt=Ten([exp[i][2]])+Ten([discount])*v2*Ten([1-exp[i][4]])
            adv=Ten([ad[i]])
            
            h=entropy(pc)
            ratio=(pc.log()-Ten([exp[i][5]]).log()).exp()
            surr=ratio*adv
            surr2=clip(ratio,1+eps,1-eps)*adv
            loss=Ten([-1])*(minten(surr,surr2)+Ten([he])*h)

            loss.onegrad()
            aloss+=loss.data[0]
            lossv=mase(v,Ten((v+adv).data))
            alossv+=lossv.data[0]
            if abs(lossv.data[0]) > 50:
                print(f"loss-v{lossv.data[0]},梯度过高")
                Operator.clean()
                return
            Operator.back()
    c=sum([len(i) for i in memo.memo])
    m.optimize(0.001/c)
    mv.optimize(0.002/c)
    with open("fil.txt","a") as f:
        f.write(f"{ar} {aloss/c} {alossv/c}\n")
    print(ar,aloss/c,alossv/c)

savename="rlt-2-rebase-ppo-end2"
if savename in os.listdir():
    print("load",savename)
    Layer.loadall(savename)
m=model()
mv=modelv()
Layer.issave=False
memo=memory(2)
count=0
while True:
    train(m,mv,memo,discount=0.98,he=0.5)
    count+=1
    if count%5==0:
        Layer.saveall(savename)
    # show(m)
