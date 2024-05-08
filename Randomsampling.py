import numpy as np

def drawSamples(pmf:dict[str, float], n:int)->list[str]:
    s=0
    g=list(pmf.items())
    samples=[]
    k=list(pmf.keys())
    for i in range(len(g)):
        s+=g[i][1]
        g[i]=tuple(list(g[i])+[s])
    for _ in range(n):
        r=np.random.uniform(0,1)
        flag=0
        for i in range(len(g)):
            if(r<g[i][2] and not flag):
                samples.append(g[i][0])
                flag=1
    return samples

pmf={"ABCD":0.03703704,"EFGH":0.07407407,"IJKL":0.14814815,"MNOP":0.22222222,"QRST":0.40740741, "UVWXYZ":0.11111111}
print(drawSamples(pmf, 4))
# dtup=list(pmf.items())
# print(dtup[1][1])

# pmf={}
# n=int(input("Enter the number of feature labels: "))
# for _ in range(n):
#     feature=input("Enter the feature: ")
#     value=float(input("Enter the pmf of the feature as a floating point value: "))
#     pmf[feature]=value
# no_of_samples=int(input("\nEnter the length of the random samples that we draw from the pmf dictionary"))