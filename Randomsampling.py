import numpy as np

def draw_samples(pmf, no_of_samples):
    s=0
    cmf={}
    samples=[]
    for (k,v) in pmf.items():
        s+=v
        cmf[k]=s
    for _ in range(no_of_samples):
        r=np.random.uniform(0,1)
        flag=0
        print(r)
        for k in cmf:
            if(r<cmf[k] and not flag):
                samples.append(k)
                flag=1
    return samples

pmf={"ABCD":0.03703704,"EFGH":0.07407407,"IJKL":0.14814815,"MNOP":0.22222222,"QRST":0.40740741, "UVWXYZ":0.11111111}
print(draw_samples(pmf, 4))

# pmf={}
# n=int(input("Enter the number of feature labels: "))
# for _ in range(n):
#     feature=input("Enter the feature: ")
#     value=float(input("Enter the pmf of the feature as a floating point value: "))
#     pmf[feature]=value
# no_of_samples=int(input("\nEnter the length of the random samples that we draw from the pmf dictionary"))