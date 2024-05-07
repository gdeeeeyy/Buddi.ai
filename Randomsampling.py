import numpy as np

def random_sampler(pmf, no_of_samples):
    sum=0
    cmf=[]
    samples=[]
    for i in range(len(pmf)):
        sum+=pmf[i]
        cmf.append(sum)
    n=len(cmf)
    for _ in range(no_of_samples):
        r=np.random.uniform(0,1)
        for i in range(n):
            if(r>cmf[i]):
                samples.append(i)
                break
    return samples

distribution={"ABCD":1,"EFGH":2,"IJKL":4,"MNOP":6,"QRST":11,"UVWXYZ":3}
data_pts=list(distribution.values())
check_arr=[]
no_of_samples=4
pmf=[data_pts[i]/sum(data_pts) for i in range(len(data_pts))]
samples=random_sampler(pmf, no_of_samples)
print(f"The samples returned from random sampling is {samples}")

# for i in range(len(data_pts)):
#     print(f"no of samples from feature {i+1} is {samples.count(i)}")


#drawsample(dict pmf(float), n_int)-> list(str)
#add comments