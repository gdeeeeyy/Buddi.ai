import numpy as np
import matplotlib.pyplot as plt

def convolve(str1: str, str2:str)->list[float]:
    #the string is converted into an array of words
    arr1=str1.split(" ")
    arr2=str2.split(" ")
    #array that contains comparison count of substrings but in reverse
    cRev=[]
    #array that contains comparison count of substrings
    cSub=[]
    #array that contain comparisons with entire substring
    cEnt=[]
    #array that contain comparisons with entire substring in rev
    cEntRev=[]
    #reversing the 2nd array
    arr2Rev=arr2[::-1]
    for i in range(len(arr2)):
        #The count of substrings is initialized
        ci, cei=0,0
        #The count of substrings in reverse is initialized
        cj, cej=0,0
        #The array is made into 0*(no of elements left) + arr[:i+1]
        arr2Sub=[0 for _ in range(len(arr1)-(i+1))]+arr2[:i+1]
        #The array is made into 0*(no of elements left) + arrRev[:i+1]
        arr2RevSub=arr2Rev[:i+1]+[0 for _ in range(len(arr1)-(i+1))]
        #going from 1 word to n words(the 2nd string is slid along with the first string and it is convolved)
        for j in range(len(arr2Sub)):
            #if a substring of arr2 is present in arr1, then ci is incremented
            if(arr2Sub[j] in arr1):
                ci+=1
            #if a substring is present in the same positing in arr1, the cei is incremented
            if(arr2Sub[j]==arr1[j]):
                cei+=1
        #going from n words to 1 word
        for k in range(len(arr2RevSub)):
            #if a substring of reverse of arr2 array is present in arr1, then cj is incremented
            if(arr2RevSub[k] in arr1):
                cj+=1
            #if a substring in the revese of arr2 array is present in the same positing in arr1, the cej is incremented
            if(arr2RevSub[k]==arr1[k]):
                cej+=1
        #The value of ci and cj is appended at the end of the loop for each iterations
        cSub.append(ci)
        cRev.append(cj)
        cEnt.append(cei)
        cEntRev.append(cej)
    return cSub+cRev[::-1], cEnt+cEntRev[::-1]

def main():
    #calling the convolve function for the strings "he is a bad boy" and "I watched bad boy 2"
    cOut, cEnt=convolve("He is a bad boy", "I watched bad boy 2")
    print(f"The output for each step of convulution for entire string comparison is: {cEnt}")
    print(f"The output for each step of convulution for partial string comparison is: {cOut}")
    plt.title("Convolving two strings")
    plt.plot(range(len(cOut)), cOut, label="Convolving two strings together(partial string comparison)")
    plt.plot(range(len(cEnt)), cEnt, label="Convolving two strings together(entire string comparison)")
    plt.figtext(0.5, 0.01, "The giving two strings are convolved and sub string matching is performed on each stride of the convolution", wrap=True, horizontalalignment='center', fontsize=8)
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()