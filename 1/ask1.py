import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# np.mean(..., axis=2) returns the average values across all three color channels.
# Could be done by using 2 for loops (rows,cols) but numpy doesn't need iterations
# Example..
# if(len(A.shape)==3):
#     for m in range (A.shape[0]):
#         for n in range (A.shape[1]):
#             R[m][n]=(A[m][n][0] + A[m][n][1] + A[m][n][2]) / 3    
                        # #Array A should be upcasted to A.astype(float)

def grayscaling(A):
    if(len(A.shape)==3):
        return np.mean(A, axis=2)
    return A


# thresholding function:
# mask is a boolean array that represents (based on index) if a value of an array is 
# True/False based on a condition (<= threshold), mask will have the same shape as the given array

# Could be done with 2 for loops again, ex.
# for m in range (A.shape[0]):
#     for n in range (A.shape[1]):
#         if(R[m,n]<=threshold):
#             ouput[m,n]=0
#         else:
#             output[m,n]=255

def thresholding(A, k):
    mask = A[:] <= k            # Boolean array, repsresents if the condition(<=K) is True for each value in A 
    A[mask] = 0                 # Everything that is True (based on index) is given the value 0 (black) 
    A[np.invert(mask)] = 255    # bit-wise inversion of mask (True -> False , False->True) and make these values 255 (white)
    return A

def main():

    if len(sys.argv) != 4:
        print("False command given")
        print("python <script.py> <input_filename> <output_filename> <threshold-k>")
        sys.exit()

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    k = int(sys.argv[3])

    R = np.array(Image.open(input_filename))
    R = grayscaling(R)
    R = thresholding(R, k)
    
    plt.imshow(R,cmap="gray")
    plt.title('Threshold' +str(k))
    plt.show()

    Image.fromarray(R.astype(np.uint8)).save(output_filename)
if __name__ == "__main__":
    main()