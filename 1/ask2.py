import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def affine_transformation(R, a1, a2, a3, a4, a5, a6):
    ret = np.zeros(R.shape)   # Array to return

    for r in range(R.shape[0]):
        for c in range(R.shape[1]):
            # Get x,y based on center 
            x = r - R.shape[0]/2  
            y = c - R.shape[1]/2
            
            # Apply affine matrix  (x', y')
            aff_x = a1*x+a2*y+a3
            aff_y = a4*x+a5*y+a6
            
            # Map new position (based on center) and round it up
            mapped_r = round(aff_x + R.shape[0]/2)
            mapped_c = round(aff_y + R.shape[1]/2)
            

            # Check if new positions are inside Images range and 
            # store the values of these positions to the ouput array
            if mapped_r in range(0,R.shape[0]) and mapped_c in range(0,R.shape[1]):
                ret[r][c]=R[mapped_r][mapped_c]
    
    return ret


def main():
    if len(sys.argv) != 9:
        print("False command given")
        print("python <script.py> <input_filename> <output_filename> <a1>...<a6>")
        sys.exit()
    
    # Open image as numpy array
    R = np.array(Image.open(sys.argv[1]))

    #Plotting image
    plt.imshow(R, cmap="gray")
    plt.title("Image before Transformation")
    plt.show()

    # Values of affine matrix
    a1, a2, a3, a4, a5, a6 = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),\
                             float(sys.argv[7]), float(sys.argv[8])
    
    # Transform
    ouput = affine_transformation(R, a1, a2, a3, a4, a5, a6)

    #Plot
    plt.imshow(ouput, cmap="gray")
    plt.title('Image after tranformation')
    plt.show()

    # Save
    Image.fromarray(ouput.astype(np.uint8)).save(sys.argv[2])
    
if __name__ == "__main__":
    main()