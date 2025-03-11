import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
from matplotlib.image import imread
import imageio
from scipy import ndimage
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


#histogram

def calculate_histogram(image):
    # Check if the image is a NumPy array
    if isinstance(image, np.ndarray):
        img = image
    else:
        # Load the image
        img = cv2.imread(image)
    
    # Ensure the image is in the correct format
    if img is None:
        raise ValueError("Invalid image input. Unable to load the image.")
    img = img.astype(np.uint8)  # Ensure correct dtype
    
    # Check if the image is RGB (3 channels)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Check if the image is essentially grayscale
        if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
            fig = Figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            ax.plot(histogram, color='black', label='Grayscale')
            ax.set_xlabel('Pixel intensity')
            ax.set_ylabel('Number of pixels')
            ax.legend()
            ax.set_title('Grayscale Histogram')
        else:
            colors = ('b', 'g', 'r')  # OpenCV loads images in BGR order
            fig = Figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            for i, color in enumerate(colors):
                histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
                ax.plot(histogram, color=color, label=f'{color.upper()} channel')
            ax.set_xlabel('Pixel intensity')
            ax.set_ylabel('Number of pixels')
            ax.legend()
            ax.set_title('RGB Histogram')
    else:
        # Convert to grayscale if not already
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        fig = Figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.plot(histogram, color='black', label='Grayscale')
        ax.set_xlabel('Pixel intensity')
        ax.set_ylabel('Number of pixels')
        ax.legend()
        ax.set_title('Grayscale Histogram')
    
    canvas = FigureCanvas(fig)
    canvas.print_figure("Histo.png")
    return canvas


#Contrast Streching
def contrast_stretching(image_path):

    #read image
    image = Image.open(image_path)


    # Check if image is grayscale 
    if np.array(image).ndim == 2:
       img_array = np.array(image)

       minimum = np.min(img_array)
       maximum = np.max(img_array)

       for i in range(img_array.shape[0]):
           for j in range(img_array.shape[1]):
               img_array[i,j] = (img_array[i,j] - minimum) * 255 / (maximum - minimum)

       stretched_image = Image.fromarray(np.uint8(img_array))

       return stretched_image
   
    # if image is color
    elif np.array(image).ndim == 3: 
       img_array = np.array(image)

       for c in range(img_array.shape[2]):
           channel_min = np.min(img_array[:,:,c])
           channel_max = np.max(img_array[:,:,c])

           for i in range(img_array.shape[0]):
               for j in range(img_array.shape[1]):
                   img_array[i,j,c] = (img_array[i,j,c] - channel_min) * 255 / (channel_max - channel_min)

       stretched_image = Image.fromarray(np.uint8(img_array))

       return stretched_image
   

#Histogram Equalization

def Histogram_equalizer(img_filename):
    ######################################
    # READ IMAGE FROM FILE
    ######################################
    #load file as pillow Image 
    img = Image.open(img_filename)

    # convert to grayscale
    imgray = img.convert(mode='L')

    #convert to NumPy array
    img_array = np.asarray(imgray)


    ######################################
    # PERFORM HISTOGRAM EQUALIZATION
    ######################################

    """
    STEP 1: Normalized cumulative histogram
    """
    #flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)

    #normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels

    #normalized cumulative histogram
    chistogram_array = np.cumsum(histogram_array)


    """
    STEP 2: Pixel mapping lookup table
    """
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)


    """
    STEP 3: Transformation
    """
    # flatten image array into 1D list
    img_list = list(img_array.flatten())

    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]

    # reshape and write back into img_array"
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    ######################################
    # WRITE EQUALIZED IMAGE TO FILE
    ######################################
    #convert NumPy array to pillow Image 
    eq_img = Image.fromarray(eq_img_array, mode='L')
    return eq_img



    



# segemtation (threshold)
def threshold_image(im,th):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    return thresholded_im

def compute_otsu_criteria(im, th):
    thresholded_im = threshold_image(im,th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1
    if weight1 == 0 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0 * var0 + weight1 * var1

def find_best_threshold(im):
    threshold_range = range(np.max(im)+1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold


#   ---- >Smoothing
# Box/Mean filter 
def box_filter(img, ksize):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Create a kernel for the box filter
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)

    # Create a new image to store the filtered result
    filtered_img = np.zeros((height, width, 3), dtype=np.float32)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the neighborhood of pixels centered around the current pixel
            y_min = max(0, y - ksize//2)
            y_max = min(height - 1, y + ksize//2)
            x_min = max(0, x - ksize//2)
            x_max = min(width - 1, x + ksize//2)
            neighborhood = img[y_min:y_max+1, x_min:x_max+1, :]

            # Compute the average of the pixel values in the neighborhood
            avg_color = np.mean(neighborhood, axis=(0, 1))

            # Set the value of the current pixel in the filtered image
            filtered_img[y, x, :] = avg_color

    # Convert the filtered image to uint8 data type
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img


#min filter

def min_filter(img, ksize):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Create a new image to store the filtered result
    filtered_img = np.zeros((height, width, 3), dtype=np.float32)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the neighborhood of pixels centered around the current pixel
            y_min = max(0, y - ksize//2)
            y_max = min(height - 1, y + ksize//2)
            x_min = max(0, x - ksize//2)
            x_max = min(width - 1, x + ksize//2)
            neighborhood = img[y_min:y_max+1, x_min:x_max+1, :]

            # Compute the mean of the pixel values in the neighborhood
            mean_color = np.min(neighborhood, axis=(0, 1))

            # Set the value of the current pixel in the filtered image
            filtered_img[y, x, :] = mean_color

    # Convert the filtered image to uint8 data type
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img

#max filter

def max_filter(img, ksize):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Create a new image to store the filtered result
    filtered_img = np.zeros((height, width, 3), dtype=np.float32)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the neighborhood of pixels centered around the current pixel
            y_min = max(0, y - ksize//2)
            y_max = min(height - 1, y + ksize//2)
            x_min = max(0, x - ksize//2)
            x_max = min(width - 1, x + ksize//2)
            neighborhood = img[y_min:y_max+1, x_min:x_max+1, :]

            # Compute the mean of the pixel values in the neighborhood
            mean_color = np.max(neighborhood, axis=(0, 1))

            # Set the value of the current pixel in the filtered image
            filtered_img[y, x, :] = mean_color

    # Convert the filtered image to uint8 data type
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img


#median filter

def median_filter(img, ksize):
    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Create a new image to store the filtered result
    filtered_img = np.zeros((height, width, 3), dtype=np.float32)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the neighborhood of pixels centered around the current pixel
            y_min = max(0, y - ksize//2)
            y_max = min(height - 1, y + ksize//2)
            x_min = max(0, x - ksize//2)
            x_max = min(width - 1, x + ksize//2)
            neighborhood = img[y_min:y_max+1, x_min:x_max+1, :]

            # Compute the mean of the pixel values in the neighborhood
            mean_color = np.median(neighborhood, axis=(0, 1))

            # Set the value of the current pixel in the filtered image
            filtered_img[y, x, :] = mean_color

    # Convert the filtered image to uint8 data type
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img


#canny Edge Detection 



def canny(img_path):
    # Load image into variable and display it
    lion = Image.open(img_path).convert(mode = "L") # Paste address of image

    lion = np.array(lion)
    # Convert color image to grayscale to help extraction of edges and plot it
    #lion_gray = np.dot(lion[...,:3], [0.299, 0.587, 0.114])
    lion_gray = lion
    lion_gray = lion_gray.astype('int32')
    
    # Blur the grayscale image so that only important edges are extracted and the noisy ones ignored
    lion_gray_blurred = ndimage.gaussian_filter(lion_gray, sigma=1.4) # Note that the value of sigma is image specific so please tune it
    # Apply Sobel Filter using the convolution operation
    # Note that in this case I have used the filter to have a maximum amgnitude of 2, but it can also be changed to other numbers for aggressive edge extraction
    # For eg [-1,0,1], [-5,0,5], [-1,0,1]
    def SobelFilter(img, direction):
        if(direction == 'x'):
            Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
            Res = ndimage.convolve(img, Gx)
            #Res = ndimage.convolve(img, Gx, mode='constant', cval=0.0)
        if(direction == 'y'):
            Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
            Res = ndimage.convolve(img, Gy)
            #Res = ndimage.convolve(img, Gy, mode='constant', cval=0.0)
    
        return Res
    # Normalize the pixel array, so that values are <= 1
    def Normalize(img):
        #img = np.multiply(img, 255 / np.max(img))
        img = img/np.max(img)
        return img
    
    # Apply Sobel Filter in X direction
    gx = SobelFilter(lion_gray_blurred, 'x')
    gx = Normalize(gx)
    
    # Apply Sobel Filter in Y direction
    gy = SobelFilter(lion_gray_blurred, 'y')
    gy = Normalize(gy)
    
    # Apply the Sobel Filter
    dx = ndimage.sobel(lion_gray_blurred, axis=1) # horizontal derivative
    dy = ndimage.sobel(lion_gray_blurred, axis=0) # vertical derivative
    
    # Calculate the magnitude of the gradients obtained
    Mag = np.hypot(gx,gy)
    Mag = Normalize(Mag)
    
    # Calculate the magnitude of the gradients 
    mag = np.hypot(dx,dy)
    mag = Normalize(mag)
    
    # Calculate direction of the gradients
    Gradient = np.degrees(np.arctan2(gy,gx))
    
    # Calculate the direction of the gradients
    gradient = np.degrees(np.arctan2(dy,dx))
    # Do Non Maximum Suppression with interpolation to get a better estimate of the magnitude values of the pixels in the gradient direction
    # This is done to get thin edges
    def NonMaxSupWithInterpol(Gmag, Grad, Gx, Gy):
        NMS = np.zeros(Gmag.shape)
        
        for i in range(1, int(Gmag.shape[0]) - 1):
            for j in range(1, int(Gmag.shape[1]) - 1):
                if((Grad[i,j] >= 0 and Grad[i,j] <= 45) or (Grad[i,j] < -135 and Grad[i,j] >= -180)):
                    yBot = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                    yTop = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 45 and Grad[i,j] <= 90) or (Grad[i,j] < -90 and Grad[i,j] >= -135)):
                    yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                    yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 90 and Grad[i,j] <= 135) or (Grad[i,j] < -45 and Grad[i,j] >= -90)):
                    yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                    yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
                if((Grad[i,j] > 135 and Grad[i,j] <= 180) or (Grad[i,j] < 0 and Grad[i,j] >= -45)):
                    yBot = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                    yTop = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                    if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                        NMS[i,j] = Gmag[i,j]
                    else:
                        NMS[i,j] = 0
    
        return NMS
    
    NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
    NMS = Normalize(NMS)
    
    # Double threshold Hysterisis
    # Note that I have used a very slow iterative approach for ease of understanding, a faster implementation using recursion can be done instead
    # This recursive approach would recurse through every strong edge and find all connected weak edges
    def DoThreshHyst(img):
        highThresholdRatio = 0.2  
        lowThresholdRatio = 0.15 
        GSup = np.copy(img)
        h = int(GSup.shape[0])
        w = int(GSup.shape[1])
        highThreshold = np.max(GSup) * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio    
        x = 0.1
        oldx=0
    
        # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
        while(oldx != x):
            oldx = x
            for i in range(1,h-1):
                for j in range(1,w-1):
                    if(GSup[i,j] > highThreshold):
                        GSup[i,j] = 1
                    elif(GSup[i,j] < lowThreshold):
                        GSup[i,j] = 0
                    else:
                        if((GSup[i-1,j-1] > highThreshold) or 
                            (GSup[i-1,j] > highThreshold) or
                            (GSup[i-1,j+1] > highThreshold) or
                            (GSup[i,j-1] > highThreshold) or
                            (GSup[i,j+1] > highThreshold) or
                            (GSup[i+1,j-1] > highThreshold) or
                            (GSup[i+1,j] > highThreshold) or
                            (GSup[i+1,j+1] > highThreshold)):
                            GSup[i,j] = 1
            x = np.sum(GSup == 1)
    
        GSup = (GSup == 1) * GSup # This is done to remove/clean all the weak edges which are not connected to strong edges
    
        return GSup
    
    Final_Image = DoThreshHyst(NMS)
    Final_Image = Final_Image.astype(np.uint8)
    return Final_Image



# Sobel Edge detection

def sobel(img_path , direction = "both"):
    
    image_file = img_path
    input_image = imread(img_path)

    if(len(np.shape(input_image)) == 3):
        #if the image is RGB we convert it to grayscale using the gamma correction method
        [nx, ny, nz] = np.shape(input_image)  
        r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
        gamma = 1.400  
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  
        grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma
        
    else:
        grayscale_image = input_image
        
        
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(grayscale_image)  
    sobel_filtered_image = np.zeros(shape=(rows, columns))
    sobel_x_image = np.zeros(shape=(rows, columns))
    sobel_y_image = np.zeros(shape=(rows, columns))
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
            sobel_x_image[i + 1, j + 1] =gx
            sobel_y_image[i + 1, j + 1] =gy
            
    if(direction == "both"):
        return sobel_filtered_image
    elif(direction == "x"):
        return sobel_x_image
    elif(direction == "y"):
        return sobel_y_image
    
#laplacian filter 

def laplacian_filter(img_path , inhanced = False):
    #read the image as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Define Laplacian filter kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    
    if (inhanced == True):
         kernel[1][1] += 1
    
    # Compute Laplacian filter using convolution
    filtered_img = np.zeros_like(img)
    padded_img = np.pad(img, ((1,1), (1,1)), mode='constant')
    for i in range(1, img.shape[0]+1):
        for j in range(1, img.shape[1]+1):
            s = np.sum(padded_img[i-1:i+2,j-1:j+2] * kernel)
            if(s<0):
                s = 0
            elif (s>255):
                s = 255
            filtered_img[i-1,j-1] = s

    # Convert back to unsigned 8-bit integer
    filtered_img = np.uint8(np.absolute(filtered_img))

    return filtered_img



def ImageNormalize(image):
    # Subtract minimum value from all pixels
    min_val = np.min(image)
    normalized_image = image - min_val

    # Scale by a factor to ensure maximum pixel value is 255
    max_val = np.max(normalized_image)
    if max_val > 0:
        normalized_image = normalized_image * (255.0 / max_val)

    return normalized_image.astype(np.uint8)













