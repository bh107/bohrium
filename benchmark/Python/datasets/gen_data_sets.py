import numpy
from PIL import Image

def gen_27_data(size, name='heat'):
    data = numpy.zeros((size,size,size))
    mid = size/2
    data[mid-8:mid+8,mid-8:mid+8]=100.0
    numpy.save(name,data)

def gen_filter(size, weight):
    kernel = numpy.zeros((size,size))
    totalsum=0
    kernelrad = size/2
    distance=0
    caleuler = 1.0 /(2.0 * numpy.pi * weight**2)
    for filterY in range(-kernelrad, kernelrad+1, 1): 
        for filterX in range(-kernelrad, kernelrad+1, 1): 
            distance = ((filterX * filterX)+(filterY*filterY))/(2 * (weight * weight)) 
            kernel[filterY + kernelrad,filterX + kernelrad] = caleuler * numpy.exp(-distance) 
            totalsum += kernel[filterY + kernelrad, filterX + kernelrad] 
    kernel *=(1.0/totalsum)
    return kernel

def gen_convolve_data(fsize, photo='Hell.jpg', iname='bigimage', fname='filter'):
    img = Image.open(photo)
    rgb = numpy.array(img)
    tones = numpy.array((0.3, 0.6, 0.11))
    rgb = numpy.add.reduce((rgb*tones[numpy.newaxis, numpy.newaxis, :]), axis=2)
    numpy.save(iname,rgb)
    kernel = gen_filter(fsize,13.0)
    numpy.save(fname,kernel)

