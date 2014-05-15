## Soble Filter on Images

### Goals

Implementing [*Sobel*](http://en.wikipedia.org/wiki/Sobel_operator) edge detection algorithm on CPU completely. And also implementing the filtering part of the *Sobel* algorithm for running on **GPU** using **CUDA**. Some performance comparison between completely-on-CPU and partially-on-GPU versions! Why the result is what it is etc.

The result of the filtering and edge detection using *Sobel* algorithm for some sample images is as following:

![Cell image before filtering](https://dl.dropboxusercontent.com/u/100502983/image_filtering_pics/cell-bmp.jpg)

![Cell Image Filtered](https://dl.dropboxusercontent.com/u/100502983/image_filtering_pics/cell_out-bmp.jpg)

![SestonDiatom before filtering](https://dl.dropboxusercontent.com/u/100502983/image_filtering_pics/sestondiatom-bmp.jpg)

![SestonDiatom Filtered](https://dl.dropboxusercontent.com/u/100502983/image_filtering_pics/sestondiatom_out-bmp.jpg)

![Chryseobacterium Filtered](https://dl.dropboxusercontent.com/u/100502983/image_filtering_pics/chryseobacterium_out-bmp.jpg)

![Chryseobacterium Filtered (somewhat inverted version of previous image)](https://dl.dropboxusercontent.com/u/100502983/image_filtering_pics/chryseobacterium_out-bmp-1.jpg)

As you can see we magnified the edges of the above images which is the goal of **Sobel Algorithm**! 

## Performance and Execution Time measurements

The execution time for the CPU and GPU versions of implementation in milliseconds are as following:

```
|    CPU     |    GPU      |        Image         |
|  0.021799  |  0.395673   |       Cell.bmp       |
|  0.073794  |  0.357644   | Chryseobacterium.bmp |
|  0.002312  |  0.341441   |   OnionTelphase.bmp  |
|  0.01477   |  0.344096   |   SestonDiatom.bmp   |
```

As you can see the execution time for the partially-on-GPU version of the implementation is considerably slower in all the cases! There are couple of reasons for that! In this timing the effect of overhead for data transfer back and forth between host and device has not eliminated from the final execution time value and because of that it's not completely accurate to say GPU version is WAY slower! But even without the overhead of data transfer the **GPU** version is not faster than the **CPU** version and the reason is the intensity of the computation we're dealing with and size of images! 

For images with this size-range and an algorithm such as *Sobel* which is not very intense (computation-wise) the GPU and all its added complexity and messiness (for the code) won't bring us benefits!

All in all, it was a very intereesting experience and fun to mess with some CUDA programming and its ideas and concepts behind it! And of course there are tons of applications that can benefit from the power of GPUs for intense computations (lots of **bioinformatics** applications are running on GPUs etc.)
