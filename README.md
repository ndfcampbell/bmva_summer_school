# BMVA Summer School
PyTorch Lab for the BMVA Summer School

Please read the PDF for the introduction to PyTorch.

Use Google colaboratory to run the notebooks!

[http://colab.research.google.com/](http://colab.research.google.com/)

# How to open the demos in Google colab:

- Open a browser and go to [http://colab.research.google.com/](http://colab.research.google.com/)

- Select "Open Notebook" from the menu

![](google_colab_screen_1.png)

- Select "GitHub" and type "`ndfcampbell/bmva_summer_school`" into the address bar

![](google_colab_screen_2.png)

- Load the PyTorch_Example file first and then take a look at the linear regression lab!

## Extension:

- If you are curious as to how PyTorch (and similar libraries) work behind the scenes then please take a look at the Scalar Auto Grad Demo file!

## Advanced:

- If you are already familiar with PyTorch then you may be interested in the advanced notebook on Vision Foundation Models kindly provided by [Li (Luis) Li](https://www.luisli.org); please see the instructions within the file, in particular using GPU acceleration when loading on the colab server.

- There is also a folder (`SimpleDiffusion`) with an example of a diffusion model kindly provided by [Teo Deveney](https://researchportal.bath.ac.uk/en/persons/teo-deveney); this is currently setup to run on your local machine but you can take the contents of train.py and sample.py to run in collab if you want - there is also a checkpoint to restore the weights from if you just want to run the sampler. This code is based on the [tutorial from Yang Song](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3), referencing [their paper](https://arxiv.org/abs/2006.09011), that you could also work through instead.
