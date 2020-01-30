# RainNet in the wild: some examples of using convolutional neural network for precipitation nowcasting

The list of so far available examples:

* Using RainNet for operational precipitation nowcasting
* Training RainNet from scratch (come up soon)

## Using RainNet for operational precipitation nowcasting

In this example, we want to demonstrate how the pretrained RainNet model can be used for operational precipitation nowcasting in Germany. To this aim, we prepared [Google Colab notebook](https://colab.research.google.com/drive/13n7wULFSil_4wDLKxjZWbhdTMo1W9_cq) showing the entire nowcasting workflow as follows:

1. Downloading the RainNet's source code from the GitHub repository: https://github.com/hydrogo/rainnet;
2. Downloading the RainNet's pretrained weights from the Zenodo repository: https://zenodo.org/record/3630429;
3. Installing `wradlib` library (https://docs.wradlib.org/) for reading the RY composite data; 
4. Importing required libraries, such as `keras`, `numpy` etc.; 
5. Building the RainNet model and loading its pretrained weights;
6. Downloading the latest RY radar composite from the DWD Open Data Repository: https://opendata.dwd.de/weather/radar/radolan/ry/;
7. Developing utils for radar data pre- and postprocessing;
8. Running the RainNet model to produce precipitation nowcasts for the next hour (in 5 minutes intervals).

The developed [Google Colab notebook](https://colab.research.google.com/drive/13n7wULFSil_4wDLKxjZWbhdTMo1W9_cq) allows running RainNet using  `GPU runtime`. That makes the computation of nowcast in order of magnitude faster than using standard CPU and feasible for operational use.

If you have any questions regarding the provided example, you can leave comments right inside the [notebook](https://colab.research.google.com/drive/13n7wULFSil_4wDLKxjZWbhdTMo1W9_cq) or create an [issue](https://github.com/hydrogo/rainnet/issues) on [RainNet's GitHub repository](https://github.com/hydrogo/rainnet/).
