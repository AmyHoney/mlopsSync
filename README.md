# mlopsSync
Code Notebook dockerfile for different frameworks.

### Tensorfolw example
#### Build image from Dockerfile:cpu
docker build -t harbor-repo.vmware.com/zyajing/tf-jupyter-260:v1.0 .

#### text generation
kubectl cp text_generation.ipynb tf-text-generation-0:/home/jovyan/text_generation.ipynb -n kubeflow-user-example-com

Please refer to https://www.tensorflow.org/text/tutorials/text_generation

### Pytorch example
#### Build image from Docker:cpu
kubectl cp torch_sgd_mnist.ipynb torch-sgd-mnist-0:quickstart_tutorial.ipynb -n kubeflow-user-example-com

Please refer to https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


### paddlepaddle example
#### Build image from Dockerfile:cpu
docker build -t harbor-repo.vmware.com/zyajing/paddle-jupyter-22:v1.0 .

#### LeNet model using mnist dataset to classification pictures
kubectl cp paddle-mnist-imageclassification.ipynb paddle-lenet-mnist-0:/home/jovyan/paddle-mnist-imageclassification.ipynb -n kubeflow-user-example-com

Please refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/cv/image_classification.html
