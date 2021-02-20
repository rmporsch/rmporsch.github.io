Model explorations and hyperparameter search with Weights & Biases and Kubernetes
=====================================

In every machine learning project we have to continuously tweek and experiment with our models. This is necessary, not only to further improve performance, but also to explore underlying model characteristics. These constant experiments require rigorous logging and performance tracking. Hence, various different provider have come up with solutions to facilitate this tracking such as Tensorboard, Comet, W&B, as well as others. Here at [Apoidea](https://www.apoidea.ai/) we make use of [W&B](https://www.wandb.com/).

Within this blog post we would like to give a more practical overview of how we run machine learning experiments and track their performance. Specifically, how we quickly set up clusters in the cloud and train our models. We hope this might help others, as well as improve our current practices by enganging in a discussion with the wider machine learning community.

Within this post we will outline the following:
- How we train your model within a kubernetes cluster and how to track the process with W&B
- How we run a W&B sweep to explore hyperparameters

We assume the reader is familiar with deep learning models, docker as well as some basics of GCP [command line](https://cloud.google.com/sdk) tools such as `gcloud`.

# Running experiments in the cloud

Fortunately for many machine learning practitioners there has been a real explosion of tools to facilitate machine learning and deep learning development and deployment. 
These tools often allow you to not only deploy your training applications in the cloud but also to fine-tune hyperparameters as well as deploy your models in production.
These tools, such as [SageMaker](https://aws.amazon.com/sagemaker/) from AWS or Google's [AI Platform](https://cloud.google.com/ai-platform/training/docs/overview), are great for fine tuning well established models on known problems but come rather short when doing more research focused development.
In addition, these products are often rather pricy.

Lukely it is not very difficult to deploy your own deep learning cluster which will give you a more fine-grained control over your experiments (and potentially will safe you money).
Hence, we will give you a step by step tutorial on how we run deep learning experiments in the cloud with the help of Kubernetes.

# Deploying on a Kubernetes cluster

Let us first discuss what [Kubernetes](https://kubernetes.io/) is. Kubernetes is a "portable, extensible, open-source platform for managing containerized workloads and services, that facilitates both declarative configuration and automation".

Kubernetes has been used for a number of different deployments and it would be outside the scope of this blog post to discuss all its components.
In essence it allows you to manage  your containerised training models on a cluster.
The great benefits of Kubernetes from a machine learning point of view are that it provides you with:

- Storage orchestration
Kubernetes allows you to automatically mount a storage system. This can include your training, validation and test data as well as a storage point to save your finished models.
- Secret and configuration management
Kubernetes allows you securly share secret keys and configurations accross all your trainining applications
- Automatic bin packing
You provide Kubernetes with a cluster of machines that it can use to run your containerized training tasks. You tell Kubernetes how much CPU and memory (RAM), GPU each container needs. Kubernetes can then fit your training applications onto your machines and make the best use of your resources.

## Getting your code ready

As we have seen above Kubernetes makes use of dockerized applications, hence it needs a version of your training code within a docker container.
We will not go into great detail how to do this since there are some great online resources available (see [here](https://towardsdatascience.com/build-a-docker-container-with-your-machine-learning-model-3cf906f5e07e) or [here](https://medium.com/analytics-vidhya/deploy-your-machine-learning-model-on-docker-ee2b931e133c) for example).

However, lets assume we have the following basic dockerfile available already:

```dockerfile
FROM gcr.io/deeplearning-platform-release/pytorch-gpu
RUN pip install wandb # install W&B resources
COPY model-training-code /train
CMD ["python", "/train/trainer.py"]
```

Hence, we have a dockerized version of our training application ready.
Within this application we log our training process with W&B (see [here](https://docs.wandb.com/library/log) on how to do this).

## Making your dockerized application accesible 

Next we need to push the dockerized version of the application to a private docker repository.
This can be done via Google's [Container Registry](https://cloud.google.com/container-registry), which doesn't cost anything expect storage.

This is again done quite simple. Just run the following code which will 
1. Build the container
2. Push the container to the google container registry.

```bash
#!/bin/bash

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_REPO_NAME=pytorch_custom_container
IMAGE_TAG="latest"
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

docker build -f Dockerfile -t "${IMAGE_URI}" ./

docker push "${IMAGE_URI}"
```

Now your containerized version is accesible throught out your GCP project. However, we now need to create a cluster of machines in order to train our model.

## Starting a GCP Cluster

Lets start up a simple cluster with only one node. We will use a `n1-standard` machine with 4 CPUs as well as a `nividia-tesla-p100` for GPU training.

```bash
#!/bin/bash
name_of_your_cluster="pytorch-training-cluster"
gcloud container clusters create $name_of_your_cluster \
    --num-nodes=1 \
    --zone=asia-east1-a \
    --accelerator="type=nvidia-tesla-p100,count=1" \
    --machine-type="n1-standard-4" \
    --scopes="gke-default,storage-rw"

# install gpu drivers across all machines
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

Do not forget to run the last line of that code since it will install all the necessary nividia drivers on each machine within the cluster.

## Running your container

Now we have an up and running cluster of nodes with GPU support. Next we need to run our container. Container specifications as well as individual resource requests are specified in a `yaml` files. For example the following shows an example how one could configurate your the deployment of your container:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gke-training-pod
spec:
  restartPolicy: Never
  containers:
  - name: my-custom-container
    image: url_to_container_image
    resources:
      limits:
        nvidia.com/gpu: 1
    env:
    - name: WANDB_API_KEY
      valueFrom:
        secretKeyRef:
          name: wandb-secret
          key: secret
```

Lets break down the aspects of this `yaml` file:

- under `container` we are specifying the container configurations we would like use for this deployment
- `resources` indicated the requested resources for the container, such as GPU, CPU and memory
- enviornmental variables for our container are set under `env`
- `image` configurates from where we should pull our docker container

As you can see we define the environmental variable `WAND_API_KEY` witin the `yaml` describtion. This will allow us to deploy our secret W&B key into the container without storing it anywhere in clear text format.

Indeed there are various ways we can store secretes (for a full overview please see [here](https://kubernetes.io/docs/concepts/configuration/secret/)). In this particular example we have chosen to deploy the secret as an environmental variable. This can be done with a separate `yaml` file which is then distributed across all nodes within the cluster. Let us see how to do that:

1. Convert your `wandb` key into `base64` with `echo -n 'my-wandb-key' | base64`

2. create a new `yaml` file, lets call it `wandb_kubernetes.yaml` which looks something like this:

   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: wandb-secret
   data:
     secret: your_secret_in_base64
   ```

3. Deploy the secret with `kubectl apply -f wandb_kubernetes.yaml`

Please make sure not to add this `wandb_kubernetes.yaml` in your git repository.

Now we are ready to deploy our container on the GCP cluster. Simply run `kubectl apply -f pod.yaml`. Your containerized training application should run on your cluster and log training metrics on W&B.

# Hyperparamenter tuning with W&B sweeps

Now since we have setup our cluster and container it is an easy step up multiple runs in parallel with slightly different parameter in order to explore our model or fine-tune hypterparamenters. We use [W&Bs sweep](https://docs.wandb.com/sweeps) which will help us to do this kind of exploration in an automated fashion.

## Setting up your experiment

W&B expects that parameters within your training script can be changed via the command line. Hence your training script will need to be able to accept paramenters such as the following:

```bash
python train/trainer.py --learning_rate=0.005 --optimizer=adam
python train/trainer.py --learning_rate=0.03 --optimizer=sgd
```

This can easily be achived with packages such as `argparse`.

We can then setup our experimentation paramenter space with a simple `yaml` file:

```yaml
program: train/trainer.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```

Then running `wandb sweep sweep.yaml` will initialise the sweep (but not yet run any code) which will give you your **sweep id**. Please take a look at the [W&B wiki](https://docs.wandb.com/sweeps/quickstart#2-sweep-config) for a more thorough explanation on how to configure your sweep.

## Running your experiments

As previously, we set up our kubernetes cluster. However, this time we increase the number of nodes from 1 to 4.

```bash
#!/bin/bash
name_of_your_cluster="pytorch-training-cluster"
gcloud container clusters create $name_of_your_cluster \
    --num-nodes=4 \
    --zone=asia-east1-a \
    --accelerator="type=nvidia-tesla-p100,count=1" \
    --machine-type="n1-standard-4" \
    --scopes="gke-default,storage-rw"

# install gpu drivers across all machines
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
# deploy secrets for W&B
kubectl apply -f wandb_secret.kubernetes.yaml
```

Since now W&B will orchestra the training we need to connect each pod to the W&B server. This is simply done with `wandb agent your_sweep_id`. We can automatically call this command in all our deployments by  simply overwrite the `CMD` field of our docker container within our deployment `yaml` specification:

```yaml
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  name: sweep-model-quality
spec:
  replicas: 4
  selector:
    matchLabels:
      app: model-quality
  template:
    metadata:
      labels:
        app: model-quality
    spec:
      containers:
      - name: model-quality
        image: url_to_container
        resources:
          limits:
            nvidia.com/gpu: 1
        command: ["wandb", "agent", "your_sweep_id"]
        env:
          - name: WANDB_API_KEY
            valueFrom:
              secretKeyRef:
                name: wandb-secret
                key: secret
```

As you can see above we have now specified that kubernetes should replicated our container 4 times and changed the `command` filed to call `wandb agent your_sweep_id`. This will let the container connect to the W&B server which will then orchestrate the hyperparameter search.

And that is it. You can now go to your W&B side to monitor all your models.

![From the W&B website](https://paper-attachments.dropbox.com/s_5D8914551A6C0AABCD5718091305DD3B64FFBA192205DD7B3C90EC93F4002090_1579066494222_image.png)

# Further readings

- [W&B docs]( https://docs.wandb.com/docs/started.html) General documentation of W&B
- [Kuberflow](https://github.com/kubeflow/kubefl) A machine learningt toolkit for Kubernetes which currently only supports tensorflow
- [An tutorial](https://towardsdatascience.com/using-docker-kubernetes-to-host-machine-learning-models-780a501fda49) on how to deploy a model with Flask on Kubernetes
- [Short tutorial](https://cloud.google.com/ai-platform/deep-learning-containers/docs/kubernetes-container) by Google on how to train models on Kubernetes
