# Deploying a CIFAR-10 CNN in ML Engine

Based from [Tensorflow's CNN tutorial](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10) and [mltoolbox_datalab_image_classification](https://github.com/googledatalab/pydatalab/tree/master/solutionbox/image_classification/mltoolbox/image/classification).

1. Build package wheel

    ```sh
    rm -r build dist
    python setup.py bdist_wheel --universal
    ```

2. Submit training job to ML Engine

    ```sh
    gcloud ml-engine jobs submit training \
        $JOB_NAME \
        --region $REGION \
        --job-dir $JOB_DIR \
        --packages $TRAINING_PACKAGE_PATH \
        --module-name trainer.task \
        --config config.yaml
    ```

3. Deploy a model version to ML Engine

    ```sh
    gcloud ml-engine models create $MODEL_NAME
    gcloud ml-engine versions create \
        $VERSION_NAME \
        --model $MODEL_NAME \
        --origin $JOB_DIR/model
    ```

4. Predict with deployed model

    ```sh
    gcloud ml-engine predict \
        --model $MODEL_NAME \
        --version $VERSION_NAME \
        --json-instances $JSON_INSTANCES
    ```

