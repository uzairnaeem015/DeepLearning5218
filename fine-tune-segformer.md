

# Fine-Tune a Semantic Segmentation Model with a Custom Dataset

Semantic segmentation is the task of classifying each pixel in an image. You can see it as a more precise way of classifying an image. It has a wide range of use cases in fields such as medical imaging and autonomous driving. For example, for our pizza delivery robot, it is important to know exactly where the sidewalk is in an image, not just whether there is a sidewalk or not.

Because semantic segmentation is a type of classification, the network architectures used for image classification and semantic segmentation are very similar. In 2014, [a seminal paper](https://arxiv.org/abs/1411.4038) by Long et al. used convolutional neural networks for semantic segmentation. More recently, Transformers have been used for image classification (e.g. [ViT](https://huggingface.co/blog/fine-tune-vit)), and now they're also being used for semantic segmentation, pushing the state-of-the-art further.

[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) is a model for semantic segmentation introduced by Xie et al. in 2021. It has a hierarchical Transformer encoder that doesn't use positional encodings (in contrast to ViT) and a simple multi-layer perceptron decoder. SegFormer achieves state-of-the-art performance on multiple common datasets. Let's see how our pizza delivery robot performs for sidewalk images.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="Pizza delivery robot segmenting a scene" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/pizza-scene.png"></medium-zoom>
</figure>



# 1. Import the dataset:

The first step in any ML project is assembling a good dataset. In order to train a semantic segmentation model, we need a dataset with semantic segmentation labels. We can either use an existing dataset from the Hugging Face Hub, such as [ADE20k](https://huggingface.co/datasets/scene_parse_150), or create our own dataset.

``` python
import tensorflow_datasets as tfds
dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

```

In the dataset, the label consists of two parts: a list of annotations and a segmentation bitmap. The annotation corresponds to the different objects in the image. For each object, the annotation contains an `id` and a `category_id`. The segmentation bitmap is an image where each pixel contains the `id` of the object at that pixel. More information can be found in the [relevant docs](https://docs.segments.ai/reference/sample-and-label-types/label-types#segmentation-labels).

For semantic segmentation, we need a semantic bitmap that contains a `category_id` for each pixel. We'll use the `get_semantic_bitmap` function from the Segments.ai SDK to convert the bitmaps to semantic bitmaps. To apply this function to all the rows in our dataset, we'll use [`dataset.map`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.map). 


The SegFormer model we're going to fine-tune later expects specific names for the features. For convenience, we'll match this format now. Thus, we'll rename the `image` feature to `pixel_values` and the `label.segmentation_bitmap` to `label` and discard the other features.


Let's shuffle the dataset and split the dataset in a train and test set.


```python
auto = tf.data.AUTOTUNE
batch_size = 4

train_ds = (
    dataset["train"]
    .cache()
    .shuffle(batch_size * 10)
    .map(load_image, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)
test_ds = (
    dataset["test"]
    .map(load_image, num_parallel_calls=auto)
    .batch(batch_size)
    .prefetch(auto)
)
```

We'll extract the number of labels and the human-readable ids, so we can configure the segmentation model correctly later on.


```python
import json
from huggingface_hub import hf_hub_download

repo_id = f"datasets/{hf_dataset_identifier}"
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
```

## Feature extractor, data augmentation, and Normalization

A SegFormer model expects the input to be of a certain shape. To transform our training data to match the expected shape, we can use `SegFormerFeatureExtractor`. We could use the `ds.map` function to apply the feature extractor to the whole training dataset in advance, but this can take up a lot of disk space. Instead, we'll use a *transform*, which will only prepare a batch of data when that data is actually used (on-the-fly). This way, we can start training without waiting for further data preprocessing.

In our transform, we'll also define some data augmentations to make our model more resilient to different lighting conditions.


```python
def load_image(datapoint, augment=True):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )
    if augment:
        # Randomly flip the image horizontally
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        # Randomly adjust the brightness and saturation of the image
        input_image = tf.image.random_brightness(input_image, 0.1)
        input_image = tf.image.random_saturation(input_image, 0.8, 1.2)

    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))

    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}
    
    def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )

    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))
    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}
```

# 3. Fine-tune a SegFormer model

## Load the model to fine-tune

The SegFormer authors define 5 models with increasing sizes: B0 to B5. The following chart (taken from the original paper) shows the performance of these different models on the ADE20K dataset, compared to other models.

<figure class="image table text-center m-0 w-full">
  <medium-zoom background="rgba(0,0,0,.7)" alt="SegFormer model variants compared with other segmentation models" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/56_fine_tune_segformer/segformer.png"></medium-zoom>
  <figcaption><a href="https://arxiv.org/abs/2105.15203">Source</a></figcaption>
</figure>

Here, we'll load the smallest SegFormer model (B0), pre-trained on ImageNet-1k. It's only about 14MB in size!
Using a small model will make sure that our model can run smoothly on our pet dataset.


```python
from transformers import SegformerForSemanticSegmentation

pretrained_model_name = "nvidia/mit-b0" 
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)
```

## Set up the Trainer

To fine-tune the model on our data, we'll use dynamic learning rate, normilzation and augmentation



```python
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    """
    A function that takes an epoch index as input and returns a new learning rate
    for that epoch. This function can be customized to implement various schedules,
    such as step decay or exponential decay.
    """
    lr = 0.001
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    input_mask -= 1
    return input_image, input_mask
    
    def load_image(datapoint, augment=True):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )
    if augment:
        # Randomly flip the image horizontally
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        # Randomly adjust the brightness and saturation of the image
        input_image = tf.image.random_brightness(input_image, 0.1)
        input_image = tf.image.random_saturation(input_image, 0.8, 1.2)

    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.transpose(input_image, (2, 0, 1))

    return {"pixel_values": input_image, "labels": tf.squeeze(input_mask)}

```

Next, we'll define a function that computes the evaluation metric we want to work with. Because we're doing semantic segmentation, we'll use the [mean Intersection over Union (mIoU)]


```python
def evaluate(model, dataset):
    ious = []
    for sample in dataset:
        images, masks = sample["pixel_values"], sample["labels"]
        pred_masks = model.predict(images).logits
        pred_masks = create_mask(pred_masks)
        masks = tf.image.resize(masks, (128, 128), method='bicubic')
        masks = tf.expand_dims(masks[..., 0], axis=-1)
        iou = calculate_iou(masks, pred_masks)
        ious.append(iou)
        mean_iou = tf.reduce_mean(ious)
        return mean_iou

mean_iou = evaluate(model, test_ds)
print("Mean IoU:", mean_iou.numpy())
```

Finally, we can instantiate a `training` object.


```python
lr = 0.00006
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer)

epochs = 30

history = model.fit(
    train_ds,
    validation_data=test_ds,
    callbacks=[DisplayCallback(test_ds)],
    epochs=epochs,
)
```

during training we can see our prediction after each epoch using the callback implementation.

```python
class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))
```


# 4. Conclusion

That's it! We have have trained and fined tuned the model, Thank you
