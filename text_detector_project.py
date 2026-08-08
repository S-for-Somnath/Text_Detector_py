import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib as mpl

im_width = 75
im_height = 75
use_normalized_coordinates = True


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color="red",
    thickness=1,
    use_normalized_coordinates=True,
):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    if use_normalized_coordinates:
        left = xmin * img_w
        right = xmax * img_w
        top = ymin * img_h
        bottom = ymax * img_h
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax

    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )


def draw_bounding_boxes_on_image(
    image,
    boxes,
    color=None,
    thickness=1,
    display_str_list=None,
):
    if color is None:
        color = ["red"] * len(boxes)
    if display_str_list is None:
        display_str_list = [""] * len(boxes)

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")

    for i in range(boxes_shape[0]):
        ymin, xmin, ymax, xmax = boxes[i]
        draw_bounding_box_on_image(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color[i] if i < len(color) else "red",
            thickness=thickness,
            use_normalized_coordinates=True,
        )


def draw_bounding_boxes_on_image_array(
    image,
    boxes,
    color=None,
    thickness=1,
    display_str_list=None,
):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image_pil = Image.fromarray(image).convert("RGB")
    draw_bounding_boxes_on_image(
        image_pil, boxes, color=color, thickness=thickness, display_str_list=display_str_list
    )
    return np.array(image_pil)


def dataset_to_numpy_util(training_dataset, validation_dataset):
    if tf.executing_eagerly():
        for validation_digits, (validation_labels, validation_bboxes) in validation_dataset.take(1):
            validation_digits = validation_digits.numpy()
            validation_labels = validation_labels.numpy()
            validation_bboxes = validation_bboxes.numpy()
            break

        for training_digits, (training_labels, training_bboxes) in training_dataset.take(1):
            training_digits = training_digits.numpy()
            training_labels = training_labels.numpy()
            training_bboxes = training_bboxes.numpy()
            break
    else:
        raise RuntimeError("Run in eager mode.")

    validation_labels = np.argmax(validation_labels, axis=1)
    training_labels = np.argmax(training_labels, axis=1)

    return (
        training_digits,
        training_labels,
        training_bboxes,
        validation_digits,
        validation_labels,
        validation_bboxes,
    )


MATPLOTLIB_FONT_DIR = os.path.join(mpl.get_data_path(), "fonts", "ttf")


def create_digits_from_local_fonts(n):
    font_labels = []
    img = Image.new("LA", (75 * n, 75), color=255)
    font1 = ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, "DejaVuSansMono-Oblique.ttf"), 25)
    font2 = ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, "STIXGeneral.ttf"), 25)
    d = ImageDraw.Draw(img)

    for i in range(n):
        font_labels.append(i % 10)
        d.text(
            (7 + i * 75, 0 if i < 10 else -4),
            str(i % 10),
            fill=(255, 255),
            font=font1 if i < 10 else font2,
        )

    font_digits = np.array(img.getdata(), np.float32)[:, 0] / 255.0
    font_digits = np.reshape(font_digits, [75, 75 * n])
    font_digits = np.concatenate(np.split(font_digits, n, axis=1), axis=0)
    font_digits = np.reshape(font_digits, [n, 75 * 75])
    return font_digits, font_labels


def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title, iou_threshold=0.5):
    n = 10
    indexes = np.random.choice(len(predictions), size=n, replace=False)

    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = iou[indexes] if len(iou) > 0 else []
    n_pred_bboxes = pred_bboxes[indexes] if len(pred_bboxes) > 0 else []
    n_bboxes = bboxes[indexes] if len(bboxes) > 0 else []

    n_digits = (n_digits * 255.0).reshape(n, 75, 75)

    fig = plt.figure(figsize=(20, 4))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    for i in range(n):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        colors = []
        labels_to_plot = []

        if len(n_bboxes) > i:
            bboxes_to_plot.append(n_bboxes[i])
            colors.append("green")
            labels_to_plot.append("True")

        if len(n_pred_bboxes) > i:
            bboxes_to_plot.append(n_pred_bboxes[i])
            colors.append("red")
            labels_to_plot.append("Pred")

        img_to_draw = draw_bounding_boxes_on_image_array(
            image=n_digits[i],
            boxes=np.array(bboxes_to_plot) if bboxes_to_plot else np.empty((0, 4)),
            color=colors,
            display_str_list=labels_to_plot,
        )

        ax.set_xlabel(str(n_predictions[i]))
        ax.set_xticks([])
        ax.set_yticks([])

        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color("red")

        ax.imshow(img_to_draw, cmap="gray")

        if len(n_iou) > i:
            color = "red" if n_iou[i][0] < iou_threshold else "black"
            ax.text(0.2, -0.3, f"iou: {n_iou[i][0]:.3f}", color=color, transform=ax.transAxes)

    plt.show()


def plot_metrics(history, metric_name, title):
    plt.figure()
    plt.title(title)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_" + metric_name], color="green", label="val_" + metric_name)
    plt.legend()
    plt.show()


strategy = tf.distribute.get_strategy()
BATCH_SIZE = 64 * strategy.num_replicas_in_sync


def read_image_tfds(image, label):
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)

    image = tf.reshape(image, (28, 28, 1))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0

    xmin_f = tf.cast(xmin, tf.float32) / 75.0
    ymin_f = tf.cast(ymin, tf.float32) / 75.0
    xmax_f = tf.cast(xmin + 28, tf.float32) / 75.0
    ymax_f = tf.cast(ymin + 28, tf.float32) / 75.0

    bbox = tf.stack([ymin_f, xmin_f, ymax_f, xmax_f])
    return image, (tf.one_hot(label, 10), bbox)


def get_training_dataset():
    dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_validation_dataset():
    dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
    dataset = dataset.map(read_image_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(10000, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


with strategy.scope():
    training_dataset = get_training_dataset()
    validation_dataset = get_validation_dataset()

(
    training_digits,
    training_labels,
    training_bboxes,
    validation_digits,
    validation_labels,
    validation_bboxes,
) = dataset_to_numpy_util(training_dataset, validation_dataset)

display_digits_with_boxes(
    training_digits,
    training_labels,
    training_labels,
    np.array([]),
    training_bboxes,
    np.array([]),
    "Training Digits & Labels",
)


def feature_extraction(inputs):
    x = tf.keras.layers.Conv2D(16, activation="relu", kernel_size=1)(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    return x


def dense_layer(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    return x


def classifier(inputs):
    return tf.keras.layers.Dense(10, activation="softmax", name="classification")(inputs)


def bounding_box_regression(inputs):
    return tf.keras.layers.Dense(4, name="bounding_box")(inputs)


def final_model(inputs):
    feature_cnn = feature_extraction(inputs)
    dense_output = dense_layer(feature_cnn)

    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regression(dense_output)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[classification_output, bounding_box_output],
    )


def define_and_compile_model(inputs):
    model = final_model(inputs)
    model.compile(
        optimizer="adam",
        loss={
            "classification": "categorical_crossentropy",
            "bounding_box": "mse",
        },
        metrics={
            "classification": "accuracy",
            "bounding_box": "mse",
        },
    )
    return model


with strategy.scope():
    inputs = tf.keras.Input(shape=(75, 75, 1))
    model = define_and_compile_model(inputs)

model.summary()

EPOCHS = 20
steps_per_epoch = 60000 // BATCH_SIZE

history = model.fit(
    training_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=1,
)

eval_results = model.evaluate(validation_dataset, steps=1)
print("\n-----------------------------------------\n")
print("Evaluation:", dict(zip(model.metrics_names, eval_results)))
print("\n-----------------------------------------\n")

plot_metrics(history, "bounding_box_loss", "Bounding Box Loss")
plot_metrics(history, "classification_loss", "Classification Loss")
plot_metrics(history, "classification_accuracy", "Classification Accuracy")


def intersection_over_union(pred_box, true_box):
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = np.split(pred_box, 4, axis=1)
    ymin_true, xmin_true, ymax_true, xmax_true = np.split(true_box, 4, axis=1)

    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)

    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0) * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = pred_box_area + true_box_area - overlap_area

    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    return iou


prediction = model.predict(validation_digits, batch_size=64)
prediction_labels = np.argmax(prediction[0], axis=1)
prediction_bboxes = prediction[1]

iou = intersection_over_union(prediction_bboxes, validation_bboxes)
iou_threshold = 0.5

display_digits_with_boxes(
    validation_digits,
    prediction_labels,
    validation_labels,
    prediction_bboxes,
    validation_bboxes,
    iou,
    "True and Pred values",
    iou_threshold=iou_threshold,
)
