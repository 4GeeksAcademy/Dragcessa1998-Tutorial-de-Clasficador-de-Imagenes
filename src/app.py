from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"

DEFAULT_RAW_SOURCE_DIR = RAW_DIR
DEFAULT_PREPARED_DIR = PROCESSED_DIR / "dogs_vs_cats"
BEST_MODEL_PATH = MODEL_DIR / "best_dogs_vs_cats_vgg.keras"
FINAL_MODEL_PATH = MODEL_DIR / "dogs_vs_cats_vgg.keras"
SUMMARY_PATH = PROCESSED_DIR / "dogs_vs_cats_training_summary.json"
CURVES_PATH = PROCESSED_DIR / "dogs_vs_cats_training_curves.png"

CLASS_NAMES = ("cat", "dog")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una CNN tipo VGG para clasificar imagenes de perros y gatos."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_SOURCE_DIR,
        help="Carpeta base del dataset. Acepta data/raw/train con archivos cat.* y dog.* o data/raw/cat + data/raw/dog.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
        help="Carpeta donde se crea la estructura train/test por clase.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Numero de epocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de lote.")
    parser.add_argument("--image-size", type=int, default=200, help="Alto y ancho de las imagenes.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion para test.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limita la cantidad total de imagenes para pruebas rapidas.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copia imagenes en lugar de crear enlaces simbolicos.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Solo prepara carpetas y visualizaciones; no entrena el modelo.",
    )
    parser.add_argument(
        "--rebuild-prepared",
        action="store_true",
        help="Reconstruye data/processed/dogs_vs_cats desde cero.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate de Adam.")
    parser.add_argument(
        "--loading-mode",
        choices=("auto", "memory", "directory"),
        default="auto",
        help="Auto selecciona memoria si el equipo tiene >12 GB RAM; tambien puedes forzar memory o directory.",
    )
    parser.add_argument(
        "--ram-threshold-gb",
        type=float,
        default=12.0,
        help="Umbral de RAM total para usar la ruta en memoria cuando loading-mode=auto.",
    )
    return parser.parse_args()


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.utils import img_to_array, load_img, to_categorical
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow no esta instalado. Ejecuta: pip install -r requirements.txt"
        ) from exc

    return {
        "tf": tf,
        "Adam": Adam,
        "Conv2D": Conv2D,
        "Dense": Dense,
        "EarlyStopping": EarlyStopping,
        "Flatten": Flatten,
        "ImageDataGenerator": ImageDataGenerator,
        "MaxPool2D": MaxPool2D,
        "ModelCheckpoint": ModelCheckpoint,
        "Sequential": Sequential,
        "img_to_array": img_to_array,
        "load_img": load_img,
        "load_model": load_model,
        "to_categorical": to_categorical,
    }


def get_total_ram_gb() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0.0
    return (page_size * page_count) / float(1024**3)


def choose_loading_mode(requested_mode: str, ram_threshold_gb: float) -> tuple[str, float]:
    total_ram_gb = get_total_ram_gb()
    if requested_mode == "auto":
        if total_ram_gb > ram_threshold_gb:
            return "memory", total_ram_gb
        return "directory", total_ram_gb
    return requested_mode, total_ram_gb


def detect_raw_dataset(raw_dir: Path) -> dict:
    """Acepta dos estructuras:

    1. data/raw/train con archivos cat.0.jpg y dog.0.jpg
    2. data/raw/cat + data/raw/dog con imagenes dentro de cada carpeta
    """
    candidates = [raw_dir, raw_dir / "train"]

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("cat*.jpg")) and any(candidate.glob("dog*.jpg")):
            return {"layout": "flat", "source": candidate}

    for candidate in candidates:
        cat_dir = candidate / "cat"
        dog_dir = candidate / "dog"
        if cat_dir.exists() and dog_dir.exists() and any(cat_dir.glob("*.jpg")) and any(dog_dir.glob("*.jpg")):
            return {"layout": "structured", "source": candidate}

    raise FileNotFoundError(
        "No encontre un dataset compatible. Usa una de estas estructuras: "
        "data/raw/train con archivos cat.0.jpg y dog.0.jpg, o bien data/raw/cat y data/raw/dog con imagenes dentro."
    )


def get_class_files(dataset_info: dict) -> dict[str, list[Path]]:
    source = dataset_info["source"]
    if dataset_info["layout"] == "flat":
        return {
            class_name: sorted(source.glob(f"{class_name}*.jpg"))
            for class_name in CLASS_NAMES
        }

    return {
        class_name: sorted((source / class_name).glob("*.jpg"))
        for class_name in CLASS_NAMES
    }


def collect_labeled_images(dataset_info: dict, sample_limit: int | None = None) -> list[tuple[Path, str]]:
    class_files = get_class_files(dataset_info)
    labeled_images: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        labeled_images.extend((path, class_name) for path in class_files[class_name])

    if sample_limit is not None:
        per_class = max(1, sample_limit // len(CLASS_NAMES))
        sampled_images = []
        for class_name in CLASS_NAMES:
            class_items = [item for item in labeled_images if item[1] == class_name][:per_class]
            sampled_images.extend(class_items)
        labeled_images = sampled_images

    if not labeled_images:
        raise ValueError(f"No se encontraron imagenes validas en {dataset_info['source']}.")

    return labeled_images


def build_train_test_items(
    dataset_info: dict,
    test_size: float,
    sample_limit: int | None = None,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    labeled_images = collect_labeled_images(dataset_info, sample_limit=sample_limit)
    return split_images(labeled_images, test_size=test_size)


def estimate_image_tuple_gb(
    train_items: list[tuple[Path, str]],
    test_items: list[tuple[Path, str]],
    image_size: int,
) -> float:
    total_images = len(train_items) + len(test_items)
    bytes_per_image = image_size * image_size * 3  # uint8 RGB
    total_bytes = total_images * bytes_per_image
    return total_bytes / float(1024**3)


def split_images(
    labeled_images: list[tuple[Path, str]],
    test_size: float,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    paths = [item[0] for item in labeled_images]
    labels = [item[1] for item in labeled_images]
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_items = list(zip(train_paths, train_labels))
    test_items = list(zip(test_paths, test_labels))
    return train_items, test_items


def link_or_copy_image(source: Path, destination: Path, copy_files: bool) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return

    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prepared_dataset_exists(prepared_dir: Path) -> bool:
    return all(
        (prepared_dir / split / class_name).exists()
        and any((prepared_dir / split / class_name).glob("*.jpg"))
        for split in ("train", "test")
        for class_name in CLASS_NAMES
    )


def prepare_directory_dataset(
    prepared_dir: Path,
    train_items: list[tuple[Path, str]],
    test_items: list[tuple[Path, str]],
    copy_files: bool,
    rebuild_prepared: bool = False,
) -> dict[str, int]:
    if rebuild_prepared and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    if prepared_dataset_exists(prepared_dir):
        return count_prepared_images(prepared_dir)

    for split_name, items in (("train", train_items), ("test", test_items)):
        for source_path, class_name in items:
            destination = prepared_dir / split_name / class_name / source_path.name
            link_or_copy_image(source_path, destination, copy_files=copy_files)

    return count_prepared_images(prepared_dir)


def count_prepared_images(prepared_dir: Path) -> dict[str, int]:
    counts = {}
    for split_name in ("train", "test"):
        for class_name in CLASS_NAMES:
            key = f"{split_name}_{class_name}"
            counts[key] = len(list((prepared_dir / split_name / class_name).glob("*.jpg")))
    return counts


def save_sample_grid(prepared_dir: Path, class_name: str, image_size: int) -> Path:
    sample_paths = sorted((prepared_dir / "train" / class_name).glob("*.jpg"))[:9]
    if len(sample_paths) < 9:
        raise ValueError(f"Hacen falta al menos 9 imagenes de {class_name} para la visualizacion.")

    output_path = PROCESSED_DIR / f"{class_name}_sample_grid.png"
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for axis, image_path in zip(axes.ravel(), sample_paths):
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB").resize((image_size, image_size)))
        axis.set_title(image_path.name, fontsize=8)
        axis.axis("off")

    fig.suptitle(f"Primeras 9 imagenes de {class_name}", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_directory_generators(prepared_dir: Path, image_size: int, batch_size: int, ImageDataGenerator):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    trdata = train_datagen.flow_from_directory(
        prepared_dir / "train",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        classes=list(CLASS_NAMES),
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_STATE,
    )
    tsdata = test_datagen.flow_from_directory(
        prepared_dir / "test",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        classes=list(CLASS_NAMES),
        class_mode="categorical",
        shuffle=False,
    )
    return trdata, tsdata


def load_images_to_tuple(
    items: list[tuple[Path, str]],
    image_size: int,
    keras_modules,
) -> tuple[np.ndarray, np.ndarray]:
    load_img = keras_modules["load_img"]
    to_categorical = keras_modules["to_categorical"]

    images = []
    labels = []
    class_to_index = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}

    for image_path, class_name in items:
        image = load_img(image_path, target_size=(image_size, image_size))
        images.append(np.asarray(image, dtype=np.uint8))
        labels.append(class_to_index[class_name])

    image_array = np.asarray(images, dtype=np.uint8)
    label_array = to_categorical(labels, num_classes=len(CLASS_NAMES))
    return image_array, label_array


def create_memory_generators(
    train_items: list[tuple[Path, str]],
    test_items: list[tuple[Path, str]],
    image_size: int,
    batch_size: int,
    keras_modules,
):
    ImageDataGenerator = keras_modules["ImageDataGenerator"]
    x_train, y_train = load_images_to_tuple(train_items, image_size=image_size, keras_modules=keras_modules)
    x_test, y_test = load_images_to_tuple(test_items, image_size=image_size, keras_modules=keras_modules)

    train_tuple = (x_train, y_train)
    test_tuple = (x_test, y_test)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    trdata = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_STATE,
    )
    tsdata = test_datagen.flow(
        x_test,
        y_test,
        batch_size=batch_size,
        shuffle=False,
    )
    return trdata, tsdata, train_tuple, test_tuple


def create_loading_pipeline(
    requested_mode: str,
    ram_threshold_gb: float,
    train_items: list[tuple[Path, str]],
    test_items: list[tuple[Path, str]],
    prepared_dir: Path,
    image_size: int,
    batch_size: int,
    keras_modules,
):
    selected_mode, total_ram_gb = choose_loading_mode(
        requested_mode=requested_mode,
        ram_threshold_gb=ram_threshold_gb,
    )
    estimated_tuple_gb = estimate_image_tuple_gb(
        train_items=train_items,
        test_items=test_items,
        image_size=image_size,
    )

    if selected_mode == "memory":
        trdata, tsdata, train_tuple, test_tuple = create_memory_generators(
            train_items=train_items,
            test_items=test_items,
            image_size=image_size,
            batch_size=batch_size,
            keras_modules=keras_modules,
        )
        metadata = {
            "selected_mode": "memory",
            "total_ram_gb": total_ram_gb,
            "estimated_tuple_gb": estimated_tuple_gb,
            "train_tuple_shape": train_tuple[0].shape,
            "test_tuple_shape": test_tuple[0].shape,
        }
        return trdata, tsdata, train_tuple, test_tuple, metadata

    trdata, tsdata = create_directory_generators(
        prepared_dir=prepared_dir,
        image_size=image_size,
        batch_size=batch_size,
        ImageDataGenerator=keras_modules["ImageDataGenerator"],
    )
    metadata = {
        "selected_mode": "directory",
        "total_ram_gb": total_ram_gb,
        "estimated_tuple_gb": estimated_tuple_gb,
    }
    return trdata, tsdata, None, None, metadata


def build_vgg_classifier(image_size: int, learning_rate: float, keras_modules):
    Sequential = keras_modules["Sequential"]
    Conv2D = keras_modules["Conv2D"]
    MaxPool2D = keras_modules["MaxPool2D"]
    Flatten = keras_modules["Flatten"]
    Dense = keras_modules["Dense"]
    Adam = keras_modules["Adam"]

    model = Sequential()
    model.add(Conv2D(input_shape=(image_size, image_size, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_with_compatible_fit(model, trdata, tsdata, epochs: int, callbacks):
    if hasattr(model, "fit_generator"):
        return model.fit_generator(
            trdata,
            epochs=epochs,
            validation_data=tsdata,
            callbacks=callbacks,
        )

    return model.fit(
        trdata,
        epochs=epochs,
        validation_data=tsdata,
        callbacks=callbacks,
    )


def reset_iterator_if_possible(iterator) -> None:
    if hasattr(iterator, "reset"):
        iterator.reset()


def extract_true_labels(iterator) -> np.ndarray:
    if hasattr(iterator, "classes"):
        return np.asarray(iterator.classes)

    if hasattr(iterator, "y"):
        labels = np.asarray(iterator.y)
        if labels.ndim > 1:
            return np.argmax(labels, axis=1)
        return labels

    raise ValueError("No pude extraer las etiquetas reales del iterador de test.")


def save_training_curves(history) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.history.get("accuracy", []), label="accuracy")
    ax.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    ax.plot(history.history.get("loss", []), label="loss")
    ax.plot(history.history.get("val_loss", []), label="val_loss")
    ax.set_title("Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)
    return CURVES_PATH


def evaluate_best_model(tsdata, keras_modules) -> dict:
    load_model = keras_modules["load_model"]
    best_model = load_model(BEST_MODEL_PATH)

    reset_iterator_if_possible(tsdata)
    loss, accuracy = best_model.evaluate(tsdata, verbose=0)

    reset_iterator_if_possible(tsdata)
    probabilities = best_model.predict(tsdata, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = extract_true_labels(tsdata)

    return {
        "test_loss": round(float(loss), 4),
        "test_accuracy": round(float(accuracy), 4),
        "class_indices": getattr(tsdata, "class_indices", {class_name: index for index, class_name in enumerate(CLASS_NAMES)}),
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
        "classification_report": classification_report(
            true_labels,
            predictions,
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_training_summary(history, dataset_counts: dict[str, int], evaluation: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_counts": dataset_counts,
        "history": {key: [round(float(value), 4) for value in values] for key, values in history.history.items()},
        "evaluation": evaluation,
        "artifacts": {
            "best_model": str(BEST_MODEL_PATH.relative_to(PROJECT_DIR)),
            "final_model": str(FINAL_MODEL_PATH.relative_to(PROJECT_DIR)),
            "cat_grid": str((PROCESSED_DIR / "cat_sample_grid.png").relative_to(PROJECT_DIR)),
            "dog_grid": str((PROCESSED_DIR / "dog_sample_grid.png").relative_to(PROJECT_DIR)),
            "curves": str(CURVES_PATH.relative_to(PROJECT_DIR)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        dataset_info = detect_raw_dataset(args.raw_dir)
    except FileNotFoundError as error:
        print(error)
        print("Cuando tengas el dataset listo, vuelve a ejecutar este script.")
        return

    train_items, test_items = build_train_test_items(
        dataset_info=dataset_info,
        test_size=args.test_size,
        sample_limit=args.sample_limit,
    )

    dataset_counts = prepare_directory_dataset(
        prepared_dir=args.prepared_dir,
        train_items=train_items,
        test_items=test_items,
        copy_files=args.copy_files,
        rebuild_prepared=args.rebuild_prepared,
    )

    dog_grid = save_sample_grid(args.prepared_dir, "dog", args.image_size)
    cat_grid = save_sample_grid(args.prepared_dir, "cat", args.image_size)
    print(f"Layout detectado: {dataset_info['layout']}")
    print(f"Origen del dataset: {dataset_info['source']}")
    print(f"Dataset preparado en: {args.prepared_dir}")
    print(f"Conteos: {dataset_counts}")
    print(f"Visualizaciones guardadas en: {dog_grid} y {cat_grid}")

    if args.prepare_only:
        print("Preparacion terminada. Ejecuta de nuevo sin --prepare-only para entrenar.")
        return

    keras_modules = import_tensorflow()
    trdata, tsdata, train_tuple, test_tuple, loading_metadata = create_loading_pipeline(
        requested_mode=args.loading_mode,
        ram_threshold_gb=args.ram_threshold_gb,
        train_items=train_items,
        test_items=test_items,
        prepared_dir=args.prepared_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        keras_modules=keras_modules,
    )

    if loading_metadata["selected_mode"] == "memory":
        print(
            "Modo de carga: memory "
            f"(RAM detectada: {loading_metadata['total_ram_gb']:.2f} GB, "
            f"estimacion de fotos en RAM: {loading_metadata['estimated_tuple_gb']:.2f} GB, "
            f"train_tuple={train_tuple[0].shape}/{train_tuple[1].shape}, "
            f"test_tuple={test_tuple[0].shape}/{test_tuple[1].shape})"
        )
    else:
        print(
            "Modo de carga: directory "
            f"(RAM detectada: {loading_metadata['total_ram_gb']:.2f} GB, "
            f"estimacion de fotos en RAM: {loading_metadata['estimated_tuple_gb']:.2f} GB, "
            "usando ImageDataGenerator.flow_from_directory)"
        )

    model = build_vgg_classifier(
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        keras_modules=keras_modules,
    )
    model.summary()

    callbacks = [
        keras_modules["ModelCheckpoint"](
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras_modules["EarlyStopping"](
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = train_with_compatible_fit(
        model=model,
        trdata=trdata,
        tsdata=tsdata,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(FINAL_MODEL_PATH)
    curves_path = save_training_curves(history)

    evaluation = evaluate_best_model(tsdata, keras_modules)
    save_training_summary(history, dataset_counts, evaluation)

    print(f"Modelo final guardado en: {FINAL_MODEL_PATH}")
    print(f"Mejor modelo guardado en: {BEST_MODEL_PATH}")
    print(f"Curvas guardadas en: {curves_path}")
    print(f"Resumen guardado en: {SUMMARY_PATH}")
    print(f"Accuracy en test: {evaluation['test_accuracy']}")


if __name__ == "__main__":
    main()
