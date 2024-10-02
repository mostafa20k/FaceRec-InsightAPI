import os
import requests


SERVER_URL = 'http://127.0.0.1:8000'


def read_file_and_pair_images_cacp(file_path, IMG_DIR_PATH):
    """Pair CALFW/CPLFW images based on the txt file."""
    labels = []
    pairs = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for index in range(0, len(lines), 2):
            parts1 = lines[index].strip().split(' ')
            parts2 = lines[index + 1].strip().split(' ')
            img1_path = os.path.join(IMG_DIR_PATH, parts1[0])
            img2_path = os.path.join(IMG_DIR_PATH, parts2[0])
            pairs.append((img1_path, img2_path))
            label = 1 if parts1[1] != '0' else 0
            labels.append(label)

    return pairs, labels


def read_file_and_pair_images_lfw(pairs_filename, LFW_DIR_PATH='lfw'):
    """Pair LFW images based on the txt file."""
    pairs = []
    labels = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            if len(pair) == 3:
                person, img_num1, img_num2 = pair
                img_path1 = os.path.join(LFW_DIR_PATH, person,
                                         f"{person}_{img_num1.zfill(4)}.jpg")
                img_path2 = os.path.join(LFW_DIR_PATH, person,
                                         f"{person}_{img_num2.zfill(4)}.jpg")
                label = 1
            else:
                person1, img_num1, person2, img_num2 = pair
                img_path1 = os.path.join(LFW_DIR_PATH, person1,
                                         f"{person1}_{img_num1.zfill(4)}.jpg")
                img_path2 = os.path.join(LFW_DIR_PATH, person2,
                                         f"{person2}_{img_num2.zfill(4)}.jpg")
                label = 0
            labels.append(label)
            pairs.append((img_path1, img_path2))
    return pairs, labels


def calculate_metrics(labels, predicted_labels):
    """Evaluate performance of the model"""
    tp = sum((1 for actual, predicted in zip(labels, predicted_labels)
              if actual == 1 and predicted == 1))
    tn = sum((1 for actual, predicted in zip(labels, predicted_labels)
              if actual == 0 and predicted == 0))
    fp = sum((1 for actual, predicted in zip(labels, predicted_labels)
              if actual == 0 and predicted == 1))
    fn = sum((1 for actual, predicted in zip(labels, predicted_labels)
              if actual == 1 and predicted == 0))

    accuracy = (tp + tn) / len(labels)
    fmr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnmr = fn / (fn + tp) if (fn + tp) != 0 else 0

    return accuracy, fmr, fnmr


def initialize_model(model_name: str, dataset: str):
    """Initialize the model on the server."""
    url = f"{SERVER_URL}/initialize_model"
    payload = {"modelName": model_name, "dataset": dataset}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(response.json()["message"])
    else:
        print(f"Error: {response.json()['detail']}")
    return dataset


def predict(image_path1: str, image_path2: str):
    """Send images to the server for prediction."""
    url = f"{SERVER_URL}/predict"
    files = {
        "file1": open(image_path1, "rb"),
        "file2": open(image_path2, "rb")
    }
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()['predicted label']
    else:
        print(f"Error: {response.json()['detail']}")
        return None


if __name__ == "__main__":
    dataset = initialize_model("buffalo_l", "cplfw")
    if dataset.lower() == "lfw":
        pairs, labels = read_file_and_pair_images_lfw('pairs.txt')
    elif dataset.lower() == "calfw":
        pairs, labels = read_file_and_pair_images_cacp('calfw/pairs_CALFW.txt', IMG_DIR_PATH='calfw/aligned images')
    else:
        pairs, labels = read_file_and_pair_images_cacp('cplfw/pairs_CPLFW.txt', IMG_DIR_PATH='cplfw/aligned images')
    predicted_labels = []
    for pair in pairs:
        predicted_label = int(predict(pair[0], pair[1]))
        predicted_labels.append(predicted_label)

    accuracy, fmr, fnmr = calculate_metrics(labels, predicted_labels)
    print(f"Accuracy: {accuracy}, FMR: {fmr} and FNMR: {fnmr}")
