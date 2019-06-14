from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import cv2 as cv
import shutil

sys.stderr = open(os.devnull, 'w')


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def find_faces(image):
    all_faces = []
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        all_faces.append((x, y, w, h))
    return all_faces


def crop(image, x, y, w, h):
    return image.copy()[y:y + h, x:x + w]


def check_if_face_is_known(path):
    t = read_tensor_from_image_file(path, input_height=input_height, input_width=input_width,
                                    input_mean=input_mean, input_std=input_std)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph_face_known.get_operation_by_name(input_name)
    output_operation = graph_face_known.get_operation_by_name(output_name)

    with tf.Session(graph=graph_face_known) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file_face_known)
    flag = True
    for i in top_k:
        if flag:
            flag = not flag

        if results[i] > 0.5 and labels[i] == "known":
            return True
    return False


def find_face_class(path):
    t = read_tensor_from_image_file(path, input_height=input_height, input_width=input_width,
                                    input_mean=input_mean, input_std=input_std)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph_face_classification.get_operation_by_name(input_name)
    output_operation = graph_face_classification.get_operation_by_name(output_name)

    with tf.Session(graph=graph_face_classification) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file_face_classification)
    flag = True
    for i in top_k:
        if results[i] > 0. and flag:
            flag = False
            if results[i] > 0.85:
                return labels[i]
    return None


def initialize_dictionary():
    return {'donald_trump': 0, 'barack_obama': 0, 'borut_pahor': 0, 'theresa_may': 0, 'angela_merkel': 0,
            'xi_jinping': 0, 'kim_jong_un': 0, 'justin_trudeau': 0, 'emmanuel_macron': 0,
            'vladimir_putin': 0, 'face': 0}


def boxing(x, y, w, h, label):
    newImage = cv.rectangle(new_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    newImage = cv.putText(newImage, label, (x, y + h + 20), cv.FONT_HERSHEY_COMPLEX, 0.5,
                          (0, 230, 0), 1, cv.LINE_AA)
    return newImage


def print_results(total, true_positives, false_positives, false_negatives, num_of_images, num_of_faces):
    print("-*" * 50)
    print("TRUE_POSITIVES")
    for key in total.keys():
        print(key + ": " + str(true_positives.get(key)))

    print("-*" * 50)
    print("FALSE_POSITIVES")
    for key in total.keys():
        print(key + ": " + str(false_positives.get(key)))

    print("-*" * 50)
    print("FALSE_NEGATIVES")
    for key in total.keys():
        print(key + ": " + str(false_negatives.get(key)))

    print("-*" * 50)
    for key in total.keys():
        print(key)
        if true_positives[key] > 0 or (false_negatives[key] > 0 and false_positives[key]):
            precision = true_positives[key] / (true_positives[key] + false_positives[key])
            recall = true_positives[key] / (true_positives[key] + false_negatives[key])
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        print("PRECISION = " + str(round(precision, 3)))
        print("RECALL = " + str(round(recall, 3)))
        print("F1 = " + str(round(f1, 3)))
        print("-*" * 50)

    sum_true_positives = 0
    sum_false_positives = 0
    sum_false_negatives = 0
    for key in total.keys():
        sum_true_positives += true_positives[key]
        sum_false_negatives += false_negatives[key]
        sum_false_positives += false_positives[key]

    sum_precision = sum_true_positives / (sum_true_positives + sum_false_positives)
    sum_recall = sum_true_positives / (sum_true_positives + sum_false_negatives)
    sum_f1 = (2 * sum_precision * sum_recall) / (sum_precision + sum_recall)

    print("AVERAGE_PRECISION = " + str(round(sum_precision, 3)))
    print("AVERAGE_RECALL = " + str(round(sum_recall, 3)))
    print("AVERAGE_F1 = " + str(round(sum_f1, 3)))
    print("-*" * 50)

    print("Number of images: ", num_of_images)
    print("Number of faces:", num_of_faces)


if __name__ == "__main__":
    model_file_face_classification = "./FaceRecognition/face_class_graph.pb"
    label_file_face_classification = "./FaceRecognition/face_class_labels.txt"
    model_file_face_known = "./FaceRecognition/known_face_graph.pb"
    label_file_face_known = "./FaceRecognition/known_face_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Mul"
    output_layer = "final_result"
    face_cascade = cv.CascadeClassifier('./FaceRecognition/haarcascade_frontalface_default.xml')
    graph_face_classification = load_graph(model_file_face_classification)
    graph_face_known = load_graph(model_file_face_known)

    tests_path = "./FaceRecognition/test_dataset/Tests/"
    files = os.listdir(tests_path)

    tmp_path = "./FaceRecognition/tmp/"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    files.sort()
    num_of_images = 0
    num_of_faces = 0
    true_positives = initialize_dictionary()
    false_positives = initialize_dictionary()
    false_negatives = initialize_dictionary()
    total = initialize_dictionary()
    print("Molimo pričekajte, testiranje može potrajati nekoliko minuta.")
    for file_name in files:
        # read image
        print("Trenutno u obradi:", file_name)
        num_of_images += 1
        image_path = tests_path + file_name
        image = cv.imread(image_path)
        all_faces = find_faces(image)

        # read annotations
        result_file = open("./FaceRecognition/test_dataset/Test_Correct_Results/" + file_name.strip(".jpg") + ".txt", "r")
        lines = result_file.readlines()
        num_of_faces += len(lines)

        new_image = image.copy()
        for face in all_faces:
            # create temporary face image
            x, y, w, h = face
            crop_image = crop(image, x, y, w, h)
            cv.imwrite(os.path.join(tmp_path, "tmp_image.jpg"), crop_image)

            # create bounding boxes
            class_found = False
            label = None
            if check_if_face_is_known(tmp_path + "tmp_image.jpg"):
                label = find_face_class(tmp_path + "tmp_image.jpg")
                if label is not None:
                    class_found = True
                    newImage = boxing(x, y, w, h, label)
            if not class_found:
                newImage = boxing(x, y, w, h, "face")

            # determine is image true positive, false positive or false negative
            for line in lines:
                splits = line.strip().split()
                if int(splits[0]) == x and int(splits[1]) == y and int(splits[2]) == w and int(splits[3]) == h:
                    true_label = splits[4]
                    if label is None:
                        if true_label == "face":
                            true_positives['face'] += 1
                        else:
                            false_negatives[true_label] += 1
                            false_positives["face"] += 1

                    else:
                        if true_label == label:
                            true_positives[label] += 1
                        else:
                            false_positives[label] += 1
                            false_negatives[true_label] += 1
                    break

        # showing images
        # cv.imshow(file_name, new_image)
        # cv.waitKey(0)
        # cv.destroyWindow(file_name)

    print_results(total, true_positives, false_positives, false_negatives, num_of_images, num_of_faces)

    shutil.rmtree(tmp_path)
