import os
from FaceDetection.darkflow.net.build import TFNet
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.stderr = open(os.devnull, 'w')

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# drawing bounding boxes
def boxing(original_img, predictions):
    new_image = np.copy(original_img)
    for result in predictions:
        if result['label'] == 'face':
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            label = result['label']  # + " " + str(round(confidence, 3))
            new_image = cv2.rectangle(new_image, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
            new_image = cv2.putText(new_image, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                    (0, 230, 0), 1, cv2.LINE_AA)
    return new_image


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def read_face_locations(face_locations_file):
    faces = []
    if not os.path.isfile(face_locations_file):
        sys.exit(1)
    with open(face_locations_file) as file:
        lines = [line.rstrip('\n') for line in file]
        for l in lines:
            split_res = l.split()
            faces.append(((int(split_res[0]), int(split_res[1])), (int(split_res[2]), int(split_res[3]))))
    return faces


def process_results(predictions):
    do_not_show_results = []
    for result in predictions:
        if result['label'] == 'face':
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']
            for other_result in predictions:
                if other_result != result:
                    if other_result['topleft']['x'] <= top_x and other_result['topleft']['y'] <= top_y and \
                            other_result['bottomright']['x'] >= btm_x and other_result['bottomright']['y'] >= btm_y:
                        if other_result['confidence'] <= result['confidence']:
                            do_not_show_results.append(other_result)
                        else:
                            do_not_show_results.append(result)
                        break

    for do_not_show_result in do_not_show_results:
        predictions.remove(do_not_show_result)
    return predictions


def point_in_image(x, y, top_x, btm_x, top_y, btm_y):
    if top_x < x < btm_x and top_y < y < btm_y:
        return True
    else:
        return False


def evaluate(results, image_name, face_locations_dir):
    faces = read_face_locations(face_locations_dir + image_name + ".txt")
    number_of_faces_on_image = len(faces)
    true_detections = 0
    false_detections = 0
    i = 0
    for result in results:
        if result['label'] == 'face':
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            size = len(faces)

            for f in faces:
                union_pixel = 0
                intersection_pixel = 0
                min_x = min(top_x, btm_x, f[0][0], f[1][0])
                max_x = max(top_x, btm_x, f[0][0], f[1][0])
                min_y = min(top_y, btm_y, f[0][1], f[1][1])
                max_y = max(top_y, btm_y, f[0][1], f[1][1])

                for x in range(min_x, max_x, 1):
                    for y in range(min_y, max_y, 1):
                        # calculate number of pixels in union and intersection of bounding boxes
                        if point_in_image(x, y, top_x, btm_x, top_y, btm_y) or point_in_image(x, y, f[0][0], f[1][0],
                                                                                              f[0][1], f[1][1]):
                            union_pixel += 1
                        if point_in_image(x, y, top_x, btm_x, top_y, btm_y) and point_in_image(x, y, f[0][0],
                                                                                               f[1][0], f[0][1],
                                                                                               f[1][1]):
                            intersection_pixel += 1

                # true detection
                if union_pixel > 0 and intersection_pixel > 0:
                    if intersection_pixel / union_pixel >= 0.5:
                        print(
                            "Lice čije koordinate gornjeg lijevog ruba su (" + str(top_x) + ", " + str(top_y) + ") i " +
                            "koordinate donjeg desnog ruba su (" + str(btm_x) + ", " + str(
                                btm_y) + ") je uspješno pronađeno.")
                        faces.remove(f)
                        true_detections += 1
                        break

            # false detection
            new_size = len(faces)
            if new_size >= size:
                print("Lice čije koordinate gornjeg lijevog ruba su (" + str(top_x) + ", " + str(top_y) + ") i " +
                      "koordinate donjeg desnog ruba su (" + str(btm_x) + ", " + str(
                    btm_y) + ") je pogrešna ili nedovoljno precizna detekcija.")
                false_detections += 1

    # results
    print("Od ukupnog broja lica: " + str(number_of_faces_on_image) + ", pronađeno je: " + str(
        true_detections) + " lica.")
    print("Broj pogrešnih detekcija je: " + str(false_detections))
    print("Broj neuspješno detektiranih lica je: " + str(len(faces)))
    print("-" * 50)
    return true_detections, false_detections, len(faces)


if __name__ == "__main__":
    # load trained model
    path = os.path.abspath(".")
    options = {"model": path + "/cfg/tiny-yolo-voc-new.cfg",
               "load": 25550,
               "json": True,
               "threshold": 0.5,
               "backup": path + "/ckpt",
               "gpu": 1.0}
    tfnet = TFNet(options)
    tfnet.load_from_ckpt()

    # get images
    images_dir = path + "/test_images/"
    images = [f for f in os.listdir(images_dir)]
    images.sort()
    test_images_face_locations_dir = path + "/test_images_face_locations/"
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    for image in images:
        image_path = images_dir + image
        imgcv = cv2.imread(image_path)

        # get detection results
        results = process_results(tfnet.return_predict(imgcv))

        #show image
        cv2.imshow(image, boxing(imgcv, results))
        print(image)

        # evaluate results
        true_positives_img, false_positives_img, false_negatives_img = evaluate(
            results, os.path.splitext(image)[0], test_images_face_locations_dir)
        true_positives += true_positives_img
        false_positives += false_positives_img
        false_negatives += false_negatives_img
        cv2.waitKey(0)
        cv2.destroyWindow(image)

    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1 = 2*precision*recall/(precision+recall)

    print("Preciznost je: " + str(precision))
    print("Odziv je; " + str(recall))
    print("F1 mjera je: " + str(f1))
