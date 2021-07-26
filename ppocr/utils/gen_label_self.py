#coding=utf-8
import os
import json
import random
import cv2
import xmltodict


def resize_image():
    base_dir = "/data/OCR/barcode/unlabel/imgs"
    dst_dir = base_dir + "_rz"
    os.makedirs(dst_dir, exist_ok=True)
    for img_file in os.listdir(base_dir):
        print(img_file)
        img_path = os.path.join(base_dir, img_file)
        image = cv2.imread(img_path)
        h, w = image.shape[0], image.shape[1]
        # 长边缩放为min_side
        scale = 4.0
        new_w, new_h = int(w / scale), int(h / scale)
        resize_img = cv2.resize(image, (new_w, new_h))

        img_path_dst = os.path.join(dst_dir, img_file)
        cv2.imwrite(img_path_dst, resize_img)


def load_json(xml_path):
    with open(xml_path, "r") as fr:
        xml_str = fr.read()
        json_data = xmltodict.parse(xml_str)
        img_bbox = json_data["Result"]["Pages"]["Page"]["Barcodes"]["Barcode"]["Polygon"]["Points"]["Point"]
        img_value = json_data["Result"]["Pages"]["Page"]["Barcodes"]["Barcode"]["Value"]
        return img_bbox, img_value


def gen_det_label(imgs_dir, labels_dir, out_label):
    with open(out_label, 'w') as out_file:
        for img_file in os.listdir(imgs_dir):
            if "EAN13" in img_file:
                label = []
                img_path = os.path.join(imgs_dir, img_file)
                img_bbox, img_value = load_json(os.path.join(labels_dir, img_file.replace(".jpg", ".xml")))

                points = []
                for cur_pt in img_bbox:
                    cur_x = cur_pt["@X"]
                    cur_y = cur_pt["@Y"]
                    points.append([int(float(cur_x)), int(float(cur_y))])
                result = {"transcription": img_value, "points": points}
                label.append(result)
            else:
                continue
            out_file.write(img_path + '\t' + json.dumps(label, ensure_ascii=False) + '\n')


def preprocess():
    base_dir = "/data/OCR/barcode/data"
    sub_list = ["iphone4", "Original", "ZVZ512"]
    for cur_sub in sub_list:
        cur_dir_labelfile = os.path.join(base_dir, cur_sub+"_Label.txt")
        img_use_list = []
        cur_dict = {}
        with open(cur_dir_labelfile, 'r') as fr:
            for c_line in fr:
                img_path, label = c_line.strip().split("\t")
                img_name = img_path.split("/")[-1]
                img_use_list.append(img_name)
                # result = json.loads(label)
                new_img_name = cur_sub + "_" + img_name
                cur_dict[new_img_name] = label
        cur_dir = os.path.join(base_dir, cur_sub)
        for img_file in os.listdir(cur_dir):
            img_path = os.path.join(cur_dir, img_file)
            if img_file not in img_use_list:
                print(img_file, " not in Label.txt")
                os.system("rm %s" % img_path)
            else:
                img_path_new = os.path.join(cur_dir, cur_sub + "_" + img_file)
                os.system("mv %s %s" % (img_path, img_path_new))

        cur_dir_labelfile_new = os.path.join(base_dir, cur_sub+"_Label_new.txt")
        with open(cur_dir_labelfile_new, 'w') as fw:
            for key, val in cur_dict.items():
                fw.write(key + '\t' + val + '\n')


def write_file(file_path, src_data):
    with open(file_path, "w") as fw:
        for cur_l in src_data:
            fw.write(cur_l)


def chect_points(points_src):
    points = sorted(points_src, key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box


def check_pt():
    src_label_file = "Original_Label.txt"
    dst_label_file = src_label_file.replace(".txt", "_new.txt")
    lines_new_list = []
    index =0
    bad_cnt = 0
    with open(src_label_file, 'r') as fr:
        for c_line in fr:
            index += 1
            # print(index)
            img_path, label_str = c_line.strip().split("\t")
            label_dicts = json.loads(label_str)
            labels_new = []
            for cur_dict in label_dicts:
                pts_src = cur_dict["points"]
                pts_new = chect_points(pts_src)
                # t1 = "".join(pts_src)
                # t2 = "".join(pts_new)
                # tt = set(pts_src).difference(set(pts_new))
                if not(pts_src[0]==pts_new[0] and pts_src[1]==pts_new[1] and pts_src[2]==pts_new[2] and pts_src[3]==pts_new[3]):
                    print(pts_src, "<---->", pts_new)
                    bad_cnt += 1
                if cur_dict.get('difficult') != None:
                    result = {"transcription": cur_dict["transcription"], "points": pts_new, "difficult":cur_dict["difficult"]}
                else:
                    result = {"transcription": cur_dict["transcription"], "points": pts_new}
                labels_new.append(result)
            lines_new = img_path + '\t' + json.dumps(labels_new, ensure_ascii=False) + '\n'
            lines_new_list.append(lines_new)
    print(bad_cnt)
    with open(dst_label_file, 'w') as fw:
        for c_line in lines_new_list:
            fw.write(c_line)


if __name__ == "__main__":
    # check_pt()
    # exit()

    # base_dir = "/data/OCR/barcode/labeled/ZVZ_real512"   # ZVZ_real512  Artelab
    # imgs_dir = os.path.join(base_dir, "JPEGImages")
    # labels_dir = os.path.join(base_dir, "Annotations")
    # output_label = os.path.join(base_dir, "labels.txt")
    # gen_det_label(imgs_dir, labels_dir, output_label)

    src_label_file = "Original_Label.txt"
    train_label_file = src_label_file.replace(".txt", "_train.txt")
    test_label_file = src_label_file.replace(".txt", "_test.txt")
    abs_dir = "/data/OCR/barcode/data/Original"
    list_all = []
    with open(src_label_file, 'r') as fr:
        for c_line in fr:
            img_path, label = c_line.strip().split("\t")
            img_path_new = os.path.join(abs_dir, img_path)
            list_all.append(img_path_new + '\t' + label + '\n')

    random.shuffle(list_all)
    thred_num = int(len(list_all)*0.9)
    train_list = list_all[:thred_num]
    test_list = list_all[thred_num:]

    write_file(train_label_file, train_list)
    write_file(test_label_file, test_list)
