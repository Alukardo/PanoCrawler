import csv
import itertools
import os
import shutil
from pathlib import Path
from typing import TextIO

source_pre_path = "../images/pano/"

inputs_pre_path = "../temp/train_A/"
output_pre_path = "../temp/train_B/"
lables_pre_path = "../temp/train_cond/"
featur_pre_path = "../temp/train_feat/"


def clean(dic: str) -> None:
    for file_name in os.listdir(dic):
        file_path = os.path.join(dic, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def process_data(filename: str) -> None:
    data = []

    with open(filename, encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(row)

    results = list(itertools.combinations(data, 2))

    i = 0

    for re in results:
        d_lat = float(re[0]["lat"]) - float(re[1]["lat"])
        d_lon = float(re[0]["lon"]) - float(re[1]["lon"])

        distance = d_lat * d_lat + d_lon * d_lon

        if distance < 0.00000001:
            source_path = source_pre_path + re[0]["pano_id"] + ".png"
            target_path = source_pre_path + re[1]["pano_id"] + ".png"

            inputs_path = inputs_pre_path + "int" + f"{i:05d}" + ".png"
            output_path = output_pre_path + "out" + f"{i:05d}" + ".png"
            lables_path = lables_pre_path + "ins" + f"{i:05d}" + ".txt"

            shutil.copyfile(source_path, inputs_path)
            shutil.copyfile(target_path, output_path)

            with open(lables_path, "w") as lables_file:
                lables_file.write(f"{d_lat:.16f}\n")
                lables_file.write(f"{d_lon:.16f}\n")

            i = i + 1

            inputs_path = inputs_pre_path + "int" + f"{i:05d}" + ".png"
            output_path = output_pre_path + "out" + f"{i:05d}" + ".png"
            lables_path = lables_pre_path + "ins" + f"{i:05d}" + ".txt"

            shutil.copyfile(source_path, output_path)
            shutil.copyfile(target_path, inputs_path)

            with open(lables_path, "w") as lables_file:
                lables_file.write(f"{-d_lat:.16f}\n")
                lables_file.write(f"{-d_lon:.16f}\n")

            i = i + 1
    print("Pairs of Data: " + str(i))


if __name__ == "__main__":
    clean(inputs_pre_path)
    clean(output_pre_path)
    clean(lables_pre_path)
    clean(featur_pre_path)

    filename = "../images/pano/info.csv"
    process_data(filename)
