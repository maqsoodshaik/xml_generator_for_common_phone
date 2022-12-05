import asyncio
import os
import csv
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import textgrid
from cp_feature_extractor import codes_low_high,get_phn_mapping,phn_ind,model_output,AutoFeatureExtractor,Wav2Vec2ForPreTraining

def get_discrete_units_from_model(subdir,file,feature_extractor,model,low,high,phns):
    # obtaining discrete units
        codebook = 1
        mdl_out_cd_1 = model_output(
            audio_path=subdir + "/wav/" + file.split(".")[0] + ".wav",
            codebook=codebook,
            feature_extractor=feature_extractor,
            model=model,
        )
        codebook = 2
        mdl_out_cd_2 = model_output(
            audio_path=subdir + "/wav/" + file.split(".")[0] + ".wav",
            codebook=codebook,
            feature_extractor=feature_extractor,
            model=model,
        )
        low_lst, high_lst = codes_low_high(subdir + "/wav/" + file.split(".")[0] + ".wav", mdl_out_cd_1)
        phn_discrete_units_1 = phn_ind(low, high, phns, low_lst, high_lst, mdl_out_cd_1)
        low_lst, high_lst = codes_low_high(subdir + "/wav/" + file.split(".")[0] + ".wav", mdl_out_cd_2)
        phn_discrete_units_2 = phn_ind(low, high, phns, low_lst, high_lst, mdl_out_cd_2)
        #combining two list of lists into one list of lists by adding 320 value to each element of the second list
        # phn_discrete_units = [x + np.add(y, 320).tolist() for x, y in zip(phn_discrete_units_1, phn_discrete_units_2)]
        return phn_discrete_units_1,phn_discrete_units_2

def get_grid_mapping(subdir,file,textgrid_path, meta_sentence):
    txt_grid = textgrid.TextGrid.fromFile(textgrid_path)
    meta_sentence_0 = ET.SubElement(meta_sentence, "interval_mapping")
    meta_sentence_0.set("start", str(txt_grid.minTime))
    meta_sentence_0.set("end", str(txt_grid.maxTime))
    text_grid_path = subdir + "/grids/" + file.split(".")[0] + ".TextGrid"
    low, high, phns = get_phn_mapping(text_grid_path)
    type_names = ["words", "words_phn", "phn"]
    for i, items in enumerate(txt_grid):
        meta_sentence_1 = ET.SubElement(meta_sentence_0, str(items.name))
        meta_sentence_1.set("start", str(items.minTime))
        meta_sentence_1.set("end", str(items.maxTime))
        #model and feature extraction
        phn_discrete_units_wav2vec2_codebook_1,phn_discrete_units_wav2vec2_codebook_2 = get_discrete_units_from_model(subdir,file,feature_extractor_wav2vec2,model_wav2vec2,low,high,phns)
        phn_discrete_units_xlsr_codebook_1,phn_discrete_units_xlsr_codebook_2 = get_discrete_units_from_model(subdir,file,feature_extractor_xlsr,model_xlsr,low,high,phns)
        #traversing through the phonemes in tetxgrid file
        for index,item in enumerate(items):
            meta_sentence_2 = ET.SubElement(meta_sentence_1, type_names[i])
            meta_sentence_2.set("type", str(item.mark))
            # print(item.mark)
            # ET.tostring(root)
            meta_sentence_2.set("start", str(item.minTime))
            meta_sentence_2.set("end", str(item.maxTime))
            if i == 2:
                meta_sentence_2.set("wav2vec_discrete_units_codebook_1", str(phn_discrete_units_wav2vec2_codebook_1[index]))
                meta_sentence_2.set("wav2vec_discrete_units_codebook_2", str(phn_discrete_units_wav2vec2_codebook_2[index]))
                meta_sentence_2.set("xlsr_discrete_units_codebook_1", str(phn_discrete_units_xlsr_codebook_1[index]))
                meta_sentence_2.set("xlsr_discrete_units_codebook_2", str(phn_discrete_units_xlsr_codebook_2[index]))
            

    return meta_sentence


# def xml_writer(save_path, dataset_path):
#     for subdir, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if "meta.csv" in file:
#                 with open(f"{subdir}/meta.csv", "rt", encoding="UTF8") as f:
#                     with open(f"{subdir}/train.csv", "rt", encoding="UTF8") as train:
#                         with open(f"{subdir}/test.csv", "rt", encoding="UTF8") as test:
#                             with open(
#                                 f"{subdir}/dev.csv", "rt", encoding="UTF8"
#                             ) as dev:
#                                 meta_data = pd.read_csv(f, sep=",").to_numpy()
#                                 train_data = pd.read_csv(train, sep=",").to_numpy()
#                                 test_data = pd.read_csv(test, sep=",").to_numpy()
#                                 dev_data = pd.read_csv(dev, sep=",").to_numpy()
#                                 for row in meta_data:
#                                     if row[0] in train_data[:, 1]:
#                                         output_path = train_data[
#                                             np.where(train_data[:, 1] == row[0])[0][0],
#                                             0,
#                                         ]
#                                     elif row[0] in test_data[:, 1]:
#                                         output_path = test_data[
#                                             test_data[:, 1].index(row[0]), 0
#                                         ]
#                                     else:
#                                         output_path = dev_data[
#                                             dev_data[:, 1].index(row[0]), 0
#                                         ]

#                                     root = ET.Element("Audio_file")
#                                     root.set("path", output_path)
#                                     meta_speaker = ET.SubElement(root, "meta_speaker")
#                                     gender = ET.SubElement(meta_speaker, "gender")
#                                     gender.text = row[1]
#                                     age = ET.SubElement(meta_speaker, "age")
#                                     age.text = row[2]
#                                     locale = ET.SubElement(meta_speaker, "locale")
#                                     locale.text = row[3]
#                                     accent = ET.SubElement(meta_speaker, "accent")
#                                     accent.text = row[4]
#                                     split = ET.SubElement(meta_speaker, "split")
#                                     split.text = row[5]
#                                     meta_sentence = ET.SubElement(root, "meta_sentence")
#                                     text_grid_path = f"{subdir}/grids/{output_path.split('.')[0]}.TextGrid"
#                                     meta_sentence = get_grid_mapping(
#                                         subdir,output_path,text_grid_path, meta_sentence
#                                     )
#                                     output_path = f"{save_path}/{subdir.split('/')[-1]}/{output_path.split('.')[0]}.xml"
#                                     if not os.path.exists(
#                                         "/".join(output_path.split("/")[:-1])
#                                     ):
#                                         os.makedirs(
#                                             "/".join(output_path.split("/")[:-1])
#                                         )
#                                     with open(output_path, "wb") as x:
#                                         xmldata = ET.tostring(root, "utf-8")
#                                         x.write(xmldata)

#     print("end")
async def async_meta_data(row, train_data, test_data, dev_data,subdir,save_path):
    if row[0] in train_data[:, 1]:
        output_path = train_data[
            np.where(train_data[:, 1] == row[0])[0][0],
            0,
        ]
    elif row[0] in test_data[:, 1]:
        output_path = test_data[
            test_data[:, 1].index(row[0]), 0
        ]
    else:
        output_path = dev_data[
            dev_data[:, 1].index(row[0]), 0
        ]

    root = ET.Element("Audio_file")
    root.set("path", output_path)
    meta_speaker = ET.SubElement(root, "meta_speaker")
    gender = ET.SubElement(meta_speaker, "gender")
    gender.text = row[1]
    age = ET.SubElement(meta_speaker, "age")
    age.text = row[2]
    locale = ET.SubElement(meta_speaker, "locale")
    locale.text = row[3]
    accent = ET.SubElement(meta_speaker, "accent")
    accent.text = row[4]
    split = ET.SubElement(meta_speaker, "split")
    split.text = row[5]
    meta_sentence = ET.SubElement(root, "meta_sentence")
    text_grid_path = f"{subdir}/grids/{output_path.split('.')[0]}.TextGrid"
    meta_sentence = get_grid_mapping(
        subdir,output_path,text_grid_path, meta_sentence
    )
    output_path = f"{save_path}/{subdir.split('/')[-1]}/{output_path.split('.')[0]}.xml"
    if not os.path.exists(
        "/".join(output_path.split("/")[:-1])
    ):
        os.makedirs(
            "/".join(output_path.split("/")[:-1])
        )
    with open(output_path, "wb") as x:
        xmldata = ET.tostring(root, "utf-8")
        await x.write(xmldata)

def xml_writer(save_path, dataset_path):
    with open(f"{dataset_path}/meta.csv", "rt", encoding="UTF8") as f:
        with open(f"{dataset_path}/train.csv", "rt", encoding="UTF8") as train:
            with open(f"{dataset_path}/test.csv", "rt", encoding="UTF8") as test:
                with open(
                    f"{dataset_path}/dev.csv", "rt", encoding="UTF8"
                ) as dev:
                    meta_data = pd.read_csv(f, sep=",").to_numpy()
                    train_data = pd.read_csv(train, sep=",").to_numpy()
                    test_data = pd.read_csv(test, sep=",").to_numpy()
                    dev_data = pd.read_csv(dev, sep=",").to_numpy()
                    #calling async function
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(asyncio.gather(*[async_meta_data(row, train_data, test_data, dev_data,dataset_path,save_path) for row in meta_data]))
                    loop.close()
                                

if __name__ == "__main__":
    dataset_path = "/Users/mohammedmaqsoodshaik/Desktop/hiwi/Common_phone_analysis/CP/en"
    save_path = f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/cp_to_xml/cp_xml"
    feature_extractor_wav2vec2 = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model_wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
    feature_extractor_xlsr = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model_xlsr = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    xml_writer(save_path, dataset_path)
    print("end")
