import os
import re
import csv
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom

from shutil import copy
inDirName = r"C:\Users\Yang\Desktop\zk"
outDirName = r"C:\Users\Yang\Desktop\xml"
i=0
with open('C:/Users/Yang/Desktop/ziku_new.csv','w',newline='') as csvFile:
    writer = csv.writer(csvFile)
    # 先写columns_name
    writer.writerow(["id","zi","bihua"])  #,"pingyin","bushou","jiegou"])
    for character in os.listdir(inDirName):
        # print(character)
        character_path = os.path.join(inDirName, character)
        # print(character_path)
        items = os.listdir(character_path)
        # print(items)
        # if len(items)<1:
        #     print(character_path)

        if 'png' in items:
            if 'config.xml' in items:
                for item in items:
                    # xml_path = os.path.join(character_path, item)
                    # if os.path.splitext(xml_path)[1] == ".xml":
                    if item == 'config.xml':
                        xml_path = os.path.join(character_path, item)
                        if i > 4:
                            print(xml_path)
                            f = open(xml_path, "r", encoding="GB18030")
                            r = f.read()
                            text = str(r.encode('UTF-8'), encoding="UTF-8")

                            # print(text)
                            # <pingyin>(\S)</pingyin> <bushou>(\S)</bushou>  <jiegou>(\S)</jiegou>')   <bihua>(\S)</bihua>
                            zi = re.compile('<zi>(\S)</zi>')
                            bihua = re.compile('<bihua>([0-9]+\s*)</bihua>')
                            zi = re.findall(zi, text)
                            bihua = re.findall(bihua, text)
                            print(zi[0])
                            print(bihua[0])
                            writer.writerow((i - 5, zi[0], bihua[0]))  # pingyin,bushou,jiegou))
            else:
                writer.writerow((i - 5, '', ''))  # pingyin,bushou,jiegou))

                            # dom = xml.dom.minidom.parse(xml_path)
                            # root = dom.documentElement
                            # xmin = root.getElementsByTagName('zi')

                            # tree = ET.parse(xml_path)
                            # zi = tree.find('info').find('zi').text
                            # print(zi)

                            # bihua = tree.findall('bihua')
                            # pingyin = tree.findall('pingyin')
                            # bushou = tree.findall('bushou')
                            # jiegou = tree.findall('jiegou')

                            # writer.writerow((i-5,zi,bihua))    #pingyin,bushou,jiegou))

                            # frompath = xml_path
                            # topath = outDirName+"\\"+"{}".format(i-5).zfill(4)
                            # if not os.path.isdir(topath):
                            #     os.makedirs(topath)
                            # copy(frompath, topath)
        i += 1