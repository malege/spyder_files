# -*- coding: utf-8 -*-
import requests
import re
import csv

def request_dang(url):
    try:
        #同步请求
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
         return None

def parse_result(html):
    pattern = re.compile('style="background:rgb\((.*?)\)" title="(.*?)"><span>')
    items = re.findall(pattern, html)
    return items
    
def main():
    url = 'https://tool.nbchao.com/seka/list/17/'
    html = request_dang(url)
    items = parse_result(html)
    
    with open('d:/pictures/tpgseka.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        #先写columns_name
        writer.writerow(["rgb","ColorCode"])
        for item in items:
            #print('开始写入数据 ====> ' + str(item))
            item = (item[0].replace(',',':'),item[1])
            writer.writerow(item)
            print(item)

if __name__ == "__main__":
    main()