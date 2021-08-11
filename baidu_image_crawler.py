# -*- coding: utf-8 -*-
import requests
import os
import urllib
 
class Spider_baidu_image():
    def __init__(self):
        self.url = 'http://image.baidu.com/search/acjson?'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}
        self.headers_image = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36', 'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E6%89%8B%E5%86%99%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%9B%BE%E7%89%87'}
       
        # self.keyword = input("请输入搜索图片关键字:")
        # self.paginator = int(input("请输入搜索页数，每页30张图片："))
        
        self.keyword = '手写中文文本图片'
        self.paginator = 100
         
        # print(type(self.keyword),self.paginator)
        # exit()
    def get_param(self):
        """
        获取url请求的参数，存入列表并返回
        :return: 
        """
        keyword = urllib.parse.quote(self.keyword)
        params = []
        for i in range(1,self.paginator+1):
            params.append('tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=1&latest=0&copyright=0&word={}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=star&pn={}&rn=30&gsm=1e&1628490893025='.format(keyword,keyword,30*i))
        return params
 
    def get_urls(self,params):
        """
        由url参数返回各个url拼接后的响应，存入列表并返回
        :return:
        """
        urls = []
        for i in params:
            urls.append(self.url + i)
        return urls
    def get_image_url(self,urls):
        image_url = []
        for url in urls:
            json_data = requests.get(url,headers = self.headers).json()
            json_data = json_data.get('data')
            for i in json_data:
                if i:
                    image_url.append(i.get('thumbURL'))
        return image_url
    def get_image(self,image_url):
        """
        根据图片url，在本地目录下新建一个以搜索关键字命名的文件夹，然后将每一个图片存入。
        :param image_url: 
        :return: 
        """
        cwd = os.getcwd()
        file_name = os.path.join(cwd, self.keyword)
        if not os.path.exists(self.keyword):
            os.mkdir(file_name)
        for index,url in enumerate(image_url):
            with open(file_name+'\\' + format(str(index), '0>5s') + '.jpg','wb') as f:
                # pic = requests.get(url,headers = self.headers_image)
                pic = requests.get(url)
                f.write(pic.content)
            if index != 0 and (index+1) % 30 == 0:
                print('{}第{}页下载完成'.format(self.keyword, (index+1)/30))
    def __call__(self, *args, **kwargs):
        params = self.get_param()
        urls = self.get_urls(params)
        image_url = self.get_image_url(urls)
        self.get_image(image_url)
 
if __name__ == '__main__':
    spider = Spider_baidu_image()
    spider()

