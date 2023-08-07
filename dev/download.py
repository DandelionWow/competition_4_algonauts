# 如果是folder则创建文件夹，如果是object则下载
# folder名字来自于data-prefix
# object名字来自于download

# Author: T_Xu(create), S_Sun(modify)
'''
获取每个POI对应的图片集和review summary

POI信息来自meta-xxx.json(from Google Local Data), 根据POI的url获取信息.
下载的数据保存在'./dataset/xxx(region name)/downloaded_multimodal_data/'中
其中, 'download.log'为下载日志, 'xxx.png'为下载图片, 'skip_img_file'为已下载的POI的图片(用于下次跳过不再下载), 
'review_summary.json'为下载的review summary, 'skip_review_summary_file'为已下载的POI的review summary(用于跳过). 
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import requests
import os
import json
from fake_useragent import UserAgent
import random
import time
import logging
import argparse

class Chrome:
    def __init__(self, proxy, driver_path, dataset_prefix_path, base_url):
        self.dataset_prefix_path = dataset_prefix_path
        self.base_url = base_url
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 无头模式，不需要打开浏览器窗口
        # options.add_argument('--proxy-server=' + proxy)  # 设置代理
        options.add_argument('--log-level=3')  # 禁用日志输出
        self.driver = webdriver.Chrome(driver_path, options=options) # Replace with the path to your ChromeDriver executable

    def get(self, uri):
        # 发请求
        self.driver.get(self.base_url if uri == '' else self.base_url + '#' + uri) 

    def download_data(self, cur_dir: str=''):
        # 睡一会
        time.sleep(1.5)
        # 当前路径
        cur_path = os.path.join(self.dataset_prefix_path, cur_dir)
        os.makedirs(cur_path, exist_ok=True) # 若不存在，则创建
        print('=======目标路径=======')
        print(cur_path)
        
        # 找到分页，遍历所有page
        pagination_list = self.driver.find_elements(By.CSS_SELECTOR, ".pagination li")
        for i, _ in enumerate(pagination_list):
            # 找到文件/文件夹列表
            s3_objects = self.driver.find_element(By.ID, "tbody-s3objects")
            folders_or_objects = s3_objects.find_elements(By.CSS_SELECTOR, 'a')
            # 遍历
            for f_or_o in folders_or_objects:
                attr_data_s3 = f_or_o.get_attribute('data-s3')
                if attr_data_s3 == 'folder': # 文件夹，需要递归调用
                    attr_data_prefix = f_or_o.get_attribute('data-prefix')
                    # 跳过已下载的文件夹
                    if attr_data_prefix in skip_img_set:
                        continue

                    # 递归文件夹内的内容
                    chrome = Chrome(args.proxy, args.chromedriver_path, args.parent_path, args.base_url)
                    chrome.get(attr_data_prefix)
                    chrome.download_data(attr_data_prefix)
                elif attr_data_s3 == 'object': # 文件，直接下载
                    attr_href = f_or_o.get_attribute('href') # 文件下载链接
                    attr_download = f_or_o.get_attribute('download') # 当作文件名
                    # 下载
                    object = self.http_get(attr_href)
                    # 保存
                    object_path = os.path.join(cur_path, attr_download)
                    with open(object_path, 'wb') as f:
                        f.write(object.content)
                        f.flush()
                        f.close()
                        # 保存下载记录
                        skip_file.write(self.driver.current_url.split('#')[1] + '\n')
                        skip_file.flush()
                        print('+++下载成功+++ ---> '+attr_href)
            # 若只有1页是3个元素
            if (i + 3) >= len(pagination_list):
                break
            # 下一页
            pagination_list[len(pagination_list) - 1].find_element(By.CSS_SELECTOR, 'a').click()
            time.sleep(1.3)
        # 关闭浏览器
        self.driver.quit()

    def http_get(self, url):
        # 随机生成User-Agent头信息，用于反爬虫
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        # get请求，自动跳转，添加头信息
        res = requests.get(url, allow_redirects=True, headers=headers, verify=False)
        
        return res

if __name__ == '__main__':
    # paremeters
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', type=str, default='http://127.0.0.1:7890', help='proxy server(http://ip+port)')
    parser.add_argument('--chromedriver_path', type=str, default='D:\Development-Files\ChromeDriver\chromedriver.exe', help='chrome driver local path')
    parser.add_argument('--parent_path', type=str, default='E:/dev/dataset/nsd', help='dataset path')
    parser.add_argument('--base_url', type=str, default='https://natural-scenes-dataset.s3.amazonaws.com/index.html', help='dataset websit url')
    args, _ = parser.parse_known_args()

    # path设置
    os.makedirs(args.parent_path, exist_ok=True)
    # 跳过文件
    skip_file_path = os.path.join(args.parent_path, 'skip_file')
    skip_file = open(skip_file_path, 'a+')
    skip_file.seek(0) # 光标移到文件开始
    skip_img_set = set(skip_file.read().splitlines()) # 记录：uri

    # 初始化浏览器
    chrome = Chrome(args.proxy, args.chromedriver_path, args.parent_path, args.base_url)
    # 打开网页
    chrome.get('')
    # 下载数据
    chrome.download_data()

    print('++++下载完成++++')