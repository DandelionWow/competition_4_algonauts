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

def http_get(url):
    # 随机生成User-Agent头信息，用于反爬虫
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    # get请求，自动跳转，添加头信息
    try:
        # 绕过代理
        proxies = {'http': None, 'https': None}
        res = requests.get(url, allow_redirects=True, headers=headers, verify=False, timeout=20., proxies=proxies)
    except requests.exceptions.ConnectionError:
        return None
    return res

def get_data_url(chrome: Chrome, cur_dir: str = ''):
    # 睡一会
    time.sleep(random.randint(1, 3))
    # 目标路径
    print('=======目标路径=======')
    print(cur_dir)
    
    # 找到分页，遍历所有page
    pagination_list = chrome.driver.find_elements(By.CSS_SELECTOR, ".pagination li")
    for i, _ in enumerate(pagination_list):
        # 找到文件/文件夹列表
        s3_objects = chrome.driver.find_element(By.ID, "tbody-s3objects")
        folders_or_objects = s3_objects.find_elements(By.CSS_SELECTOR, 'a')
        # 遍历
        for f_or_o in folders_or_objects:
            attr_data_s3 = f_or_o.get_attribute('data-s3')
            if attr_data_s3 == 'folder': # 文件夹，需要递归调用
                attr_data_prefix = f_or_o.get_attribute('data-prefix')
                # 手动跳过一些目录
                if attr_data_prefix in []:
                    continue
                
                # 递归文件夹内的内容
                chrome_ = Chrome(args.proxy, args.chromedriver_path, args.parent_path, args.base_url)
                chrome_.get(attr_data_prefix)
                get_data_url(chrome_, attr_data_prefix)
            elif attr_data_s3 == 'object': # 文件，直接下载
                attr_href = f_or_o.get_attribute('href') # 文件下载链接
                attr_download = f_or_o.get_attribute('download') # 当作文件名

                # 保存url
                obj = {
                    'dir': cur_dir, 
                    'file_path': cur_dir + attr_download, 
                    'url': attr_href,
                }
                line = json.dumps(obj) 
                # 跳过已记录
                if line in data_url_set:
                    continue
                data_url_file.write(line + '\n')
                data_url_file.flush()
                
        # 若只有1页是3个元素
        if (i + 3) >= len(pagination_list):
            break
        # 下一页
        time.sleep(1.8)
        pagination_list = chrome.driver.find_elements(By.CSS_SELECTOR, ".pagination li") # 再定位元素，防止StaleElementReferenceException
        pagination_list[len(pagination_list) - 1].find_element(By.CSS_SELECTOR, 'a').click()
        time.sleep(1.3)
    # 关闭浏览器
    chrome.driver.quit()

def download_data(dataset_prefix_path, data_url_file, skip_file_set):
    # 再次转set
    data_url_file.seek(0)
    data_url_set = set(data_url_file.read().splitlines())
    
    # 遍历
    for obj in data_url_set:
        obj: dict = json.loads(obj)
        # 跳过已下载
        if obj['url'] in skip_file_set:
            continue
        # 创建文件夹
        os.makedirs(os.path.join(dataset_prefix_path, obj['dir']), exist_ok=True)
        # 下载文件
        data = http_get(obj['url'])
        if data is None:
            print('+++下载失败+++ ---> ' + obj['url'])
        else:
            # 保存
            with open(os.path.join(dataset_prefix_path, obj['file_path']), 'wb') as f:
                f.write(data.content)
                f.flush()
                f.close()
                # 保存下载记录
                skip_file.write(obj['url'] + '\n')
                skip_file.flush()
                print('+++下载成功+++ ---> ' + obj['url'])

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
    # 保存下载链接文件 保存格式(每行)：{'dir': '', 'file_path': '', 'url': ''}
    data_url_file_path = os.path.join(args.parent_path, 'data_url_file')
    data_url_file = open(data_url_file_path, 'a+')
    data_url_file.seek(0)
    data_url_set = set(data_url_file.read().splitlines())
    # 跳过文件
    skip_file_path = os.path.join(args.parent_path, 'skip_file')
    skip_file = open(skip_file_path, 'a+')
    skip_file.seek(0) # 光标移到文件开始
    skip_file_set = set(skip_file.read().splitlines()) # 记录：uri

    # 初始化浏览器
    chrome = Chrome(args.proxy, args.chromedriver_path, args.parent_path, args.base_url)
    # 打开网页首页
    chrome.get('')
    # 获取所有数据url
    get_data_url(chrome)

    # 下载数据
    download_data(args.parent_path, data_url_file, skip_file_set)

    print('++++下载完成++++')