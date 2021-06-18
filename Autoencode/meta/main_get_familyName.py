import os, glob, time, argparse
from urllib import parse
import numpy as np 
import pandas as pd
# import requests
# from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--waittime", type=int, default=20, help="waiting time for 'WebDriverWait(driver, waittime)', default=20")
parser.add_argument("--sleeptime", type=int, default=3, help="sleep time between actions', default=3")
parser.add_argument("--start", type=int, default=0, help="if break,  start fron ith iteration , default=0")
args = parser.parse_args()
# args = parser.parse_args(args=[]) # in jupyter
wait_time = args.waittime
sleep_time =args.sleeptime
start = args.start
print(args)


# 啟動瀏覽器工具的選項
# 使用 Chrome 的 WebDriver (含 options)
options = webdriver.ChromeOptions()
# options.add_argument("--start-maximized")         #最大化視窗
options.add_argument("--incognito")                 #開啟無痕模式
options.add_argument("--disable-popup-blocking ")   #禁用彈出攔截
options.add_argument('--disable-gpu')               #官方文件建議設置，避免各種bug
options.add_argument('blink-settings=imagesEnabled=false')             #不加载图片, 提升速度
options.add_experimental_option('excludeSwitches', ['enable-logging']) #禁止打印日志
options.add_argument('--headless')                  #不開啟實體瀏覽器背景執行

# 建立科名:屬名對應的Series資料集，方便後續取用
moth_meta = pd.read_csv(f'moth_meta.csv')
family_genus = (moth_meta.groupby(['Family','Genus']).count()
                .reset_index(level='Genus')['Genus'])
# print(len(family_genus)) # 4949

# 檢視屬名屬名對應多個科名的資料
# mask = family_genus.duplicated(keep=False)
# family_genus[mask].sort_values()             # 有47筆 
# moth_meta



#---------------------Define fuctions-----------------------------------------------------------------------------------------------------------------------------


def get_url(family: str = '', genus: str = '') -> str :
    '''input query targets'''
    family = family
    genus = genus
    website = 'https://www.nhm.ac.uk/our-science/data/butmoth/search/GenusList3.dsml?'
    query ={
        'FAMILY':family,
        'GENUS':genus
    }
    query_url = website + parse.urlencode(query)
    return query_url

def wait_class_element(class_: str = '', driver=None) -> None: 
    ''' 當強制等待後，class_element還是沒有抓到想要的內容時，送入迴圈處理'''
    c=0
    while class_ == '':                  # 當強制等待後，target還是沒有抓到想要的內容時，送入迴圈處理
        c+=1
        sleep(1)                         # 強制等待後重新補抓
        class_ = driver.find_element(By.CSS_SELECTOR, class_selector).text  
        if class_ == '':
            print(f'\t\tWait again {c:2d}', end='\r')
            if c>5:                      # 迴圈重複執行n次還是沒結果的話，強制引發異常，並執行
                class_ = "can't get classes"
                raise Exception(f"\tAssign classes as '{class_}'")
            continue                     # 如果還是沒抓到的話，再回到迴圈等待
        else:
            print(f'\t\tBreak loop')
            break
    return class_


def get_higherClassification(family: str = '', genus: str = '', url: str = None) -> str:
    ''' 函式前需先指定driver為全域變數'''
    assert(family!=''); assert(genus!=''); assert(url!='')
    try:
        driver.get(url)  # 開啟網頁
        print('\tLaunch url')

        # 等待Accept cookies畫面跳出，並按下 
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#cookie-banner-form > button')))
        driver.find_element(By.CSS_SELECTOR, '#cookie-banner-form > button').click()
        print('\tClick "Accept cookies"')

        # 輸入的資料查詢到結果，如果在這一步查詢不到，則交由NoSuchElementException攔截處理
        getrecord  = driver.find_element(By.CLASS_NAME, 'RecSetLoc')
        print(f'\t{getrecord.text}')  # 如果能執行到這一步，表示屬名有資料

        # 等待查詢物種畫面跳出、按下欲查詢的屬名
        wait.until(EC.presence_of_element_located((By.LINK_TEXT, genus)))
        genus_elements = driver.find_elements(By.LINK_TEXT, genus)
        for link in genus_elements:
            link = link.get_attribute('href')                    # 取得連結
            driver.execute_script('window.open("'+link+'");')    # 開啟分頁

        sleep(sleep_time)                                        # 等待一秒等新的分頁跳出否則又會跑出cookies
        driver.close()                                           # 開啟分頁B後將分頁A關閉 

        # 獲取目前視窗控制碼(list) 
        handles = driver.window_handles

        # 取得類別名稱
        # 等待屬名詳細頁面跳出，抓取Higher classification欄位內的文字 'Higher classification:\nSuperfamily : Family : SubFamily : Tribe'
        classes = []
        for handle in handles:                                 
            driver.switch_to.window(handle)                                                  # 切換至不同分頁進行操作
            sleep(sleep_time)                                                                # 等待時間不夠久會出現抓不到 class_selector        
            class_selector = '#microsite-body > table.dataTable_ms > tbody > tr:nth-child(3) > td > p'
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, class_selector)))
            sleep(sleep_time)                                                                # 這邊需要強制等待一段時間才能抓到
            class_ = driver.find_element(By.CSS_SELECTOR, class_selector).text
            
            # 如果等待時間不夠久，class_尚未抓到內文，則送入 wait_class_element執行等待，直到取值
            if class_ == '' :                                       
                class_ = wait_class_element(class_=class_, driver=driver)     
            class_ = class_.split('\n')[1]
            classes.append(class_)

        # 比對目前的科名，如果能與目前科名一致，就填入一致的，否則保留目前科名(1或多個)
        if classes != 'genus not exist':
            print('\tComparing family name ...')
            for c in classes:
                family_ = c.split(':')[1].strip()
                if family.upper() == family_:
                    classes = c
            # 如果沒有比對出科名的，資料型態會是list，則取第一筆連結的資料 !事後確認 
            if type(classes) is list:  
                classes = classes[0]
            
        print(f'\tGot "{classes}" !')
        return classes

    except NoSuchElementException as exc:    # 攔截查詢不到的錯誤訊息 
        error_selector = '#microsite-body > table > tbody > tr > td > p.msgError.center'
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, error_selector)))  # 等待目標元素出現
        error_element = driver.find_element(By.CSS_SELECTOR, error_selector)
        classes = "genus not exist"
        print(f"\tCan't find '{family} {genus}' Assign classes as '{classes}' !")
        return classes

    except IndexError as e:
        print(e)
    except TimeoutException:
        print("\t等待逾時，關閉瀏覽器")
        sleep(1)
        driver.quit()
    except Exception as e:
        print(e)


def write_file(i, family: str, genus: str, classes: str) -> None:
    with open('./classification.csv','a') as file:
        file.write(",".join([str(i), family ,genus, f'{str(classes)}\n']))  # 最後一項要加入換行符號 \n 

#---------------------Define main()-------------------------------------------------------------------------------------------------------------------------------
def main():
    global driver, wait
    start_time = time.time()
    for i, (family, genus) in enumerate(family_genus[start:].items(), start=start) : 
        print(f'{i}/{len(family_genus)}, Querying... {family}, {genus}')

        driver = webdriver.Chrome(options = options)
        wait = WebDriverWait(driver, wait_time)

        url = get_url(family='', genus=genus)    #　僅用屬名查詢
        classes = get_higherClassification(family=family, genus=genus, url=url)
        write_file(i, family, genus, classes)
        pass_ = time.time() - start_time
        print(f'Time Passed: {pass_//(60*60):3.0f}h,{pass_//60%60:2.0f}m,{pass_%60:2.0f}s.')

        driver.quit()
    

'''主程式'''
if __name__ == "__main__":
    main()


