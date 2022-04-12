
# coding: utf-8

# 本番

# In[68]:


# coding: UTF-8
from time import sleep
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import httplib2, os
from apiclient import discovery
from oauth2client import client, tools
from oauth2client.file import Storage

import re
import pandas as pd
from datetime import datetime


# In[2]:


if __name__ == '__main__':
 
    # URL関連
    url = "https://eth-trade.jp/"
    login = "hiyokomamitsu@gmail.com"
    password = "Allisgame3574"
 
    # ヘッドレスモードの設定。
    # True => ブラウザを描写しない。
    # False => ブラウザを描写する。
    options = Options()
    options.set_headless(True)
 
    # Chromeを起動
    driver = webdriver.Chrome(executable_path="./chromedriver", chrome_options=options)
 
    # ログインページを開く
    driver.get(url)
 
    # ログオン処理
    # ユーザー名入力
    driver.find_element_by_id("UserDataLoginId").send_keys(login)
    # パスワード入力
    driver.find_element_by_id("UserDataLoginPass").send_keys(password)
    driver.find_element_by_class_name("btn").send_keys(Keys.ENTER)   
    
    # ブラウザの描写が完了させるためにsleep
    sleep(10)


# In[3]:



# Gmail権限のスコープを指定
SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
# 認証ファイル
CLIENT_SECRET_FILE = 'client_id.json'
USER_SECRET_FILE = 'credentials-gmail.json'
# ------------------------------------
# ユーザ認証データの取得
def gmail_user_auth():
    store = Storage(USER_SECRET_FILE)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = 'Python Gmail API'
        credentials = tools.run_flow(flow, store, None)
        print('認証結果を保存しました:' + USER_SECRET_FILE)
    return credentials
# Gmailのサービスを取得
def gmail_get_service():
    credentials = gmail_user_auth()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    return service
# ------------------------------------
# GmailのAPIが使えるようにする
service = gmail_get_service()


# In[4]:


# pincodeを取得
messages = service.users().messages()
msg_list = messages.list(userId='me', maxResults=2).execute()
# 先頭のメッセージ情報を得る
msg = msg_list['messages'][0]
# idを得る
id = msg['id']
threadid = msg['threadId']
# メッセージの本体を取得する
data = messages.get(userId='me', id=id).execute()
pattern=r'([+-]?[0-9]+\.?[0-9]*)'
#pincode = data['snippet'][27:33]
pincode = re.findall(pattern,str(data['snippet']))


# In[5]:


driver.find_element_by_id("PinDataCode").send_keys(pincode[0])
driver.find_element_by_class_name("btn").click()
sleep(5)

driver.find_element_by_class_name("menubtn").click()
sleep(5)
driver.find_element_by_class_name("icon07").click()
sleep(5)


# In[93]:


soup = BeautifulSoup(driver.page_source, "lxml")
driver.close()
money_html = soup.find_all(class_="blance-wrap")
pattern=r'([+-]?[0-9]+\.?[0-9]*)'

money_data =re.findall(pattern,str(money_html))
date = datetime.now().strftime("%Y/%m/%d")
time = datetime.now().strftime("%H:%M:%S")


# In[94]:


df =pd.DataFrame({"date" : [date],
              "time" : [time],
              "USDT" : [money_data[0]],
             "ETH" :[money_data[1]]})
# 一発目用
#df.to_csv("eth_trade.csv", index=False)
df.to_csv("eth_trade.csv", index=False, mode='a', header=False)

