import httplib2, os
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

# Gmail権限のスコープを指定
SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
# ダウンロードした権限ファイルのパス
CLIENT_SECRET_FILE = 'client_id.json'
# ユーザーごとの設定ファイルの保存パス
USER_SECRET_FILE = 'credentials-gmail.json'

# ユーザー認証データの取得
def gmail_user_auth():
    # ユーザーの認証データの読み取り
    store = Storage(USER_SECRET_FILE)
    credentials = store.get()
    # ユーザーが認証済みか?
    if not credentials or credentials.invalid:
        # 新規で認証する
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = 'Python Gmail API'
        credentials = tools.run_flow(flow, store, None)
        print('認証結果を保存しました:' + USER_SECRET_FILE)
    return credentials

# 認証実行
gmail_user_auth()