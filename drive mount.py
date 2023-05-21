# import
from googleapiclient. discovery import build from googleapiclient.http import MediaFileUpload from httplib2 import Http from oauth2client import file
# APT 연결 및 사전정보 입력
store = file.Storage('client_secret_901041769348-259lp90nacgtglknoi1g403bjg3q2k2l.apps.googleusercontent.com') #위에서 받은 OAuth ID Json 파일
creds = store.get ()

service = build('drive', 'v3', http=creds.authorize(Http() ))

folder id = “1Vc2LSYnS7_tGVHQqrukBq3xsRCL5J5Ms" #위에서 복사한 구글드라이브 폴더의 id
file_paths = “file.csv" # 업로드하고자 하는 파일

request_body = {'name': file_paths, “parents': [folder_id], 'uploadType': multipart'} # 얼로드할 파일의 정보 점의
media = MediaFileUpload(file_paths, mimetype='text/cs') # u8
file_info = service. files (). create (body=request_body, media_body=media, fields='id,webViewLink').execute ()
# 구글드라이브 링크 얻기
print ("File webViewLink :",file_info.get('webViewLink'))
webviewlink = file info.get('webViewLink')