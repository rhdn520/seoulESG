from os.path import exists
import requests
import csv
import datetime
import json
import re
from kiwipiepy import Kiwi
kiwi = Kiwi()

kiwi.load_user_dictionary('user_dict.txt')
with open(f'dictionary/gov-public-agent-list.csv', 'r', newline = '', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        kiwi.add_user_word(row['name'])

with open('dictionary/corporation-list.csv', 'r', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        if(row['name'].count(' ') > 0): #기업명에 띄어쓰기 포함된 경우 제외
            continue
        kiwi.add_user_word(row['name'])



def trim_zerospace(w):
    newString = (w.encode('ascii', 'ignore')).decode("utf-8")
    return newString

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def clean_text(input):
    output = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', input)
    return output

##query keywords
queryKeyword = []
with open('dictionary/esg-keyword.txt','r',encoding='utf-8') as txtfile:
    lines = txtfile.readlines()
    for line in lines:
        queryKeyword.append(line.replace('\n',''))


##define today's date
startDate = '2022-09-01'
endDate = '2022-09-02'
print(startDate)
print(endDate)

print(" OR ".join(queryKeyword))
keywordPayload = {
    "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
    "argument": {
        "query":{
            "content": " OR ".join(queryKeyword)
        }, 
        "published_at": {
            "from": startDate,
            "until": endDate
        },
        "provider": [
            "경향신문",
            "국민일보",
            "내일신문",
            "동아일보",
            "문화일보",
            "서울신문",
            "세계일보",
            "조선일보",
            "중앙일보",
            "한겨레",
            "한국일보",
            # "경기일보",
            # "경인일보",
            # "강원도민일보",
            # "강원일보",
            # "대전일보",
            # "중도일보",
            # "중부매일",
            # "중부일보",
            # "충북일보",
            # "충청일보",
            # "충청투데이",
            # "경남신문",
            # "경남도민일보",
            # "경상일보",
            # "국제신문",
            # "대구일보",
            # "매일신문",
            # "부산일보",
            # "영남일보",
            # "울산매일",
            # "광주매일신문",
            # "광주일보",
            # "무등일보",
            # "전남일보",
            # "전북도민일보",
            # "전북일보",
            # "제민일보",
            # "한라일보",
            "매일경제",
            "머니투데이",
            "서울경제",
            "파이낸셜뉴스",
            "한국경제",
            "헤럴드경제",
            "아시아경제",
            "아주경제",
            "전자신문",
            "KBS",
            "MBC",
            "SBS",
            "YTN",
            "OBS"
        ],
        "category": [
            '경제'
        ],
        "category_incident": [
            # // "범죄",
            # // "교통사고",
            # // "재해>자연재해" 
        ],
        "sort": { "date": "asc" },
        # // "hilight": 200,
        # // "return_from": 0,
        "return_size": 10000,
        "fields": [
            "news_id",
            "title",
            "content",
            "published_at",
            "provider",
            "provider_news_id",
            "category",
            "category_incident",
            "byline",
            "images",
            "provider_link_page"
        ]
    }
}

##Get Query Results
# categoryQueryResult = requests.post("http://tools.kinds.or.kr:8888/search/news",json=categoryPayload)
keywordQueryResult = requests.post("http://tools.kinds.or.kr:8888/search/news",json=keywordPayload)
# queryResult = categoryQueryResult.json()['return_object']['documents'] + keywordQueryResult.json()['return_object']['documents']
print("API Query Finished...")

#delete duplicate news
print("Deleting Duplicate News")
newsList = []
newsIdList = []
for news in keywordQueryResult.json()['return_object']['documents']: 
    if news['news_id'] in newsIdList:
        continue
    token = kiwi.tokenize(clean_text(news['content']))
    morph = [item.form for item in token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
    morph = set(morph)
    news['token_set'] = list(morph)
    newsList.append(news)
    newsIdList.append(news['news_id'])

print(f"Queried {len(newsList)} News!!")

with open(f'esg-news-list-json/esg{startDate}-{endDate}.json','w', encoding='UTF-8') as f:
    json.dump(newsList, f, indent=2, ensure_ascii=False)
