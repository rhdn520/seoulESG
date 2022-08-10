import requests
import json
import re
from kiwipiepy import Kiwi
kiwi = Kiwi()

def clean_text(input):
    output = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', input)
    return output


access_key = "511b5d13-222e-4bb9-8fa1-0ec4491d7166"

querykeyword = '여성 임원'

# word_cloud_payload = {
#     "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
#     "argument":{
#         "query":querykeyword,
#         "published_at":{
#             "from" : "2022-05-01",
#             "until": "2022-08-09"
#         }
#     }
# }

# payload2 = {
#      "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
#      "argument":{}
# }

# result2 = requests.post("http://tools.kinds.or.kr:8888/today_category_keyword", json=payload2)

# print(result2.json()['return_object']['cate_keyword'])
# for keyword in result2.json()['return_object']['cate_keyword']:
#     print(keyword)


# result3 = requests.post("http://tools.kinds.or.kr:8888/word_cloud", json=word_cloud_payload)
# keywords = sorted(result3.json()['return_object']['nodes'], key=lambda x:x['weight'], reverse=True)

# queryList = []

# count = 0
# while(count < 11):
#     name = keywords[count]['name']
#     queryList.append(f'(\'{querykeyword}\' AND \'{name}\')')
#     count = count + 1

# print(" OR ".join(queryList))


keywordPayload = {
    "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
    "argument": {
        "query":{
            "title":querykeyword,
            # "content": querykeyword
        }, 
        "published_at": {
            "from": '2022-05-01',
            "until": '2022-08-11'
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

keywordQueryResult = requests.post("http://tools.kinds.or.kr:8888/search/news",json=keywordPayload)
# print(keywordQueryResult.json())
queryResult = keywordQueryResult.json()['return_object']['documents']


newsList = []
newsIdList = []
for news in queryResult: 
    if news['news_id'] in newsIdList:
        continue
    token = kiwi.tokenize(clean_text(news['content']))
    morph = [item.form for item in token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
    morph = set(morph)
    news['token_set'] = list(morph)
    newsList.append(news)
    newsIdList.append(news['news_id'])

print(f"Queried {len(newsList)} News!!")

with open(f'esg-news-list-json/{querykeyword}.json','w', encoding='UTF-8') as f:
    json.dump(newsList, f, indent=2, ensure_ascii=False)
