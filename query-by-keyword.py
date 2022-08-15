from calendar import c
import requests
import json
import re
from kiwipiepy import Kiwi
kiwi = Kiwi()
from datetime import datetime
from datetime import timedelta

def clean_text(input):
    output = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', input)
    return output


access_key = "511b5d13-222e-4bb9-8fa1-0ec4491d7166"

querykeyword = input('검색할 단어를 입력하세요')
period = int(input('최근 몇일간의 기사를 검색하시겠습니까?'))
today = datetime.today()
from_date = (datetime.today() - timedelta(days=period)).strftime('%Y-%m-%d')
until_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')


print(f'\'{querykeyword}\' 키워드로 지난 {period}일 간의 기사를 검색합니다')

keywordPayload = {
    "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
    "argument": {
        "query":{
            # "title":querykeyword,
            "content": querykeyword
        }, 
        "published_at": {
            "from": from_date,
            "until": until_date
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
            # "provider_news_id",
            "category",
            "category_incident",
            "byline",
            # "images",
            "provider_link_page"
        ]
    }
}

server_response = requests.post("http://tools.kinds.or.kr:8888/search/news",json=keywordPayload)
# print(keywordQueryResult.json())
queryResult = server_response.json()['return_object']['documents']
print(f"queried {len(queryResult)} news!")

#뉴스 기사 별로 토크나이즈
for news in queryResult: 
    token = kiwi.tokenize(clean_text(news['content']))
    morph = [item.form for item in token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
    morph = set(morph)
    news['token_set'] = list(morph)


# 뉴스 기사 날짜별로 묶기
def find(list, key, value):
    for i, dic in enumerate(list):
        if dic[key] == value:
            return i
    return -1

def cluster_by_date(queryResult):
    date_cluster_list = []
    for news in queryResult:
        index = find(date_cluster_list, 'date', news['published_at'][0:10])
        if(index >= 0):
            date_cluster_list[index]['news_list'].append(news)
            date_cluster_list[index]['count'] = date_cluster_list[index]['count'] + 1
        else:
            new_date_cluster = {
                'date': news['dateline'][0:10],
                'count': 1,
                'news_list':[
                    news
                ]
            }
            date_cluster_list.append(new_date_cluster)
    
    return date_cluster_list

news_clusters_by_date = cluster_by_date(queryResult= queryResult)

#날짜별로 묶인 뉴스기사들 다시 주제별로 클러스터링하기(자카드)

threadhold = 0.2

def jaccard_from_string_set(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return (len(set1 & set2) / len(set1 | set2))


for cluster_by_date in news_clusters_by_date:
    jaccard_calculate = []
    for index,news in enumerate(cluster_by_date['news_list']):
        clustertmp = [index]
        jndex = index + 1
        while jndex < len(cluster_by_date['news_list']):
            if (jaccard_from_string_set(cluster_by_date['news_list'][index]['token_set'], cluster_by_date['news_list'][jndex]['token_set']) > threadhold):
                clustertmp.append(jndex)
            jndex = jndex + 1
        jaccard_calculate.append(clustertmp)
    
    clusters = []
    ignoreIdx = []
    count = 0

    while count < len(jaccard_calculate):
        if count in ignoreIdx:
            count = count + 1
            continue

        for index in jaccard_calculate[count]:
            if(len(set(jaccard_calculate[index]) - set(jaccard_calculate[count])) != 0):
                jaccard_calculate[count] = jaccard_calculate[count] + list(set(jaccard_calculate[index]) - set(jaccard_calculate[count]))
        
        clusters.append(jaccard_calculate[count])
        ignoreIdx = ignoreIdx + jaccard_calculate[count]
        count = count + 1

    i = 0
    while i < len(clusters):
        j = i + 1
        while j < len(clusters):
            if(len(set(clusters[i]) & set(clusters[j])) > 0):
                duplicateItems = list(set(clusters[i]) & set(clusters[j]))
                
                #make cluster without duplicate items
                clusterI = list(set(clusters[i]) - (set(clusters[i]) & set(clusters[j])))
                clusterJ = list(set(clusters[i]) - (set(clusters[i]) & set(clusters[j])))

                clusterIToken = []
                for idx in clusterI:
                    clusterIToken = clusterIToken + cluster_by_date['news_list'][idx]['token_set']
                clusterITokenSet = set(clusterIToken)

                clusterJToken = []
                for idx in clusterJ:
                    clusterJToken = clusterJToken + cluster_by_date['news_list'][idx]['token_set']
                clusterJTokenSet = set(clusterJToken)

                for item in duplicateItems:
                    jaccardI = len(clusterITokenSet & set(cluster_by_date['news_list'][item]['token_set'])) / len(clusterITokenSet | set(cluster_by_date['news_list'][item]['token_set']))
                    jaccardJ = len(clusterJTokenSet & set(cluster_by_date['news_list'][item]['token_set'])) / len(clusterJTokenSet | set(cluster_by_date['news_list'][item]['token_set']))
                    if jaccardI > jaccardJ:
                        clusters[j] = list(set(clusters[j])-set([item]))
                    else:
                        clusters[i] = list(set(clusters[i])-set([item]))
                
                i=0
                j=0
            j = j + 1
        i = i + 1
    
    cluster_by_date['cluster_by_jaccard'] = []
    for cluster in clusters:
        tmp = []
        for index in cluster:
            tmp.append(cluster_by_date['news_list'][index])
        cluster_by_date['cluster_by_jaccard'].append(tmp)

print(news_clusters_by_date[0]['cluster_by_jaccard'])



# newsList = []
# newsIdList = []
# for news in queryResult: 
#     if news['news_id'] in newsIdList:
#         continue
#     token = kiwi.tokenize(clean_text(news['content']))
#     morph = [item.form for item in token if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
#     morph = set(morph)
#     news['token_set'] = list(morph)
#     newsList.append(news)
#     newsIdList.append(news['news_id'])

# print(f"Queried {len(newsList)} News!!")

# with open(f'esg-news-list-json/{querykeyword}.json','w', encoding='UTF-8') as f:
#     json.dump(newsList, f, indent=2, ensure_ascii=False)

file = open(f'cluster/{querykeyword}.txt', 'w+')
for cluster in news_clusters_by_date:
    file.write(cluster['date'] + '\n')
    for news_list in cluster['cluster_by_jaccard']:
        for index, news in enumerate(news_list):
            if(index == len(news_list) - 1):
                file.write('\t' + news['title'] + ' | ' + news['provider_link_page'] + '\n' + '\n')
            else:
                file.write('\t' + news['title'] + ' | ' + news['provider_link_page'] + '\n')

file.close()
            