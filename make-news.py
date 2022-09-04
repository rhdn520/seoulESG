from itertools import count
from os.path import exists
import requests
import csv
import datetime
import json
import re
from kiwipiepy import Kiwi
kiwi = Kiwi()
from datetime import datetime
from datetime import timedelta
from collections import Counter

def clean_text(dirty_text):
    cleaned_text = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', dirty_text)
    return cleaned_text

def search_news(searchWord, period=90):
    today = datetime.today()
    until_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=period)).strftime('%Y-%m-%d')
    keywordPayload = {
        "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
        "argument":{
            "query":{
                "content": searchWord
            },
            "published_at":{
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
            "경제"
        ],
        "sort": {"date": "asc"},
        "return_size": 10000,
        "fields":[
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
    
    apiSearchResult = requests.post("http://tools.kinds.or.kr:8888/search/news",json=keywordPayload)

    searchResult = apiSearchResult.json()['return_object']['documents']

    return searchResult

def count_token_occurrence(keyword_morph, content_morph):
    count = 0
    content_counter = Counter(content_morph)
    for morph in keyword_morph:
        if(content_counter.get(morph) == None):
            continue
        count = count + content_counter.get(morph)
    return count/len(content_morph)

def jaccard_from_list(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return (len(set1 & set2) / len(set1 | set2))


def filter_news(keyword, newsList):
    filtered_news_list = []
    keyword_tokens = kiwi.tokenize(clean_text(keyword))
    keyword_morph = [item.form for item in keyword_tokens if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
    keyword_token_list = list(set(keyword_morph))

    for news in newsList:
        #tokenize news
        news_tokens = kiwi.tokenize(clean_text(news['content']))
        news_morph = [item.form for item in news_tokens if item.tag == 'NNG' or item.tag == 'NNP' or item.tag =='NNB' or item.tag == 'NR' or item.tag == 'NP' or item.tag == 'SN']
        news['token_list'] = list(set(news_morph))

        #calculate token occurrence
        token_occurrence = count_token_occurrence(keyword_morph, news_morph)

        #calculate jaccard coefficient with title & keyword
        title_jaccard = jaccard_from_list(keyword_token_list, news['token_list'])

        if((token_occurrence * 2) + (title_jaccard * 3) > 0.055):
            filtered_news_list.append(news)
    
    return filtered_news_list

def find(list, key, value):
    for i, dic in enumerate(list):
        if dic[key] == value:
            return i
    return -1

def cluster_news(news_list):
    jaccard_calculate = []
    for index, news in enumerate(news_list):
        tmp = [index]
        jndex = index + 1
        while jndex < len(news_list):
            if(jaccard_from_list(news_list[index]['token_list'], news_list[jndex]['token_list']) > 0.2):
                tmp.append(jndex)
            jndex = jndex + 1
        jaccard_calculate.append(tmp)

    clusters = []
    ignoreIdx = []
    count = 0
    while count < len(jaccard_calculate):
        if count in ignoreIdx:
            count = count + 1
            continue
        cluster = jaccard_calculate[count]
        for index in cluster:
            if len(set(jaccard_calculate[index]) - set(cluster)) != 0:
                cluster = cluster + list(set(jaccard_calculate[index]) - set(cluster))
        clusters.append(cluster)
        ignoreIdx = ignoreIdx + cluster
        count = count + 1
    
    i = 0
    while i < len(clusters):
        j = i + 1
        while j < len(clusters):
            if(len(set(clusters[i]) & set(clusters[j])) > 0):
                duplicateItems = list(set(clusters[i]) & set(clusters[j]))
                clusterI = list(set(clusters[i]) - (set(clusters[i]) & set(clusters[j])))
                clusterJ = list(set(clusters[i]) - (set(clusters[i]) & set(clusters[j])))

                #make tokens to compare
                clusterIToken = []
                for idx in clusterI:
                    clusterIToken = clusterIToken + news_list[idx]['token_list']
                clusterITokenSet = set(clusterIToken)

                clusterJToken = []
                for idx in clusterJ:
                    clusterJToken = clusterJToken + news_list[idx]['token_list']
                clusterJTokenSet = set(clusterJToken)

                for item in duplicateItems:
                    jaccardI = len(clusterITokenSet & set(news_list[item]['token_list'])) / len(clusterITokenSet | set(news_list[item]['token_list']))
                    jaccardJ = len(clusterJTokenSet & set(news_list[item]['token_list'])) / len(clusterJTokenSet | set(news_list[item]['token_list']))
                    if jaccardI > jaccardJ:
                        clusters[j] = list(set(clusters[j])-set([item]))
                    else:
                        clusters[i] = list(set(clusters[i])-set([item]))
                
                i = 0
                j = 0
            j = j + 1
        i = i + 1
    
    cluster_list = []
    for cluster in clusters:
        date_cluster_list = []
        for news_index in cluster:
            print(news_list[news_index])
            index = find(date_cluster_list, 'date', news_list[news_index]['dateline'][0:10])
            if(index >= 0):
                date_cluster_list[index]['news_list'].append(news_list[news_index])
                date_cluster_list[index]['count'] = date_cluster_list[index]['count'] + 1
            else:
                news_date_cluster = {
                    'date': news_list[news_index]['dateline'][0:10],
                    'count':1,
                    'news_list':[
                        news_list[news_index]
                    ]
                }
                date_cluster_list.append(news_date_cluster)
        
        for date_cluster in date_cluster_list:
            cluster_list.append(date_cluster)
    
    cluster_list = sorted(cluster_list, key=lambda x:x['date'])

    return cluster_list


request_result = search_news('폐배터리 재활용')
filtered_result = filter_news('폐배터리 재활용', request_result)
print(len(request_result))
print(len(filter_news('폐배터리 재활용', request_result)))

with open(f'esg-news-list-json/testtest.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_news(filtered_result), f, indent=2, ensure_ascii=False)