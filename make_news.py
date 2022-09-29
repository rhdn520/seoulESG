#-*- coding: utf-8 -*-

from itertools import count
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
from datetime import datetime
from datetime import timedelta
from collections import Counter
import inquirer
from make_news_sum import summarize_test

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

        if((token_occurrence * 2) + (title_jaccard * 3) > 0.05):
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
            index = find(date_cluster_list, 'date', news_list[news_index]['dateline'][0:10])
            if(index >= 0):
                date_cluster_list[index]['news_list'].append(news_list[news_index])
                date_cluster_list[index]['count'] = date_cluster_list[index]['count'] + 1
            else:
                news_date_cluster = {
                    'cluster_title':news_list[news_index]['title'],
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

def select_news(cluster_list, threadhold = 0.12, NNP = False):
    question = [
        inquirer.List(
            'standard',
            message = "기준이 될 클러스터를 고르세요",
            choices= [cluster['cluster_title'] for cluster in cluster_list[-10:]]
        )
    ]
    answer = inquirer.prompt(question)
    print(answer['standard'])
    standardIndex = find(cluster_list, 'cluster_title', answer['standard'])
    standard_tokens = []
    for news in cluster_list[standardIndex]['news_list']:
        news_tokens = kiwi.tokenize(clean_text(news['content']))
        news_morph = [item.form for item in news_tokens if item.tag == 'NNG']
        if NNP == True:
            news_morph = [item.form for item in news_tokens if item.tag == 'NNG' or item.tag == 'NNP']

        standard_tokens = standard_tokens + list(set(news_morph))
    standard_tokens = list(set(standard_tokens))

    insight_indices = []
    for index,cluster in enumerate(cluster_list):
        # if(index == standardIndex):
        #     continue
        cluster_tokens = []
        for news in cluster['news_list']:
            news_tokens = kiwi.tokenize(clean_text(news['content']))
            news_morph = [item.form for item in news_tokens if item.tag == 'NNG']
            if NNP == True:
                news_morph = [item.form for item in news_tokens if item.tag == 'NNG' or item.tag == 'NNP']

            cluster_tokens = cluster_tokens + list(set(news_morph))
        cluster_tokens = list(set(cluster_tokens))

        if(jaccard_from_list(standard_tokens, cluster_tokens) > threadhold):
            index_obj = {
                'index': index,
                'jaccard': jaccard_from_list(standard_tokens, cluster_tokens)
            }
            insight_indices.append(index_obj)

    insight_indices = sorted(insight_indices, key=lambda x:x['jaccard'], reverse=True)

    selected_cluster = []
    for index in insight_indices:
        selected_cluster.append(cluster_list[index['index']])

    return selected_cluster 


search_keyword = '포스코 청정수소'
date_period = 90
select_threadhold = 0.12
NNP_on = False

request_result = search_news(search_keyword, 120)
print(len(request_result))
filtered_result = filter_news(search_keyword, request_result)
cluster_list = cluster_news(filtered_result)
if (len(cluster_list) < 4):
    print('뉴스의 개수가 너무 적습니다')
    exit()

selected_clusters = select_news(cluster_list, select_threadhold, NNP_on)

question = [
    inquirer.Checkbox('material',
    "글감이 될 뉴스를 선택하세요",
    [cluster['date']+' | '+ cluster['cluster_title'] for cluster in selected_clusters])
]

materials = inquirer.prompt(question)['material']

for material in materials:
    # print(material)
    news = selected_clusters[find(selected_clusters, 'cluster_title', material[13:])]['news_list'][0]
    rtn = summarize_test(news['content'])

    print(news['dateline'][:10] + ' | ' + news['title'] + ' | ' + news['provider_link_page'])
    for sent in rtn:
        print(sent.text)
    
    print('\n\n')
    

# for cluster in selected_clusters:
#     date = cluster['news_list'][0]['dateline'][0:10]
#     title =  cluster['news_list'][0]['title']
#     link = cluster['news_list'][0]['provider_link_page']
#     print(f'{date} | {title} | {link}')



