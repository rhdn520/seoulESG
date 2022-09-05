# -*- coding: utf-8 -*-
from kiwipiepy import Kiwi
kiwi = Kiwi()
import re
import inquirer
import json

def clean_text(input):
    output = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', input)
    return output


def jaccard_from_string_set(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return (len(set1 & set2) / len(set1 | set2))



cluster_by_date_list = []

with open('esg-news-list-json/LG유플러스 탄소중립.json', encoding='utf-8') as file:
    cluster_by_date_list = json.load(file)

question = [
    inquirer.List('standard',
    message="기준이 될 뉴스를 고르세요",
    choices = [news['title'] for news in cluster_by_date_list[-1]['news_list']]
    )
]

answers = inquirer.prompt(question)
print(answers['standard'])

#make standard token list
standard_cluster_index = None
for index, jaccard_cluster in enumerate(cluster_by_date_list[-1]['cluster_by_jaccard']):
    for news in jaccard_cluster:
        if (news['title'] == answers['standard']):
            # print(news['title'])
            # print(answers['standard'])
            standard_cluster_index = index
            break

standard_token_list = []
for news in cluster_by_date_list[-1]['cluster_by_jaccard'][standard_cluster_index]:
    news_tokens = kiwi.tokenize(clean_text(news['content']))
    news_morph = [item.form for item in news_tokens if item.tag == 'NNG']
    standard_token_list = standard_token_list + list(set(news_morph))

standard_token_list = list(set(standard_token_list))

#select news
#jaccard 클러스터를 돌면서 기준 기사와 자카드 값 비교, 기준치 이상의 클러스터에서 첫 번째 기사를 반환한다
selected_news_list = []
for date_cluster in cluster_by_date_list:
    for jaccard_cluster in date_cluster['cluster_by_jaccard']:
        cluster_tokens_list= []
        for news in jaccard_cluster:
            news_tokens = kiwi.tokenize(clean_text(news['content']))
            news_morph = [item.form for item in news_tokens if item.tag == 'NNG']
            cluster_tokens_list = cluster_tokens_list + list(set(news_morph))
        cluster_tokens_list = list(set(cluster_tokens_list))
        if(jaccard_from_string_set(cluster_tokens_list, standard_token_list) > 0.12):
            selected_news_list.append(jaccard_cluster[0])
    

for news in selected_news_list:
    print(news['title'] + ' | ' + news['provider_link_page'])