import requests

word_cloud_payload = {
    "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
    "argument":{
        "query":'ESG',
        "published_at":{
            "from" : "2022-05-01",
            "until": "2022-08-09"
        }
    }
}

result = requests.post("http://tools.kinds.or.kr:8888/word_cloud", json=word_cloud_payload)
keywords = sorted(result.json()['return_object']['nodes'], key=lambda x:x['weight'], reverse=True)

# print(len(keywords))
esg_related_words = []



for i, keyword in enumerate(keywords):
    # print(keyword['name'])
    name = keyword['name']
    if((keyword['name'] in esg_related_words) != True):
        esg_related_words.append(keyword['name'])

    payload = {
    "access_key": "511b5d13-222e-4bb9-8fa1-0ec4491d7166",
    "argument":{
        "query": f'\'ESG\' AND \'{name}\'',
        "published_at":{
            "from" : "2020-01-01",
            "until": "2022-08-11"
        }
    }}
    keyword_of_keyword = requests.post("http://tools.kinds.or.kr:8888/word_cloud", json=payload).json()['return_object']['nodes']
    print(keyword_of_keyword)
    for keyword2 in keyword_of_keyword:
        if((keyword2['name'] in esg_related_words) == False):
            esg_related_words.append(keyword2['name'])

with open('dictionary/esg-related-keyword.txt', 'w') as file:
    for word in esg_related_words:
        file.write(word+'\n')






