# -*- coding: utf-8 -*-

from collections import Counter
from make_news_sum import summarize_test
from kiwipiepy import Kiwi
kiwi = Kiwi()


def main():
    test_context = '''
    '''
    rtn = summarize_test(test_context)
    rtn_text = ''
    for sent in rtn:
        rtn_text = rtn_text + ' ' + sent.text
    
    # print(rtn_text)
    rtn_text_tokens = kiwi.tokenize(rtn_text)
    rtn_text_nn_tokens = [token.form for token in rtn_text_tokens if token.tag == 'NNG' or token.tag == 'NNP']
    print(Counter(rtn_text_nn_tokens))
    # print(rtn_text_nn_tokens)
    exit()

if __name__ == "__main__":
    main()


