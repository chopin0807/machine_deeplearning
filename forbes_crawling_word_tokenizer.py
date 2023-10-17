import requests
from bs4 import BeautifulSoup

URL = 'https://www.forbes.com/sites/adrianbridgwater/2019/04/15/what-drove-the-ai-renaissance/?ss=ai-big-data&sh=164b43231f25'
res = requests.get(URL)
soup = BeautifulSoup(res.text, 'html.parser')
search = soup.select('div.article-body p')
content = ''
for i in search:
    content += i.text

# word_tokenize
import nltk
# nltk punkt tokenizer download
nltk.download('punkt')
from nltk.tokenize import word_tokenize
token1 = word_tokenize(content)
print('word_tokenize: ', token1)
print('=' * 100)
# WordPunctTokenizer() : 알파벳이 아닌문자를 구분하여 토큰화
import nltk
from nltk.tokenize import WordPunctTokenizer
token2 = WordPunctTokenizer().tokenize(content)
print('WordPunctTokenizer: ', token2)
print('=' * 100)
# TreebankWordTokenizer() : 정규표현식에 기반한 토큰화
import nltk
from nltk.tokenize import TreebankWordTokenizer
token = TreebankWordTokenizer().tokenize(content)
print('TreebankWordTokenizer: ', token[:20])
print('=' * 100)
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')
taggedToken = pos_tag(token1)
print('pos_tag: ', taggedToken[:20])
print('=' * 100)
nltk.download('words')
nltk.download('maxent_ne_chunker')
from nltk.tokenize import word_tokenize
# 토큰화
token1 = word_tokenize('Barack Obama likes fried chicken very much')
print('token:',token1)
print('=' * 100)
# pos-tag
taggedToken = pos_tag(token1)
print('pos-tag:',taggedToken)
print('=' * 100)
# chunking
from nltk import ne_chunk
neToken = ne_chunk(taggedToken)
print('ne_chunk: ', neToken)