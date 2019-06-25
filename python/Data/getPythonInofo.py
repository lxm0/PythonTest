import requests
from bs4 import BeautifulSoup
if __name__ == '__main__':
    num=0

    url = 'https://bugs.python.org/issue35424'

    req = requests.get(url,{'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'})

    soup = BeautifulSoup(req.text,'lxml')
    liResutl = soup.findAll('', attrs = "")
    print (liResutl)
    # print (req.content)

