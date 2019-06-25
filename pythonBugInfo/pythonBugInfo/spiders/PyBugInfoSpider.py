import re

import pandas as pd
import scrapy
import csv
from pythonBugInfo.items import PythonbuginfoItem


class PyBugInfoSpider(scrapy.Spider):
    name = 'pyBugInfo'
    allowed_domain = ['bugs.python.org']
    start_urls = ['https://bugs.python.org/issue11654']
    def parse(self, response):
        yield scrapy.Request(response.url, callback = self.parse_page)
        address=pd.read_csv("C:\\Users\\GT\\Desktop\\query.csv",encoding="utf-8",usecols=[0])
        for number in address["id"]:
            link = 'https://bugs.python.org/issue'+str(number)
            yield scrapy.Request(link, callback = self.parse_page)

    def parse_page(self, response):
        # for item in response.xpath('//table[@class="form"]'):

            pybunInfo = PythonbuginfoItem()
            id = response.xpath('//div[@id="breadcrumb"]/text()').extract()[0]
            id = re.sub("\D", "", id)
            pybunInfo['id'] = id
            pybunInfo['title']  = response.xpath('//table[1]/tr[1]/td[1]/span/text()').extract()[0]
            item =  response.xpath('//table[1]/tr[2]/td[2]/text()')
            pybunInfo['State'] = item.extract()[0]
            pybunInfo['type'] = response.xpath('//table[1]/tr[2]/td[1]/text()').extract()[0].replace("\n ", "").strip()
            pybunInfo['Component'] = response.xpath('//table[1]/tr[3]/td[1]/text()').extract()[0]
            pybunInfo['Version'] = response.xpath('//table[1]/tr[3]/td[2]/text()').extract()[0]
            pybunInfo['Status'] = response.xpath('//table[@class="form"]/tr[1]/td[1]/text()').extract()[3]
            item  = response.xpath('//table[@class="form"]/tr[2]/td[1]').extract()[1]
            item = re.findall(r"<td>(.+?)</td>", item)
            # print (item)
            pybunInfo['Resolution'] = response.xpath('//table[@class="form"]/tr[1]/td[2]/text()').extract()
            pybunInfo['Dependencies'] = item
            # test= response.xpath('//table[@class="form"]/tr[2]/td[2]').extract()[1]
            # test = re.findall(r"<td>(.+?)</td>", test)
            # print(test)
            pybunInfo['Superseder'] = response.xpath('//table[@class="form"]/tr[2]/td[2]/text()').extract()[1].replace("\n ", "").strip()
            pybunInfo['Assigned'] = response.xpath('//table[@class="form"]/tr[3]/td[1]/text()').extract()[1].replace("\n ", "").strip()
            pybunInfo['Nosy_List'] = response.xpath('//table[@class="form"]/tr[3]/td[2]/text()').extract()[1].replace("\n ", "").strip()
            pybunInfo['Priority'] = response.xpath('//table[@class="form"]/tr[4]/td[1]/text()').extract()[0]
            pybunInfo['Keywords'] = response.xpath('//table[@class="form"]/tr[4]/td[2]/text()').extract()[0]
            # ex = response.xpath('//body/table')
            # print(ex)


            # for i in response.xpath('//table[@class="form"]'):


            yield pybunInfo
