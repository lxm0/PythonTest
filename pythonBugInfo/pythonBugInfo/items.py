# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class PythonbuginfoItem(scrapy.Item):
    id = scrapy.Field()
    title = scrapy.Field()
    type = scrapy.Field()
    State = scrapy.Field()
    Component = scrapy.Field()
    Version = scrapy.Field()
    Status = scrapy.Field()
    Resolution = scrapy.Field()
    Dependencies = scrapy.Field()
    Superseder = scrapy.Field()
    Assigned = scrapy.Field()
    Nosy_List = scrapy.Field()

    Priority = scrapy.Field()
    Keywords = scrapy.Field()

