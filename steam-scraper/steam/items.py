# -*- coding: utf-8 -*-
from datetime import datetime, date
import logging

import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import Compose, Join, MapCompose, TakeFirst

logger = logging.getLogger(__name__)


class StripText:#把转义符删去
    def __init__(self, chars=' \r\t\n'):
        self.chars = chars

    def __call__(self, value):
        try:
            return value.strip(self.chars)
        except:  # noqa E722
            return value


def simplify_recommended(x):
    return True if x == 'Recommended' else False



def standardize_date(x):
    """
    Convert x from recognized input formats to desired output format,
    or leave unchanged if input format is not recognized.
    """
    fmt_fail = False

    #b月份缩写，B月份全称，d补零后的日，Y年
    for fmt in ['%b %d, %Y', '%B %d, %Y','%d %B, %Y', '%d %b, %Y']:
        try:
            return datetime.strptime(x, fmt).strftime('%Y-%m-%d')
        except ValueError:
            fmt_fail = True

    # Induce year to current year if it is missing.
    for fmt in ['%b %d', '%B %d','%d %b', '%d %B']:
        try:
            d = datetime.strptime(x, fmt)
            d = d.replace(year=date.today().year)
            return d.strftime('%Y-%m-%d')
        except ValueError:
            fmt_fail = True


    if fmt_fail:
        logger.debug(f'Could not process date {x}')

    return x


def str_to_float(x):
    x = x.replace(',', '')
    try:
        return float(x)
    except:  # noqa E722
        return x


def str_to_int(x):
    try:
        return int(str_to_float(x))
    except:  # noqa E722
        return x


class ReviewItem(scrapy.Item):#指明每个字段的元数据
    product_id = scrapy.Field()
    #page = scrapy.Field()
    #page_order = scrapy.Field()
    recommended = scrapy.Field(
        output_processor=Compose(TakeFirst(), simplify_recommended),
    )
    date = scrapy.Field(
        output_processor=Compose(TakeFirst(), standardize_date)
    )
    text = scrapy.Field(
        input_processor=MapCompose(StripText()),
        output_processor=Compose(Join('\n'), StripText())
    )
    hours = scrapy.Field(
        output_processor=Compose(TakeFirst(), str_to_float)
    )
    found_helpful = scrapy.Field(
        output_processor=Compose(TakeFirst(), str_to_int)
    )
    found_funny = scrapy.Field(
        output_processor=Compose(TakeFirst(), str_to_int)
    )
    username = scrapy.Field()
    user_id = scrapy.Field()
    products = scrapy.Field(
        output_processor=Compose(TakeFirst(), str_to_int)
    )



class ReviewItemLoader(ItemLoader):
    default_output_processor = TakeFirst()
