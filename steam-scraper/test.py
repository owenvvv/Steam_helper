from scrapy.loader.processors import Compose, Join, MapCompose, TakeFirst
import pandas as pd
"""
pipi = Compose(lambda x: x[0], str.upper)
print(pipi(['iss', 'nus', 'mtech', 'ebac']))

pipi = MapCompose(lambda x: x[0], str.upper)
print(pipi(['iss', 'nus', 'mtech', 'ebac']))
"""
steam_id = pd.read_csv("D:\\NUS BA\\class\\nlp\\Project\\steam-scraper-master\\steam\\spiders\\steam_id.csv", header=None)
steam_id = list(steam_id.iloc[:,0])
print(len(steam_id))
print(len(list(set(steam_id))))
#steam_id .to_csv("steam_id.csv",header=False,index=False)