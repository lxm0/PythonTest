
# -*- coding: utf-8 -*-
import pandas as pd
import xlrd
import xlwt
from datetime import date,datetime
import csv
def read_excel():
    # csv_reader = csv.reader(open("C:\\Users\\GT\\Desktop\\query.csv",'r'))
    # for row in csv_reader:
    #   print (row)
    with open('C:\\Users\\GT\\Desktop\\query.csv','rt', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    # print (rows)
    address=pd.read_csv("C:\\Users\\GT\\Desktop\\query.csv",encoding="utf-8",usecols=[0])
    # print (address)
    id1 = address["id"][1]
    print(id1)
    for number in address["id"]:
        link = 'https://bugs.python.org/issue'
        link =link+str(number)
        print(link)
        print(number)
    # for row in address:
    #     print (row)
if __name__ == '__main__':
    read_excel()
