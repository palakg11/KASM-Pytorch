import gensim
import pickle as cPickle
import time
import sys
from collections import defaultdict

class Dataset(object):
    
    def load_data(self, dataset):
        
        self.data = dataset
        
        if self.data == 'yahoo':
            self.loadpath = "./data/LEAM_dataset/yahoo.p"
            self.embpath = "./data/LEAM_dataset/yahoo_glove.p"
            self.num_class = 10
            self.class_name = ['Society Culture',
                'Science Mathematics',
                'Health' ,
                'Education Reference' ,
                'Computers Internet' ,
                'Sports' ,
                'Business Finance' ,
                'Entertainment Music' ,
                'Family Relationships' ,
                'Politics Government']
        elif self.data == 'agnews':
            self.loadpath = "../ag_news.p"
            self.embpath = "../ag_news_glove.p"
            self.num_class = 4
            self.class_name = ['World',
                            'Sports',
                            'Business',
                            'Science']    
        elif self.data == 'dbpedia':
            self.loadpath = "./data/LEAM_dataset/dbpedia.p"
            self.embpath = "./data/LEAM_dataset/dbpedia_glove.p"
            self.num_class = 14
            self.class_name = ['Company',
                'Educational Institution',
                'Artist',
                'Athlete',
                'Office Holder',
                'Mean Of Transportation',
                'Building',
                'Natural Place',
                'Village',
                'Animal',
                'Plant',
                'Album',
                'Film',
                'Written Work',
                ]
        elif self.data == 'yelp_full':
            self.loadpath = "./data/LEAM_dataset/yelp_full.p"
            self.embpath = "./data/LEAM_dataset/yelp_full_glove.p"
            self.num_class = 5
            self.class_name = ['worst',
                            'bad',
                            'middle',
                            'good',
                            'best']

        x = cPickle.load(open(self.loadpath, "rb"), encoding = "latin1")
        self.train, self.val, self.test = x[0], x[1], x[2]
        self.train_lab, self.val_lab, self.test_lab = x[3], x[4], x[5]
        self.wordtoix, self.ixtoword = x[6], x[7]
        del x
        
        print("load data finished:", self.data)