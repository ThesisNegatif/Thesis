import nltk
from rake_nltk import Rake
import shlex, subprocess
import pandas as pd


text_trial = "Nordstrom is a leading fashion specialty retailer offering apparel, shoes, cosmetics and accessories for women, men, young adults and children. We offer an extensive selection of high-quality brand-name and private label merchandise through our various channels, including Nordstrom U.S. and Canada full-line stores, Nordstrom.com, Nordstrom Rack stores, Nordstromrack.com/HauteLook, Trunk Club clubhouses and TrunkClub.com, Jeffrey boutiques and Last Chance clearance stores. As of January 28, 2017, our stores are located in 40 states throughout the United States and in three provinces in Canada. Our customers can participate in our Nordstrom Rewards loyalty program which allows them to earn points based on their level of spending. We also offer our customers a variety of payment products and services, including credit and debit cards. Our 2016 earnings per diluted share of $2.02, which included a goodwill impairment charge of $1.12, exceeded our outlook of $1.70 to $1.80. Our results were driven by continued operational efficiencies in inventory and expense execution and demonstrated our team’s speed and agility in responding to changes in business conditions. We reached record sales of $14.5 billion for the year, reflecting a net sales increase of 2.9% and comparable sales decrease of 0.4% primarily driven by full-line stores. We achieved the following milestones in multiple growth areas. Our expansion into Canada where we currently have five full-line stores, including two that opened last fall, contributed total sales of $300 in 2016. Nordstrom.com sales reached over $2.5 billion, representing approximately 25% of full-price sales. Our off-price business reached $4.5 billion, with growth mainly driven by our online net sales increase of 32% and 21 new store openings. Off-price continues to be our largest source of new customers, gaining approximately 6 million in 2016. Our expanded Nordstrom Rewards program, which launched in the second quarter, drove a strong customer response with 3.7 million customers joining through our non-tender offer. We ended the year with a total of 7.8 million active Nordstrom Rewards customers. Our working capital improvements contributed to the $1.6 billion in operating cash flow and $0.6 billion in free cash flow. From a merchandising perspective, we strive to offer a curated selection of the best brands. As we look for new opportunities through our vendor partnerships, we will continue to be strategic and pursue partnerships that are similar to our portfolio and maintain relevance with our customers by delivering newness. Our strategies around product differentiation include our ongoing efforts to grow limited distribution brands such as Ivy Park, J.Crew and Good American, in addition to our Nordstrom exclusive offering. In 2016, we made focused efforts to improve our productivity, particularly around our technology, supply chain and marketing. In technology, we increased the productivity of delivering features to enhance the customer experience. In supply chain, we focused on overall profitability by reducing split shipments and editing out less profitable items online. In marketing, we strengthened our capabilities around digital engagement so that we reach customers in a more efficient and cost-effective manner. Through these efforts, we made significant progress in improving operational efficiencies, reflected by moderated expense growth of 10% in these three key areas, relative to an annual average of 20% over the past five years. With customer expectations changing faster than ever, it is important that we remain focused on the customer. Moving forward, we believe our strategies give us a platform for enhanced capabilities to better serve customers and increase market share. Our obsession with our customers keeps us focused on speed, convenience and personalization. We have good momentum in place and will continue to make changes to ensure we are best serving customers and improving our business now and into the future."

class Keyworder(object):
    stop_word_additions = set()
    auto_name = "Standard"
    def this_filter(self, text, top=100):
        r = Rake()
        r.extract_keywords_from_text(text)
        phrases = r.get_ranked_phrases()
        top_phrases = phrases[0:top]
        return top_phrases

command_line = '-Xmx1024m -jar maui-standalone-1.1-SNAPSHOT.jar train -l theses100/theses80/text/ -m models/theses100_model -v none -o 2'
args = shlex.split(command_line)
# print(args)
command_line2 = '-Xmx1024m -jar maui-standalone-1.1-SNAPSHOT.jar run input_string -m models/theses100_model -v none -n top'
args2 = shlex.split(command_line2)
# print(args2)
# subprocess.popen(['-Xmx1024m', '-jar', 'maui-standalone-1.1-SNAPSHOT.jar', 'train', '-l', 'theses100/theses80/text/', '-m', 'models/theses100_model', '-v', 'none', '-o', '2'])
# subprocess.popen(['-Xmx1024m', '-jar', 'maui-standalone-1.1-SNAPSHOT.jar', 'run', input_string, '-m', 'models/theses100_model', '-v', 'none', '-n', top])

def maui_keyworder():
    # subprocess.Popen(['java', '-Xmx1024m', '-jar', 'maui-standalone-1.1-SNAPSHOT.jar', 'train', '-l', 'theses100/theses80/text/', '-m', 'models/theses100_model', '-v', 'none', '-o', '2'])
    output = subprocess.Popen(['-Xmx1024m', '-jar', 'maui-standalone-1.1-SNAPSHOT.jar', 'run', 'text_trial.txt', '-m', 'models/theses80', '-v', 'none', '-n', '10'])
    return output
print(maui_keyworder())

# def this_filter(text, top=100):
#     r = Rake()
#     r.extract_keywords_from_text(text)
#     phrases = r.get_ranked_phrases()
#     top_phrases = phrases[0:top]
#     return top_phrases

keyworderobject = Keyworder()
phrases_extracted = keyworderobject.this_filter(text_trial, top=10)

print(phrases_extracted)
