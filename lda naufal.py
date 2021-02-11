from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import pyLDAvis
import pyLDAvis.gensim

from io import StringIO

def pdf_to_text(pdfname):

    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, laparams=laparams, codec=codec)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open('tes.pdf', 'rb')
    for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text


handbook_string = pdf_to_text('tes.pdf')
clean_chars = [",", ".", "'", ";", "\n"]

def clean_text(text_string, special_characters):
    cleaned_string = text_string
    cleaned_string = cleaned_string.lower()
    for string in special_characters:
        cleaned_string = cleaned_string.replace(string, "")
    cleaned_string = cleaned_string.lower()
    return(cleaned_string)

def tokenize(text_string, special_characters):
    cleaned_handbook = clean_text(handbook_string, clean_chars)
    handbook_tokens = cleaned_handbook.split(" ")
    return(handbook_tokens)

tokenized_handbook = tokenize(handbook_string, clean_chars)



# remove stop words
from stop_words import get_stop_words
# create bahasa indonesia stop words list
en_stop = get_stop_words('indonesian')

# remove stop words from tokens
stopped_tokenized_handbook = [i for i in tokenized_handbook if not i in en_stop]


# #### Stemming kata-kata indonesia
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# stem token
stemmed_stopped_tokenized_handbook = [stemmer.stem(i) for i 
                                      in stopped_tokenized_handbook]
texts = stemmed_stopped_tokenized_handbook


# from collections import Counter
# import operator
# from pprint import pprint
# 
# counts = dict(Counter(texts))
# sorted_counts =sorted(counts.items(), key=operator.itemgetter(1),reverse=True)
# pprint(sorted_counts[:30])

# #### Constructing a document-term matrix


from gensim import corpora
import gensim

dictionary = corpora.Dictionary([texts])



# convert the dictionary into a bag-of-words
corpus = [dictionary.doc2bow([text]) for text in texts]


# #### Applying the LDA model


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, 
                                           id2word = dictionary, passes=50)


# #### Examining the results


print(ldamodel.print_topics(num_topics=4, num_words=10))


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
vis
pyLDAvis.show(vis)
pyLDAvis.save_html(vis, 'hasil.html')


