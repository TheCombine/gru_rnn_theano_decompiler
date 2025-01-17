from time import strftime, localtime
from os import listdir
from numpy import *

def LoadVocabulary():
    fileNames = listdir('savesVocabulary/')
    if len(fileNames) == 0:
        return None
    recentFileName = None
    recentCreationDate = -1
    for fileName in fileNames:
        try:
            creationDate = int(''.join([c for c in fileName if c.isdigit()]))
        except:
            continue
        if creationDate > recentCreationDate:
            recentFileName = fileName
            recentCreationDate = creationDate
    if recentFileName == None:
        return None
    print 'Recent vocabulary save:', recentFileName
    file = open('savesVocabulary/%s' % recentFileName, 'r')
    contents = file.read()
    file.close()
    vocabulary = Vocabulary(False)
    StrTokenObj, StrTokenSrc = contents.split('<<<VOCABULARY SEPARATOR>>>')

    rows = filter(None, StrTokenObj.split('\n'))
    for row in rows:
        i, token = row.split()
        if not vocabulary.token_to_index_obj.has_key(token):
            i = int(i)
            vocabulary.index_to_token_obj.insert(i, token)
            vocabulary.token_to_index_obj[token] = i

    rows = filter(None, StrTokenSrc.split('\n'))
    for row in rows:
        i, token = row.split()
        if not vocabulary.token_to_index_src.has_key(token):
            i = int(i)
            vocabulary.index_to_token_src.insert(i, token)
            vocabulary.token_to_index_src[token] = i

    return vocabulary

class Vocabulary(object):
    def SaveVocabulary(self):
        file = open('savesVocabulary/%s.txt' % strftime('%Y-%m-%d %H;%M;%S', localtime()), 'w')
        for i in range(len(self.index_to_token_obj)):
            print>>file, i, self.index_to_token_obj[i]
        print>>file, '<<<VOCABULARY SEPARATOR>>>'
        for i in range(len(self.index_to_token_src)):
            print>>file, i, self.index_to_token_src[i]
        file.close()

    def __init__(self, insertStartAndEndToken=True):
        self.token_to_index_obj = {}
        self.index_to_token_obj = []
        self.token_to_index_src = {}
        self.index_to_token_src = []

        if insertStartAndEndToken == True:
            special_tokens = ['<s>', '<e>']
            for i in xrange(len(special_tokens)):
                self.token_to_index_obj[special_tokens[i]] = i
                self.index_to_token_obj.insert(i, special_tokens[i])
                self.token_to_index_src[special_tokens[i]] = i
                self.index_to_token_src.insert(i, special_tokens[i])

    def FeedObj(self, SeqSeqTokenObj):
        """Adds unseen tokens to obj vocabulary"""
        self._feed(SeqSeqTokenObj, self.token_to_index_obj, self.index_to_token_obj)

    def FeedSrc(self, SeqSeqTokenSrc):
        """Adds unseen tokens to src vocabulary"""
        self._feed(SeqSeqTokenSrc, self.token_to_index_src, self.index_to_token_src)

    def _feed(self, SeqSeqToken, token_to_index, index_to_token):
        """Called by FeedObj and FeedSrc"""
        for SeqToken in SeqSeqToken:
            for Token in SeqToken:
                if not token_to_index.has_key(Token):
                    i = len(index_to_token)
                    token_to_index[Token] = i
                    index_to_token.insert(i, Token)

    def SeqSeqTokenObjToIndices(self, SeqSeqTokenObj):
        """out: SeqSeqIndexObj"""
        return self._toIndices(SeqSeqTokenObj, self.token_to_index_obj)

    def SeqSeqTokenSrcToIndices(self, SeqSeqTokenSrc):
        """out: SeqSeqIndexSrc"""
        return self._toIndices(SeqSeqTokenSrc, self.token_to_index_src)

    def _toIndices(self, SeqSeqToken, token_to_index):
        """Called by SeqSeqToken(Obj/Src)ToIndices"""
        SeqSeqIndex = []
        for SeqToken in SeqSeqToken:
            SeqIndex = []
            for Token in SeqToken:
                SeqIndex.append(token_to_index[Token])
            #SeqIndex.append(0)
            SeqSeqIndex.append(SeqIndex)
        return SeqSeqIndex

    def SeqIndexObjToTokens(self, SeqIndexObj):
        """out: SeqTokenObj"""
        return self._toTokens(SeqIndexObj, self.index_to_token_obj)

    def SeqIndexSrcToTokens(self, SeqIndexSrc):
        """out: SeqTokenSrc"""
        return self._toTokens(SeqIndexSrc, self.index_to_token_src)

    def _toTokens(self, SeqIndex, index_to_token):
        """Called by SeqIndex(Obj/Src)ToTokens"""
        SeqToken = []
        for Index in SeqIndex:
            if index_to_token[Index] == '<e>':
                break
            SeqToken.append(index_to_token[Index])
        return SeqToken

    def CloneSeqSeqX(self, SeqSeqX):
        """out: clone of SeqSeqX"""
        SeqSeqXClone = []
        for SeqX in SeqSeqX:
            SeqXClone = []
            for X in SeqX:
                SeqXClone.append(X)
            SeqSeqXClone.append(SeqXClone)
        return SeqSeqXClone

    def ConcatSeqTokensX(self, SeqTokenX):
        String = ''
        for Token in SeqTokenX:
            if Token == '{' or Token == '}':
                String += '\n'
            String += Token + ' '
            if Token == '{' or Token == '}':
                String += '\n'
        return String

    def PrintPrediction(self, SeqTokenTruth, SeqTokenPred):
        file = open('predictions/prediction.txt', 'a')
        StringTruth = self.ConcatSeqTokensX(SeqTokenTruth)
        StringPred = self.ConcatSeqTokensX(SeqTokenPred)
        print>>file, 'Expected:'
        print>>file, StringTruth
        print>>file, 'Prediction:'
        print>>file, StringPred
        print>>file, "\r\n\r\n"
        file.close()
