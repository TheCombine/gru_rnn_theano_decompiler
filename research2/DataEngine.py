import re

o1test = 'data/artificial/ds_O1_test_5011.txt'
o2test = 'data/artificial/ds_O2_test_4977.txt'
o3test = 'data/artificial/ds_O3_test_5159.txt'
o1train = 'data/artificial/ds_O1_train_39977.txt'
o2train = 'data/artificial/ds_O2_train_39945.txt'
o3train = 'data/artificial/ds_O3_train_39811.txt'
o1valid = 'data/artificial/ds_O1_valid_5012.txt'
o2valid = 'data/artificial/ds_O2_valid_5078.txt'
o3valid = 'data/artificial/ds_O3_valid_5030.txt'

class DataEngine(object):
    def LoadData(self, path):
        """out: SeqSeqTokenObj, SeqSeqTokenSrc, SeqStringAli"""
        #read all file contents
        file = open(path)
        contents = file.read()
        file.close()
        #split contents into obj, src, ali
        SeqFunction = contents.split('<<<sep_out_sample>>>\n')
        SeqStringObj = []
        SeqStringSrc = []
        SeqStringAli = []
        for Function in SeqFunction:
            sectors = Function.split('<<<sep_in_sample>>>\n')
            SeqStringObj.append(sectors[0])
            SeqStringSrc.append(sectors[1])
            SeqStringAli.append(sectors[2])
        SeqSeqTokenObj = self._tokenize(SeqStringObj, '%[a-z0-9]+|\$|,|\(|\)|\-|:|<|>|\+|[a-zA-Z0-9_]+') #'%[a-z0-9]+|\$|[0-9]|,|\(|\)|x|\-|:|<|>|\+|[a-zA-Z0-9_]+'
        SeqSeqTokenSrc = self._tokenize(SeqStringSrc, 'if|else|switch|case|default|break|int|float|char|double|long|for|while|do|void|goto|auto|signed|const|extern|register|unsigned|return|continue|enum|sizeof|struct|typedef|union|volatile|\+|\-|/|\*|&|\||\^|~|\?|\(|\)|\[|\]|\{|\}|<|>|=|!|,|\'|"|;|#|\\\|[a-zA-Z0-9_]+')
        return SeqSeqTokenObj, SeqSeqTokenSrc, SeqStringAli

    def _tokenize(self, SeqString, Pattern):
        """Called by LoadData"""
        SeqSeqToken = []
        for String in SeqString:
            SeqToken = re.findall(Pattern, String)
            #split everything thats not a variable name (re [a-zA-Z0-9_]+)
            for Token in SeqToken:
                if Token[0].isdigit():
                    TokenIndex = SeqToken.index(Token)
                    SeqToken = SeqToken[0:TokenIndex] + list(Token) + SeqToken[TokenIndex + 1:]
            SeqSeqToken.append(SeqToken)
        return SeqSeqToken