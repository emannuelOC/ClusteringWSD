# -*- coding: utf-8 -*-
import math

################ PREPARAÇÃO DO CORPUS ###################

# preparing text
def prepareText(rawText):
    newText = rawText.split();
    ponctuation = '.,::\'"?!-/()[]{}'
    newText = [w.strip(ponctuation) for w in newText]
    newText = [w.lower() for w in newText if w != " "]
    return newText
    

# striping away stop words
def stripStopList(corpus, threshold, probabilities):
    return [x for x in corpus if probabilities[x] < threshold]
    
    
################# MEDIDAS IMPORTANTES ####################    
    
# association measures
def simpleFeatureVector(context, corpus):
    """Returns a list of tuples, each one 
    containing a word and 1 if that words 
    is the context or 0 otherwise"""
    vector = list(set(corpus))
    for i in range(len(vector)):
        if vector[i] in context: 
            vector[i] = (vector[i], 1)
        else:
            vector[i] = (vector[i], 0)
    return vector
    
# t-test
def tTestAssociation(context, corpus, word):
    """There must be a global variable contexts"""
    vector = list(set(corpus))
    wordContexts = [ctx for ctx in contexts if word in ctx]
    
    for i in range(len(vector)):
        if vector[i] in context: 
            prob = len([c for c in wordContexts if vector[i] in c])/float(len(contexts))
            tTest = (prob - probabilities[word] * probabilities[vector[i]])/math.sqrt(probabilities[vector[i]] * probabilities[word])
            vector[i] = (vector[i], tTest)
        else:
            vector[i] = (vector[i], 0)
    return vector

# distance measures
def euclideanDistance(vector1, vector2):
    """Returns the Euclidean distance between
    SAME SIZE vectors! 
    Jurafsky&Martin pg. 28 Chapter 20 - Computational
    Lexical Semantics"""
    return math.sqrt(sum((vector1[i] - vector2[i])**2 for i in range(len(vector1))))
    
def manhattanDistance(vector1, vector2):
    """Returns the Manhattan distance between
    SAME SIZE vectors!
    Jurafsky&Martin pg. 28 Chapter 20 - Computational
    Lexical Semantics"""
    return sum([abs(vector1[i] - vector2[i]) for i in range(len(vector1))])
    
def cosineSimilarity(vector1, vector2):
    """Returns the cosine similarity between
    the two vectors"""
    sum1, sum2, sum3 = 0, 0, 0
    for i in range(len(vector1)):
        sum1 += vector1[i] * vector2[i]
        sum2 += vector1[i]**2
        sum3 += vector2[i]**2
    return sum1 / (math.sqrt(sum2) * math.sqrt(sum3))

# getting contexts of occurences of a given word in the corpus
def occurrenceContexts(word, corpus, window):
    return [corpus[i - window : i] + corpus[i + 1 : i + 1 + window] for i in range(len(corpus)) if corpus[i] == word]
    
    
# getting a context
def contextForIndex(index, corpus, windowSize):
    return corpus[index - windowSize:index] + corpus[index + 1:index + 1 + windowSize]

# creating word vector for a given context:
def wordVectorForContext(context, corpus):
    vector = [w for w in set(corpus)]
    wordVector = []
    for i in vector:        
        if i in context:
            wordVector += [1]
        else:
            wordVector += [0]
    return wordVector
    
# testando as distancias
def idealSense(word, window):
    """not ideal yet"""
    for i in corpus[30:]:
        if i == word: 
            return wordVectorForContext(occurrenceContexts(w, corpus, window)[0], corpus)

def testaDistancia(word1, word2, window, distancia):
    s1, s2 = idealSense(word1, window), idealSense(word2, window)
    return distancia(s1, s2)
    

# opening file
myFile = open("../corpora/mangaCorpus.txt", 'r')
corpus = myFile.read()
myFile.close()
corpus = prepareText(corpus)

# getting probabilities - always helpful! 
histogram = {}
for w in corpus:
    if w in histogram.keys():
        histogram[w] += 1
    else:
        histogram[w] = 1

probabilities = {}
for k in histogram.keys():
    probabilities[k] = histogram[k] / float(len(corpus))
    
# coocurrence contexts
window = input("What should be the size of the window? ")
contexts = []
for i in range(window, len(corpus) - window):
    context = contextForIndex(i, corpus, window)
    contexts += [context]

occurrences = {}
for w in set(corpus):
    occurrences[w] = occurrenceContexts(w, corpus, window)


class Ocorrencia:
    """..."""
    def __init__(self, palavra, contexto, indiceCorpus, tamanhoContexto):
        self.palavra = palavra
        self.contexto = contexto
        self.indiceCorpus = indiceCorpus
        self.tamanhoContexto = tamanhoContexto
        
    @property
    def palavra(self):
        return self.palavra
        
    @palavra.setter
    def setPalavra(self, palavra):
        self.palavra = palavra
    
    @property
    def contexto(self):
        return self.contexto
    
    @contexto.setter
    def setContexto(self, contexto):
        self.contexto = contexto
        
    @property
    def indiceCorpus(self):
        return self.indiceCorpus
        
    @indiceCorpus.setter
    def setIndiceCorpus(self, indiceCorpus):
        self.indiceCorpus = indiceCorpus
        
    @property
    def tamanhoContexto(self):
        return self.tamanhoContexto
        
    @tamanhoContexto.setter
    def setTamanhoContexto(self, tamanhoContexto):
        self.tamanhoContexto = tamanhoContexto
        
    @property
    def vectorTTest(self):
        return [x[1] for x in tTestAssociation(self.contexto, corpus, self.palavra)]
        
    @property
    def vectorSimple(self):
        return [x[1] for x in simpleFeatureVector(self.contexto, corpus)]
        
    def euclideanDistanceT(self, ocorrencia):
        return euclideanDistance(self.vectorTTest, ocorrencia.vectorTTest)
        
    def euclideanDistanceSimple(self, ocorrencia):
        return euclideanDistance(self.vectorSimple, ocorrencia.vectorSimple)
        
class Cluster:
    
    @property
    def ocorrencias(self):
        return self.ocorrencias
        
    @ocorrencias.setter
    def setOcorrencias(self, ocorrencias):
        self.ocorrencias = ocorrencias
        
    def addOcorrencia(self, ocorrencia):
        self.ocorrencias.append(ocorrencia)
        
    def removeOcorrencia(self, ocorrencia):
        self.ocorrencias.remove(ocorrencia)
        
    @property
    def centroid(self):
        centroid = [[0] for i in range(len(self.ocorrencias[-1].vectorTTest))]
        for i in range(len(centroid)):
            centroid[i] = sum([o.vectorTTest[i] for o in self.ocorrencias])
        centroid = [x/len(centroid) for x in centroid]
        return centroid
        
    def centroidSimple(self):
        centroid = [[0] for i in range(len(self.ocorrencias[-1].vectorTTest))]
        for i in range(len(centroid)):
            centroid[i] = sum([o.vectorSimple[i] for o in self.ocorrencias])
        centroid = [x/len(centroid) for x in centroid]
        return centroid
        
def ocorrencias(palavra):
    lista = []
    for i in range(len(corpus)):
        if corpus[i] == palavra:
            ctx = contextForIndex(i, corpus, window)
            indiceCorpus = i
            tamanhoContexto = window
            o = Ocorrencia(palavra, ctx, indiceCorpus, tamanhoContexto)
            lista += [o]
    return lista
                  

"""TODO LIST:
   . CREATE A DICTIONARY t_test WITH THE T-TEST ASSOCIATION VALUE
   FOR EACH TWO WORDS;
   . USE THE CLUSTERS FOR A DISAMBIGUATION PROBLEM
   """
   
