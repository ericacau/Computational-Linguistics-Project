# -*- coding: utf-8 -*-
import sys
import codecs
import math
import nltk
from nltk import bigrams
from nltk import trigrams


def annotazioneLinguistica(frasi):
    lunghezzaTOT = 0
    tokensTOT = []  
    for frase in frasi:
    #--- divido la frase in token
        tokens = nltk.word_tokenize(frase)
    #--- concateno la frase tokenizzata con le altre frasi gia' tokenizzate
        tokensTOT = tokensTOT + tokens
    #--- eseguo l'analisi morfo-sintattica
    tokensPOS = nltk.pos_tag(tokensTOT)
    #--- calcolo il numero di token totali del corpus
    lunghezzaTOT = len(tokensTOT)
    #--- restituisco la lista contenente tutti i token del corpus, il numero totale di token e l'analisi delle POS
    return tokensTOT, lunghezzaTOT, tokensPOS

def ordinaDizionario(dict):
    return sorted(dict.items(), key=lambda x: x[1], reverse=True)

def calcolaFrequenza(token):
    #--- creo una lista in cui inserire i token (escludendo la punteggiatura)
    tokenSenzaPunteggiatura = []
    #--- scorro i token
    for tok in token: 
        #--- controllo che il token non sia un segno di punteggiatura
        if tok not in {".", "''", ":", "(", ")", "-", ","}:
            #--- se rispetta la condizione, il token viene inserito nella lista tokenSenzaPunteggiatura
            tokenSenzaPunteggiatura.append(tok)
    #--- calcolo la frequenza dei token
    distribuzioneDiFrequenza = nltk.FreqDist(tokenSenzaPunteggiatura)
    #--- ordino i token in ordine decrescente di frequenza
    tokenOrdinati = distribuzioneDiFrequenza.most_common(len(distribuzioneDiFrequenza))
    return tokenOrdinati

def estraiSostantivi(POSTag):
    #--- Creo una lista che conterrà i sostantivi estratti
    sostantivi = []
    #--- Scorro gli elementi della lista di tokenPOS
    for token in POSTag:
        #se la POS è un sostantivo la inserisco nella lista sostantivi
        if token[1] in {'NN', 'NNS', 'NNP', 'NNPS'}:
            sostantivi.append(token[0])
    #--- Calcolo la distribuzione di frequenza e ordino i sostantivi in ordine decrescente di frequenza
    distribuzioneDiFrequenzaSostantivi = nltk.FreqDist(sostantivi)
    sostantiviOrdinati = distribuzioneDiFrequenzaSostantivi.most_common(len(distribuzioneDiFrequenzaSostantivi))
    return sostantiviOrdinati

def estraiAggettivi(POSTag):
    #--- Creo una lista che conterrà gli aggettivi estratti
    aggettivi = []
    #--- Scorro gli elementi della lista di tokenPOS
    for token in POSTag: 
        #--- se la POS è un aggettivo la inserisco nella lista aggettivi
        if token[1] in {'JJ', 'JJR', 'JJS'}:
            aggettivi.append(token[0])
    #--- Calcolo la distribuzione di frequenza e ordino i sostantivi in ordine decrescente di frequenza
    distribuzioneDiFrequenzaAggettivi = nltk.FreqDist(aggettivi)
    aggettiviOrdinati = distribuzioneDiFrequenzaAggettivi.most_common(len(distribuzioneDiFrequenzaAggettivi))
    return aggettiviOrdinati

def frequenzaBigrammi(tokenPOS):
    #--- creo una lista in cui inserirò i bigrammi senza punteggiatura, articoli (DT) e congiunzioni (CC)
    bigrammiCorretti = []
    #calcolo i bigrammi totali del corpus
    bigrammiTOT = (list(nltk.bigrams(tokenPOS)))
    #scorro i bigrammi totali
    for bigramma in bigrammiTOT:
        #controllo che la POS non sia un segno di punteggiatura, un articolo o una congiunzione
        if ((bigramma[0][1] not in {"CC", "DT", ".", "''", ":", "(", ")", "-", ","}) and (bigramma[1][1] not in {"CC", "DT", ".", ",", "''", ":", "(", ")", "-"})):
           #inserisco i bigrammi che rispettano la condizione nella lista bigrammiCorretti
            bigrammiCorretti.append(bigramma)
    #calcolo la frequenza dei bigrammi
    listaBigrammiConFrequenza = nltk.FreqDist(bigrammiCorretti)
    bigrammiOrdinati = listaBigrammiConFrequenza.most_common(20)
    return bigrammiOrdinati, bigrammiTOT

def calcolaPOS(POSTag):
    #--- Creo una lista che conterrà le pos estratte
    pos = []
    #--- scorro le POStag
    for elem in POSTag:
        #inserisco nella lista solo le POS tag estratte
        pos.append(elem[1])
    #--- calcolo la frequenza delle pos
    posFrequentiTOT = nltk.FreqDist(pos)
    #--- ordino in ordine decrescente di frequenza e estraggo le prime 10 più frequenti
    posFrequentiDieci = posFrequentiTOT.most_common(10) 
    return posFrequentiDieci, pos

def bigrammiPOSFrequenti (POSEstratte):
    #--- dalla lista contenente la sequenza delle pos estratte dal corpus, estraggo i bigrammi
    bigrammiPOS = (list(nltk.bigrams(POSEstratte)))
    #--- calcolo le frequenze dei bigrammi
    frequenzaBigrammiPOS = nltk.FreqDist(bigrammiPOS)
    #--- ordino in ordine decrescente di frequenza e estraggo i 10 bigrammi più frequenti
    frequenzaBigrammiPOS10 = frequenzaBigrammiPOS.most_common(10)
    return frequenzaBigrammiPOS10

def trigrammiPOSFrequenti(posEstratte):
    #--- estraggo i trigrammi dalla sequenza di POS estratte
    trigrammi = (list(nltk.trigrams(posEstratte)))
    #--- calcolo la frequenza dei trigrammi e estraggo i 10 più frequenti
    frequenzaTrigrammiPOS = nltk.FreqDist(trigrammi)
    frequenzaTrigrammiPOS10 = frequenzaTrigrammiPOS.most_common(10)
    return frequenzaTrigrammiPOS10

def bigrammiAggettivoSostantivo(bigrammi, tokenPOS):
    #--- creo la lista che conterrà i bigrammi composti da AGGETTIVO e SOSTANTIVO
    bigrammiTokenMaggioreDue = []
    #--- scorro i bigrammi
    for bigramma in bigrammi:
        #--- creo due variabili in cui conservare il primo e il secondo token di cui si compone il bigramma 
        primaParolaBigramma = bigramma[0]
        secondaParolaBigramma = bigramma[1]
        #--- calcolo la frequenza della prima parola di cui si compone il bigramma
        freqToken1 = tokenPOS.count(primaParolaBigramma)
        #--- calcolo la frequenza della seconda parola di cui si compone il bigramma
        freqToken2 = tokenPOS.count(secondaParolaBigramma)
        #--- per ogni bigramma controllo se la prima parola è un aggettivo e la seconda è un nome
        if (bigramma[0][1]) in {'JJ', 'JJR', 'JJS'} and bigramma[1][1] in {'NN', 'NNS', 'NNP', 'NNPS'}:
            #--- controllo che la frequenza della prima e della seconda parola sia superiore a 2
            if freqToken1 > 2 and freqToken2 > 2:
                #--- se la condizione viene rispettata inserisco il bigramma nella lista
                bigrammiTokenMaggioreDue.append(bigramma)
    #--- calcolo la frequenza dei bigrammi 
    frequenzaBigrammi = nltk.FreqDist(bigrammiTokenMaggioreDue)
    #--- restituisco la lista dei bigrammi aggettivo-sostantivo e la frequenza di tali bigrammi
    return bigrammiTokenMaggioreDue, frequenzaBigrammi

def calcolaProbabilitaCongiunta(bigrammiAggSost, token, numeroToken):
    #--- creo il dizionario che conterra' le coppie "bigramma-probabilita' congiunta" 
    dizionarioProbabilita = {}
    #--- conservo in una variabile bigrammi diversi nel corpus
    bigrammiDiversi = set(bigrammiAggSost)
    for bigramma in bigrammiDiversi:
        #--- calcolo la frequenza assoluta del bigramma
        frequenzaBigramma = bigrammiAggSost.count(bigramma)
        #--- calcolo la frequenza all'interno del corpus della prima parola di cui si compone il bigramma
        frequenzaAssolutaA = token.count(bigramma[0][0])
        #--- calcolo la probabilita' della prima parola di cui si compone il bigramma come |Freq.A|/|C|
        probabilitaA = float(frequenzaAssolutaA)/float(numeroToken)
        #--- calcolo la probabilita' condizionata del bigramma come freq. del bigramma/freq. assoluta di A
        probabilitaCondizionata = float(frequenzaBigramma)/float(frequenzaAssolutaA)
        #--- utilizzo la probabilita' condizionata per il calcolo della probabilita' congiunta
        probabilitaCongiunta = (float(probabilitaA))*(float(probabilitaCondizionata))
        #--- associo la probabilita' congiunta al bigramma
        dizionarioProbabilita[bigramma] = probabilitaCongiunta
    #--- invoco la funzione ordinaDizionario per ordinare il dizionario 
    dizionarioProbabilitaBigrammaOrdinato = ordinaDizionario(dizionarioProbabilita)
    return dizionarioProbabilitaBigrammaOrdinato

def calcolaLocalMutualInformation(bigrammiAggSost, numeroToken, token):
    #--- Inizializzo il dizionario che conterra' le coppie "bigramma-LMI"
    localMutual = {}
    #--- conservo in una variabile bigrammi diversi nel corpus
    bigrammiDiversi = set(bigrammiAggSost)
    #--- scorro i bigrammi
    for bigramma in bigrammiDiversi:
        #--- calcolo la frequenza del bigramma
        frequenzaBigramma = bigrammiAggSost.count(bigramma)
        #--- calcolo la probabilita' del bigramma
        probabilitaBigramma = float(frequenzaBigramma)/float(numeroToken)
        #--- calcolo la probabilita' della prima parola di cui si compone il bigramma come freq.token/|C|
        probabilitaU = float(token.count(bigramma[0][0]))/float(numeroToken)
        #--- calcolo la probabilita' della seconda parola di cui si compone il bigramma come freq.token/|C|
        probabilitaV = float(token.count(bigramma[1][0]))/float(numeroToken)
        #--- calcolo la LMI
        probabMI = float(probabilitaBigramma)/float(probabilitaU*probabilitaV)
        localMutualInformation = frequenzaBigramma * math.log(probabMI,2)
        #--- inserisco nel dizionario il bigramma e associato ad esso, la LMI
        localMutual[bigramma] = localMutualInformation
    #--- invoco la funzione ordinaDizionario per ordinare il dizionario contenente la LMI
    LMIOrdinata = ordinaDizionario(localMutual)
    return LMIOrdinata

def frasiMarkov (frasi, token, corpus, frequenza):
    #--- definisco una lista che conterrà le frasi estratte
    fraseMarkov = []
    #--- scorro le frasi
    for frase in frasi:
        #--- tokenizzo la singola frase
        tokenFrase = nltk.word_tokenize(frase)
        #--- calcolo da quanti token è composta la frase
        lunghezzaFrase = len(tokenFrase)
        #--- per ogni frase inizializzo a 0 una variabile
        i = 0
        #--- per ogni frase controllo che sia lunga minimo 6 token e massimo 8
        if lunghezzaFrase >= 6 and lunghezzaFrase <= 8:
            #--- controllo che ogni token della frase presa in analisi abbia una frequenza maggiore di 2 
            for tok in tokenFrase:
                if frequenza[tok] > 2:
                    #--- conto quanti token nella frase hanno una frequenza maggiore di 2 nel corpus
                    i += 1
            #--- se il numero di token nella frase con freq > 2 corrisponde al numero di token totali della frase
            if i == lunghezzaFrase:
                #--- inserisco la frase nella lista
                fraseMarkov.append(frase)
    return fraseMarkov


#--- MARKOV 0: P(A1,...,An) = P(A)*P(A2)*...*P(An)
def calcolaMarkov0(frasiMarkov, lunghezzaCorpus, frequenza):
    #--- creo un dizionario in cui inserirò le frasi e la probabilita' associata ad ognuna di esse
    dizionarioMarkov = {}
    #--- scorro le frasi tra quelle lunghe al massimo 8 token e con i token con frequenza > 2
    for frase in frasiMarkov:
        #--- tokenizzo la frase
        tokenFrase = nltk.word_tokenize(frase)
        #---inizializzo una variabile per calcolare la probabilita' di quella frase con una catena di Markov di ordine 0 
        probabilita = 1.0
        #--- scorro i token della frase in analisi
        for tok in tokenFrase:
            #---calcolo la probabilita' del singolo token come frequenza/|C|
            probabilitaToken = float(frequenza[tok])/float(lunghezzaCorpus)
            #---aggiorno la probabilita'
            probabilita = probabilita*probabilitaToken
        #---associo la probabilita' della frase alla frase
        dizionarioMarkov[frase] = probabilita
    #---invoco la funzione ordinaDizionario per ordinare in ordine decrescente il dizionario
    dizionarioMarkovOrdinato = ordinaDizionario(dizionarioMarkov)
    #---restituisco solo la frase con maggiore probabilita'
    return dizionarioMarkovOrdinato

#--- MARKOV 1: P(A1,...,An) = P(A)*(A2|A1)*P(A3|A2)*...*P(An|An-1)
def calcolaMarkov1(frasiMarkov, lunghezzaCorpus, token, frequenza):
    dizionarioMarkov = {}
    #--- calcolo i bigrammi del testo (inclusa la punteggiatura)
    bigrammi = list(nltk.bigrams(token))
    #--- calcolo la distribuzione di frequenza dei bigrammi
    frequenzaBigr = nltk.FreqDist(bigrammi)
    #scorro le varie frasi
    for frase in frasiMarkov:
        #--- divido la frase in token
        tokenFrase = nltk.word_tokenize(frase)
        #--- calcolo i bigrammi della frase
        bigrammiFrase = list(nltk.bigrams(tokenFrase))
        #--- salvo in una variabile la prima parola della frase
        primoToken = tokenFrase[0]
        #--- calcolo la probabilita' della prima parola
        probabilita = float(frequenza[primoToken])/float(lunghezzaCorpus)
        #--- scorro i bigrammi presenti nella frase
        for bigramma in bigrammiFrase:
            #recupero la frequenza del bigramma e la salvo in una variabile
            frequenzaBigramma = frequenzaBigr[bigramma]
            #--- calcolo la probabilita' del bigramma
            probabilitaBigramma = float(frequenzaBigramma)/float(frequenza[bigramma[0]])
            #--- aggiorno la variabile contenente la probabilita'
            probabilita = float(probabilitaBigramma)*float(probabilita)
        #--- dopo aver calcolato la probabilita', inserisco la coppia "bigramma - probabilita'" nel dizionario
        dizionarioMarkov[frase] = probabilita
    #--- invoco la funzione per ordinare il dizionario
    dizionarioMarkovOrdinato = ordinaDizionario(dizionarioMarkov)
    return dizionarioMarkovOrdinato


def main(testo1, testo2):
    fileInput1 = codecs.open(testo1, "r", "utf-8")
    fileInput2 = codecs.open(testo2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#--- DIVIDO IL TESTO IN FRASI
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
#--- CREO DUE VARIABILI CON IL NUMERO TOTALE DELLE FRASI NEI DUE CORPORA
    numeroFrasi1 = len(frasi1)
    numeroFrasi2 = len(frasi2)

#--------------------INVOCAZIONE DELLE FUNZIONI-----------------

#--1.1) 20 token piu' frequenti escludendo la punteggiatura
    token1, numeroToken1, tokenPOS1 = annotazioneLinguistica(frasi1)
    token2, numeroToken2, tokenPOS2 = annotazioneLinguistica(frasi2)
    frequenzaTokenTOT1 = nltk.FreqDist(token1)
    frequenzaTokenTOT2 = nltk.FreqDist(token2)
    #---calcolo la frequenza dei token
    tokenFrequenti_1 = calcolaFrequenza(token1)
    tokenFrequenti_2 = calcolaFrequenza(token2)

#--1.2) 20 sostantivi piu'frequenti
    #---estraggo gli sostantivi
    sostantivi_1 = estraiSostantivi(tokenPOS1)
    sostantivi_2 = estraiSostantivi(tokenPOS2)
    
#--1.3) 20 aggettivi piu' frequenti
    aggettivi_1 = estraiAggettivi(tokenPOS1)
    aggettivi_2 = estraiAggettivi(tokenPOS2)

#--1.4) 20 bigrammi piu' frequenti escludendo punteggiatura, articoli e congiunzioni
    frequenzaVentiBigrammi_1, bigrammiTOT_1 = frequenzaBigrammi(tokenPOS1)
    frequenzaVentiBigrammi_2, bigrammiTOT_2 = frequenzaBigrammi(tokenPOS2)

#--1.5) 10 POS piu' frequenti
    posFrequenti_1, posEstratte_1 = calcolaPOS(tokenPOS1)
    posFrequenti_2, posEstratte_2 = calcolaPOS(tokenPOS2)
#--1.6) 10 bigrammi di POS piu' frequenti
    bigrammiPOSFrequenti_1 = bigrammiPOSFrequenti(posEstratte_1)
    bigrammiPOSFrequenti_2 = bigrammiPOSFrequenti(posEstratte_2)
#--1.7) 10 trigrammi di POS piu' frequenti
    trigrammiPOSFrequenti_1 = trigrammiPOSFrequenti(posEstratte_1)
    trigrammiPOSFrequenti_2 = trigrammiPOSFrequenti(posEstratte_2)

#--2) 20 bigrammi composti da Aggettivo e Sostantivo dove ogni token deve avere frequenza > 2
    listaBigrammiAggSost_1, frequenzaBigrammiAggSost_1 = bigrammiAggettivoSostantivo(bigrammiTOT_1, tokenPOS1)
    listaBigrammiAggSost_2, frequenzaBigrammiAggSost_2 = bigrammiAggettivoSostantivo(bigrammiTOT_2, tokenPOS2)
#--2.1)
#--2.2)
    probabilitaCongiunta_1 = calcolaProbabilitaCongiunta(listaBigrammiAggSost_1, token1, numeroToken1)
    probabilitaCongiunta_2 = calcolaProbabilitaCongiunta(listaBigrammiAggSost_2, token2, numeroToken2)
#--2.3) con forza associativa massima, calcolata in termini di LMI, indicando anche il relativo valore.
    localMutualInformation_1 = calcolaLocalMutualInformation(listaBigrammiAggSost_1, numeroToken1, token1)
    localMutualInformation_2 = calcolaLocalMutualInformation(listaBigrammiAggSost_2, numeroToken2, token2)

#--3)MODELLI MARKOVIANI
    #---estraggo le frasi
    frasiMarkov0_1 = frasiMarkov(frasi1, token1, numeroToken1,frequenzaTokenTOT1)
    frasiMarkov0_2 = frasiMarkov(frasi2, token2, numeroToken2,frequenzaTokenTOT2)

    #---Estraggo la frase piu' probabile secondo un modello markoviano di ordine 0
    fraseMarkov0_1 = calcolaMarkov0(frasiMarkov0_1, numeroToken1, frequenzaTokenTOT1)
    fraseMarkov0_2 = calcolaMarkov0(frasiMarkov0_2, numeroToken2, frequenzaTokenTOT2)
    maxFraseMarkov0_1 = fraseMarkov0_1[0]
    maxFraseMarkov0_2 = fraseMarkov0_2[0]

    #---Estraggo la frase piu' probabile secondo un modello markoviano di ordine 1
    
    fraseMarkov1_1 = calcolaMarkov1(frasiMarkov0_1, numeroFrasi1, token1, frequenzaTokenTOT1)
    fraseMarkov1_2 = calcolaMarkov1(frasiMarkov0_2, numeroFrasi2, token2, frequenzaTokenTOT2)
    maxFraseMarkov1_1 = fraseMarkov1_1[0]
    maxFraseMarkov1_2 = fraseMarkov1_2[0]


#-----------------OUTPUT-----------------
    print "\nSecondo programma per il progetto di Linguistica Computazionale (A.A 2018/2019)\n"
#--- 20 TOKEN PIU' FREQUENTI (SENZA CONSIDERARE LA PUNTEGGIATURA)
    print "Estraggo dai due corpora:\n"
    print"- I 20 TOKEN PIU' FREQUENTI"
    print "Corpus:", testo1
    for token in tokenFrequenti_1[:20]:
            print token[0].encode('utf-8'), "occorre\t", token[1], "volte"
    print "\nCorpus:", testo2
    for token in tokenFrequenti_2[:20]:
            print token[0].encode('utf-8'), "occorre\t", token[1], "volte"
    print

#--- 20 SOSTANTIVI PIU' FREQUENTI
    print "- I 20 SOSTANTIVI PIU' FREQUENTI"
    print "Corpus:", testo1
    for token in sostantivi_1[:20]:
            print token[0].encode('utf-8'), "occorre\t", token[1], "volte"
    print "\nCorpus:", testo2
    for token in sostantivi_2[:20]:
            print token[0].encode('utf-8'), "occorre\t", token[1], "volte"
    print

#--- 20 AGGETTIVI PIU' FREQUENTI
    print "- I 20 AGGETTIVI PIU' FREQUENTI"
    print "Corpus:", testo1
    for token in aggettivi_1[:20]:
            print token[0].encode('utf-8'), "occorre\t", token[1], "volte"
    print "\nCorpus:", testo2
    for token in aggettivi_2[:20]:
        print token[0].encode('utf-8'), "occorre\t", token[1], "volte"
    print
    
#--- 20 BIGRAMMI DI TOKEN PIU' FREQUENTI (SENZA CONSIDERARE PUNTEGGIATURA, ARTICOLI, CONGIUNZIONI)
    print "- I 20 BIGRAMMI PIU' FREQUENTI SENZA PUNTEGGIATURA, ARTICOLI E CONGIUNZIONI"
    print "Corpus:", testo1
    for bigramma in frequenzaVentiBigrammi_1:
        print bigramma[0][0][0], bigramma[0][1][0], "\toccorre\t", bigramma[1], "volte."
    print "\nCorpus:", testo2
    for bigramma in frequenzaVentiBigrammi_2:
        print bigramma[0][0][0].encode('utf-8'), bigramma[0][1][0].encode('utf-8'), "\toccorre\t", bigramma[1], "volte."
    print    

#--- 10 POS PIU' FREQUENTI
    print "- LE 10 POS (PART-OF-SPEECH) PIU' FREQUENTI"
    print "Corpus:", testo1
    for pos in posFrequenti_1:
        print pos[0], "\toccorre\t", pos[1], "volte"
    print "\nCorpus:", testo2
    for pos in posFrequenti_2:
        print pos[0], "\toccorre\t", pos[1], "volte"
    print

#--- 10 BIGRAMMI DI POS PIU' FREQUENTI
    print "- I 10 BIGRAMMI DI POS PIU' FREQUENTI"
    print "Corpus:", testo1
    for bigrammaPOS in bigrammiPOSFrequenti_1:
        print "Il bigramma di POS", bigrammaPOS[0], "occorre\t", bigrammaPOS[1], "volte"
    print "\nCorpus:", testo2
    for bigrammaPOS in bigrammiPOSFrequenti_2:
        print "Il bigramma di POS", bigrammaPOS[0], "occorre\t", bigrammaPOS[1], "volte"
    print

#--- 10 TRIGRAMMI DI POS PIU' FREQUENTI
    print "- I 10 TRIGRAMMI DI POS PIU' FREQUENTI"
    print "Corpus:", testo1
    for trigrammaPOS in trigrammiPOSFrequenti_1:
        print "Il trigramma di POS", trigrammaPOS[0], "occorre\t", trigrammaPOS[1], "volte"
    print "\nCorpus:", testo2
    for trigrammaPOS in trigrammiPOSFrequenti_2:
        print "Il trigramma di POS", trigrammaPOS[0], "occorre\t", trigrammaPOS[1], "volte"
    print
    
#--- BIGRAMMI COMPOSTI DA AGGETTIVO E SOSTANTIVO CON FREQUENZA TOKEN > 2
#------------ FREQUENZA MASSIMA
    print "- I 20 BIGRAMMI FORMATI DA AGGETTIVO E SOSTANTIVO, DOVE OGNI TOKEN HA FREQUENZA > 2:"
    print " - Piu' frequenti nel corpus", testo1
    frequenzaBigrammiAggettivoSostantivo_1 = frequenzaBigrammiAggSost_1.most_common(20)
    for bigramma in frequenzaBigrammiAggettivoSostantivo_1:
        freqToken1_1 = token1.count(bigramma[0][0][0])
        freqToken2_1 = token1. count(bigramma[0][1][0])
        print bigramma[0][0][0].encode('utf-8'), bigramma[0][1][0].encode('utf-8'),"\toccorre\t", bigramma[1], "volte."
        print ">>>", bigramma[0][0][0].encode('utf-8'), "ricorre nel corpus", freqToken1_1, "volte"
        print ">>>", bigramma[0][1][0].encode('utf-8'), "ricorre nel corpus", freqToken2_1, "volte\n"
    print
    print " - Piu' frequenti nel corpus", testo2
    frequenzaBigrammiAggettivoSostantivo_2 = frequenzaBigrammiAggSost_2.most_common(20)
    for bigramma in frequenzaBigrammiAggettivoSostantivo_2:
        freqToken1_2 = token2.count(bigramma[0][0][0])
        freqToken2_2 = token2. count(bigramma[0][1][0])
        print bigramma[0][0][0].encode('utf-8'), bigramma[0][1][0].encode('utf-8'),"\toccorre\t", bigramma[1], "volte."
        print ">>>", bigramma[0][0][0].encode('utf-8'), "ricorre nel corpus", freqToken1_2, "volte"
        print ">>>", bigramma[0][1][0].encode('utf-8'), "ricorre nel corpus", freqToken2_2, "volte\n"

#------------ PROBABILITA CONGIUNTA MASSIMA
    print "\n - Con la probabilita' congiunta massima nel corpus", testo1, "\n"
    for elem in probabilitaCongiunta_1[:20]:
        print elem[0][0][0].encode('utf-8'), elem[0][1][0].encode('utf-8'), "\t---probabilita congiunta:     ", elem[1]
    print
    print " - Con la probabilita' congiunta massima nel corpus", testo2, "\n"
    for elem in probabilitaCongiunta_2[:20]:
        print elem[0][0][0].encode('utf-8'), elem[0][1][0].encode('utf-8'), "\t---probabilita' congiunta:     ", elem[1]

#------------ LOCAL MUTUAL INFORMATION
    print "\n - Con maggiore forza associativa (in termini di LMI) nel corpus", testo1
    for elem in localMutualInformation_1[:20]:
        print elem[0][0][0].encode('utf-8'), elem[0][1][0].encode('utf-8'), " LMI:\t", elem[1]
    print
    print " - Con maggiore forza associativa (in termini di LMI)", testo2
    for elem in localMutualInformation_2[:20]:
        print elem[0][0][0].encode('utf-8'), elem[0][1][0].encode('utf-8'), " LMI:\t", elem[1]
    print

#FRASI CON PROBABILITA' MASSIMA
    print "- LE DUE FRASI CON LA PROBABILITA' PIU' ALTA"
    print "- Probabilita' piu' alta secondo un modello markoviano di ordine 0 nel corpus", testo1
    print maxFraseMarkov0_1[0].encode('utf-8'), "\tcon probabilita':     ", maxFraseMarkov0_1[1], "\n"
    print "- Probabilita' piu' alta secondo un modello markoviano di ordine 0 nel corpus", testo2
    print maxFraseMarkov0_2[0].encode('utf-8'), "\tcon probabilita':     ", maxFraseMarkov0_2[1], "\n\n"
    print "- Probabilita' piu' alta secondo un modello markoviano di ordine 1 nel corpus", testo1
    print maxFraseMarkov1_1[0].encode('utf-8'), "\tcon probabilita':     ", maxFraseMarkov1_1[1], "\n"
    print "- Probabilita' piu' alta secondo un modello markoviano di ordine 1 nel corpus", testo2
    print maxFraseMarkov1_2[0].encode('utf-8'), "\tcon probabilita':     ", maxFraseMarkov1_2[1], "\n\n"
    

main(sys.argv[1], sys.argv[2])

