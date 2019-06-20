# -*- coding: utf-8 -*-

import sys
import codecs
import nltk

def annotazioneLinguistica(frasi):
    lunghezzaTOT = 0
    tokensTOT = []  
    for frase in frasi:
    #--- divido la frase in token
        tokens = nltk.word_tokenize(frase)
    #--- concateno la frase tokenizzata con le altre frasi già tokenizzate
        tokensTOT = tokensTOT + tokens
    #--- eseguo l'analisi morfo-sintattica
    tokensPOS = nltk.pos_tag(tokensTOT)
    #--- calcolo il numero di token totali del corpus
    lunghezzaTOT = len(tokensTOT)
    #--- restituisco la lista contenente tutti i token del corpus, il numero totale di token e il risultato del POS tag
    return tokensTOT, lunghezzaTOT, tokensPOS

def calcolaLunghezzaMediaFrasi(frasi, token):
    #--- calcolo la media come n° di token nel corpus / numero totale delle frasi
    lunghezzaMediaFrasi = float(token)/float(frasi)
    return lunghezzaMediaFrasi

def calcolaLunghezzaMediaParole(token, numeroToken):
    caratteri = 0
    #--- calcolo il numero di caratteri totali
    for tok in token:
        caratteri = caratteri + len(tok)
    #--- calcolo il numero totale di caratteri come numero totale di caratteri / numero totale di token
    lunghezzaMediaParole = float(caratteri)/float(numeroToken)
    return lunghezzaMediaParole

def calcolaVocabolarioTTR (token, numeroToken):
    #--- Creo un contatore per scorrere token
    n = 1000
    #--- Creo due liste che conterranno la lunghezza del vocabolario e il valore della TTR calcolata su porzioni incrementali
    vocabolario = []
    corpus_TTR = []
    #--- Fino a quando n non supera il numero di token, calcolo il vocabolario e la TTR
    while n < numeroToken:
        #-- Calcolo il vocabolario sui primi N token
        tokenVocabolario = set(token[:n])
        lunghezzaVocabolario = len(tokenVocabolario)
        #--- Calcolo la TTR dei primi N token come rapporto tra vocabolario e lunghezza del corpus
        typeTokenRatio = float(lunghezzaVocabolario)/float(n)
        #--- Inserisco i valori ottenuti nella lista vocabolario e nella lista corpus_TTR
        vocabolario.append(lunghezzaVocabolario)
        corpus_TTR.append(typeTokenRatio)
        #--- Aggiorno il contatore
        n += 1000
    return vocabolario, corpus_TTR

def classiDiFrequenza(token):
    #--- Definisco tre variabili per contare il numero di parole con f = 3, f = 6, f = 9
    classeDiFrequenza3 = 0
    classeDiFrequenza6 = 0
    classeDiFrequenza9 = 0
    #--- Definisco una variabile che contiene i primi 5000 token
    primi5000Token = token[:5000]
    #--- Calcolo la distribuzione di frequenza dei primi 5000 token
    distribuzioneDiFrequenzaToken = nltk.FreqDist(primi5000Token)
    #--- Riordino i token in base alla loro frequenza
    tokenOrdinati = distribuzioneDiFrequenzaToken.most_common(len(distribuzioneDiFrequenzaToken))
    #--- Scorro i token e calcolo le classi di frequenza
    for tok in tokenOrdinati:
        if tok[1] == 3: 
            classeDiFrequenza3 += 1
        if tok[1] == 6:
            classeDiFrequenza6 += 1
        if tok[1] == 9:
            classeDiFrequenza9 += 1
    return classeDiFrequenza3, classeDiFrequenza6, classeDiFrequenza9

def numeroMedioSostAggVbAvv (frasi, tokenPOS):
    #--- Inizializzo tre variabili per calcolare il n° di nomi, aggettivi, verbi
    numeroNomi = 0
    numeroAggettivi = 0
    numeroVerbi = 0
    #--- Scorro i singoli token del corpus e controllo la Part Of Speech
    for pos in tokenPOS:
        #--- Se la POS è un Nome, un Aggettivo o un Verbo, viene incrementato il contatore appropriato
        if pos[1] in {'NN','NNS','NNP','NNPS'}:
            numeroNomi += 1
        if pos[1] in {'JJ','JJR','JJS'}:
            numeroAggettivi += 1
        if pos[1] in {'VB','VBD','VBG','VBN','VBP', 'VBZ'}:
            numeroVerbi += 1
    #--- Calcolo la media dei nomi per frase
    numeroNomiMedia = float(numeroNomi)/frasi
    #--- Calcolo la media degli aggettivi per frase
    numeroAggettiviMedia = float(numeroAggettivi)/frasi
    #--- Calcolo la media dei verbi per frase
    numeroVerbiMedia = float(numeroVerbi)/frasi
    return numeroNomiMedia, numeroAggettiviMedia, numeroVerbiMedia

def densitaLessicale (tokenPOS, corpus):
    numeroOccorrenze = 0
    numeroPuntiVirgole = 0
    for pos in tokenPOS:
        if pos[1] in {'NN','NNS','NNP','NNPS','JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP', 'VBZ', 'RB', 'RBR', 'RBS'}:
            numeroOccorrenze += 1
        #Conto quante volte appaiono i punti e le virgole per sottrarle dalla grandezza totale del corpus
        if pos[1] in {'.', ','}:
            numeroPuntiVirgole += 1
    densLessicale = float(numeroOccorrenze)/(corpus-numeroPuntiVirgole)
    return densLessicale
    
def main(testo1, testo2):
    fileInput1 = codecs.open(testo1, "r", "utf-8")
    fileInput2 = codecs.open(testo2, "r", "utf-8")
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# DIVIDO IL TESTO IN FRASI
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
# CREO DUE VARIABILI CON IL NUMERO TOTALE DELLE FRASI NEI DUE CORPORA
    numeroFrasi1 = len(frasi1)
    numeroFrasi2 = len(frasi2)

#-----------------INVOCAZIONE DELLE FUNZIONI-----------------

#--1) CALCOLO I TOKEN, IL NUMERO TOTALE DI TOKEN E EFFETTUO IL POS TAGGING
    token1, numeroToken1, tokenPOS1 = annotazioneLinguistica(frasi1)
    token2, numeroToken2, tokenPOS2 = annotazioneLinguistica(frasi2)

#--2) CALCOLO LA LUNGHEZZA MEDIA DELLE FRASI DEI DUE CORPORA
    lunghezzaMediaFrasi1 = calcolaLunghezzaMediaFrasi(numeroFrasi1, numeroToken1)
    lunghezzaMediaFrasi2 = calcolaLunghezzaMediaFrasi(numeroFrasi2, numeroToken2)

#--3) CALCOLO LA LUNGHEZZA MEDIA DELLE PAROLE DEI DUE CORPORA
    lunghezzaMediaParole1 = calcolaLunghezzaMediaParole(token1, numeroToken1)
    lunghezzaMediaParole2 = calcolaLunghezzaMediaParole(token2, numeroToken2)

#--4) CALCOLO IL VOCABOLARIO E LA TTR PER PORZIONI INCREMENTALI
    vocabolario1, typeTokenRatio1 = calcolaVocabolarioTTR(token1, numeroToken1)
    vocabolario2, typeTokenRatio2 = calcolaVocabolarioTTR(token2, numeroToken2)
#--5) CALCOLO LE CLASSI DI FREQUENZA SUI PRIMI 5000 TOKEN
    classiDiFrequenzaTesto1_V3, classiDiFrequenzaTesto1_V6, classiDiFrequenzaTesto1_V9 = classiDiFrequenza(token1)
    classiDiFrequenzaTesto2_V3, classiDiFrequenzaTesto2_V6, classiDiFrequenzaTesto2_V9 = classiDiFrequenza(token2)

#--6) POS TAGGING
    mediaSostantivi1, mediaAggettivi1, mediaVerbi1 = numeroMedioSostAggVbAvv(numeroFrasi1, tokenPOS1)
    mediaSostantivi2, mediaAggettivi2, mediaVerbi2 = numeroMedioSostAggVbAvv(numeroFrasi2, tokenPOS2)

    densitaLessicale_1 = densitaLessicale(tokenPOS1, numeroToken1)
    densitaLessicale_2 = densitaLessicale(tokenPOS2, numeroToken2)


#-----------------OUTPUT-----------------
    print "\nPrimo programma per il progetto di Linguistica Computazionale (A.A. 2018-2019)\n"
#---NUMERO TOTALE DI FRASI E DI TOKEN
    print "Confronto i due corpora per:\n"
    print "- NUMERO FRASI"
    print "Il corpus", testo1, "contiene:\t", numeroFrasi1, "frasi"
    print "Il corpus", testo2, "contiene:\t", numeroFrasi2, "frasi"
    if numeroFrasi1 > numeroFrasi2:
        print "Il corpus", testo1, "ha più frasi del corpus", testo2, "\n"
    elif numeroFrasi2 > numeroFrasi1:
        print "Il corpus", testo2, "ha più frasi del corpus", testo1, "\n"
    else:
        print "I due corpora possiedono lo stesso numero di frasi\n"

    print "- NUMERO DI TOKEN"
    print "Il corpus", testo1, "possiede:\t", numeroToken1, "token"
    print "Il corpus", testo2, "possiede:\t", numeroToken2, "token"
    if numeroToken1 > numeroToken2:
        print "Il corpus", testo1, "ha più token del corpus", testo2, "\n"
    elif numeroToken2 > numeroToken1:
        print "Il corpus", testo2, "ha più token del corpus", testo1, "\n"
    else:
        print "I due corpora possiedono lo stesso numero di token\n"

#---STAMPO LA LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN
    print "- LUNGHEZZA MEDIA DELLE FRASI IN TERMINI DI TOKEN ---"
    print "La lunghezza media delle frasi del corpus", testo1, "è pari a:\t", lunghezzaMediaFrasi1, "token"
    print "La lunghezza media delle frasi del corpus", testo2, "è pari a:\t", lunghezzaMediaFrasi2, "token"
    if lunghezzaMediaFrasi1 > lunghezzaMediaFrasi2:
        print "Il corpus", testo1, "ha una lunghezza media delle frasi in termini di token maggiore rispetto al corpus", testo2, "\n"
    elif lunghezzaMediaFrasi2 > lunghezzaMediaFrasi1:
        print "Il corpus", testo2, "ha una lunghezza media delle frasi in termini di token maggiore rispetto al corpus", testo1, "\n"
    else:
        print "La lunghezza media delle frasi è uguale nei due corpora\n"

#---STAMPO LA LUNGHEZZA MEDIA DELLE PAROLE IN TERMINI DI CARATTERI
    print "- LUNGHEZZA MEDIA DELLE PAROLE IN TERMINI DI CARATTERI ---"
    print "La lunghezza media delle parole nel corpus", testo1, "e' di:\t", lunghezzaMediaParole1, "caratteri"
    print "La lunghezza media delle parole nel corpus", testo2, "e' di:\t", lunghezzaMediaParole2, "caratteri"
    if lunghezzaMediaParole1 > lunghezzaMediaParole2:
        print "Il corpus", testo1, "ha una lunghezza media dei token in termini di caratteri maggiore rispetto al corpus", testo2, "\n"
    elif lunghezzaMediaParole2 > lunghezzaMediaParole1:
        print "Il corpus", testo2, "ha una lunghezza media dei token in termini di caratteri maggiore rispetto al corpus", testo1, "\n"
    else:
        print "La lunghezza media delle parole è uguale nei due corpora\n"


#---STAMPO LA GRANDEZZA DEL VOCABOLARIO DEL PRIMO CORPUS (recensioni positive)
    print "- GRANDEZZA DEL VOCABOLARIO E VALORE DELLA TYPE TOKEN RATIO (TTR) PER PORZIONI INCREMENTALI DI 1000 TOKEN---"
    print "Crescita del vocabolario nel corpus", testo1
    n = 1000
    m = 1000
    for i in vocabolario1:
        print "Vocabolario per i primi", n, "token:\t", i
        n += 1000
    print
    print "Crescita della TTR nel corpus", testo1
    for t in typeTokenRatio1:
        print "TTR calcolata sui primi", m, "token:\t", t
        m += 1000
    print "\n"

    print "Crescita del vocabolario nel corpus", testo2
    l = 1000
    o = 1000
    for i in vocabolario2:
        print "Vocabolario per i primi", l, "token:\t", i
        l += 1000
    print
    print "Crescita della TTR nel corpus", testo2
    for t in typeTokenRatio2:
        print "TTR calcolata sui primi", o, "token:\t", t
        o += 1000
    print "\n"


#---STAMPO LA GRANDEZZA DELLE CLASSI DI FREQUENZA V3, V6, V9 SUI PRIMI 5000 TOKEN
    print "- CLASSI DI FREQUENZA |V3|, |V6|, |V9| SUI PRIMI 5000 TOKEN"
    print "Corpus:\t", testo1
    print "Classe di frequenza |V3| = ", classiDiFrequenzaTesto1_V3
    print "Classe di frequenza |V6| = ", classiDiFrequenzaTesto1_V6
    print "Classe di frequenza |V9| = ", classiDiFrequenzaTesto1_V9, "\n"
    print "Corpus:\t", testo2
    print "Classe di frequenza |V3| = ", classiDiFrequenzaTesto2_V3
    print "Classe di frequenza |V6| = ", classiDiFrequenzaTesto2_V6
    print "Classe di frequenza |V9| = ", classiDiFrequenzaTesto2_V9, "\n"

#---STAMPO IL NUMERO MEDIO DI SOSTANTIVI, AGGETTIVI E VERBI PER FRASE:"
    print "- NUMERO MEDIO SOSTANTIVI, AGGETTIVI E VERBI"
    print "Corpus:\t", testo1
    print "Sostantivi:\t", mediaSostantivi1
    print "Aggettivi:\t", mediaAggettivi1
    print "Verbi:\t", mediaVerbi1, "\n"
    print "Corpus:\t", testo2
    print "Sostantivi: \t", mediaSostantivi2
    print "Aggettivi: \t", mediaAggettivi2
    print "Verbi: \t", mediaVerbi2, "\n"

#---STAMPO LA DENSITA' LESSICALE
    print "- DENSITA' LESSICALE"
    print "Densità lessicale nel corpus", testo1, "\t", densitaLessicale_1
    print "Densità lessicale nel corpus", testo2, "\t", densitaLessicale_2, "\n\n\n"
        
main(sys.argv[1], sys.argv[2])
