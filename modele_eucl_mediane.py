import math
import os
import re
import random
from statistics import median

import nltk
from collections import Counter, OrderedDict
from nltk import word_tokenize, ngrams
import numpy as np


# Objet Etiquette_from_sentence qui stockera le meilleur mot pour chaque etiquette
class Etiquette_from_sentence():
	etiquette = ""
	theme = []
	best_word = ""
	word_before = ""
	word_after = ""
	word_before_is_etiquette = False

	def __init__(self):
		self.etiquette = ""
		self.theme = []
		self.best_word = ""
		self.word_before = ""
		self.word_after = ""
		word_before_is_etiquette = False

	def set_best_word(self, word):
		self.best_word = word

	def set_word_before(self, word):
		self.word_before = word

	def set_word_after(self, word):
		self.word_after = word

	def set_word_before_etiquette(self):
		self.word_before_is_etiquette = True


# Objet Etiquette_from_table qui stockera la liste des mots pour une etiquette a partir de la table associative
class Etiquette_from_table():
	etiquette = ""
	words = []

	def __init__(self):
		self.etiquette = ""
		self.words = []


dict_bi_gram = None
table_of_words = []
list_of_etiquettes_sentence = []
dict_vectors = {}
query = ""
all_sentences = []


# Pre-traitement : extraction des mot alphabetique seulement (elimination des caracteres speciaux et numeriques)
def traitement(file):
	words = ' '.join(re.findall('\w+', file.read()))
	words = words.lower()
	words = re.sub("[0-9]", "", words)
	# print(words)
	return words


# Creation d'un dictionnaire qui represente le model du n-gram correspondant
def model(path, n_gram_size):
	words = ''
	for filename in os.listdir(path):
		file = open(path + '/' + filename, 'r', encoding="utf8")
		words += traitement(file)
	n_gram = extract_ngrams(words, n_gram_size)
	dict_n_gram = get_dictionnary(n_gram)
	return dict_n_gram


# Ecriture du model n-gram dans un fichier
def write_model(dict, n_gram_size):
	file = open('models/ML-' + str(n_gram_size) + '.txt', 'w', encoding="utf8")
	for key in dict:
		line = ''
		for word in key.split():
			line += word + '\t'
		line += str(dict.get(key)) + '\n'
		file.write(line)


# Creation d'un dictionnaire du model n-gram a partir d'un fichier
def create_dictionnary_from_file(file, n_gram_size):
	key = ''
	dict = {}
	for line in file:
		list_of_words = line.split('\t')
		for i in range(n_gram_size):
			key += list_of_words[i] + ' '
		dict[key] = list_of_words[n_gram_size].split('\n')[0]
		key = ''
	return dict


# Recuperation du model n-gram a partir d'un fichier
def get_model_from_file(n_gram_size):
	if (n_gram_size == 2):
		file = open('models/ML-2.txt', 'r', encoding="utf8")
		return create_dictionnary_from_file(file, 2)

	elif (n_gram_size == 3):
		file = open('models/ML-3.txt', 'r')
		return create_dictionnary_from_file(file, 3)

	else:
		return 'No model'


# Generer les n-grams a partir d'un texte
def extract_ngrams(data, num):
	n_grams = ngrams(word_tokenize(data), num)
	return [' '.join(grams) for grams in n_grams]


# Reordonner un dictionnaire
def get_dictionnary(data):
	dict = Counter(data)
	return OrderedDict(sorted(dict.items(), key=lambda x: x[1], reverse=True))


# Recuperer le nombre d'occurence a partir d'un dictionnaire
def get_occurences(dict, n_gram):
	return dict.get(n_gram)


# Recuperer le prochain mot
def guess_next_word(word, dict):
	for key in dict:
		if (word == key.split()[0]):
			return key.split()[1]


# Generer une phrase a partir d'un modele de langage (n-gram)
def get_sentence(first_word, dict, size):
	sentence = first_word + ' '
	for i in range(size - 1):
		next_word = guess_next_word(first_word, dict)
		sentence += next_word + ' '
		first_word = next_word
	return sentence


# Recuperer les donnees des etiquettes a partir de la table associative (TA)
def get_TA(path):
	file = open(path, 'r')
	dict_TA = {}
	for line in file:
		line = line.rstrip('\n').split('\t')
		pos = line[0]
		value = line[1:]
		dict_TA[pos] = value
	return dict_TA


# Recuperer un dictionnaire qui represente les mots et leurs valeurs vectorielles
def get_vector_model(path):
	file = open(path, 'r', encoding="utf8")
	dict_vectors = {}
	for line in file:
		line = line.split('\t')
		pos = line[0]
		pos = pos.lower()
		value = line[1].replace('[', '').replace(']', '').replace(',', '').split()
		dict_vectors[pos] = np.array(value).astype(np.float64)
	return dict_vectors


# Calcul de la distance euclidienne entres 2 vecteurs
def angle_between(v1, v2):
	v1_u = np.array(v1)
	v2_u = np.array(v2)
	return np.sqrt(np.sum(np.square(v1_u - v2_u)))


# Recuperer les templates de phrases
def get_sentence_templates(path):
	file = open(path, 'r', encoding="utf8")
	templates = []
	for line in file:
		line = line.split(' ')
		templates.append(line)
	return templates


# Recuperer la liste des etiquettes par phrase
def get_etiquettes_from_template(template):
	etiquettes = []
	# for element in template:
	for i in range(len(template)):
		element = template[i]
		if element[0] == "*":
			tmp = element.split("/")
			tmp_etiquette = Etiquette_from_sentence();
			tmp_etiquette.etiquette = tmp[0].replace('*', '')
			if tmp[1] != tmp[2]:
				# tmp_etiquette.theme.append(tmp[1])
				tmp_etiquette.theme.append(tmp[2])
			else:
				tmp_etiquette.theme.append(tmp[2])

			# tmp_etiquette.set_word_after(template[i + 1])
			tmp_etiquette.set_word_before(template[i - 1])
			etiquettes.append(tmp_etiquette)
	return etiquettes


# Recuperer la liste des mots pour chaque etiquette de la table associative
def get_list_words_etiquette(path):
	file = open(path, 'r', encoding="utf8")
	etiquettes_model = []
	for line in file:
		line = line.replace('\n', '')
		line = line.split('\t')
		tmp_etiquette = Etiquette_from_table();
		tmp_etiquette.etiquette = line[0]
		tmp_etiquette.words = line[1:]
		etiquettes_model.append(tmp_etiquette)
	return etiquettes_model


# Mettre a jour une etiquette pendant le remplissage des autres etiquettes
def update_etiquette(etiquette_sentence):
	for i in range(len(list_of_etiquettes_sentence)):
		for j in range(len(list_of_etiquettes_sentence[i])):
			last_etiquette = None
			if j > 0:
				last_etiquette = list_of_etiquettes_sentence[i][j - 1]
			etiquette = list_of_etiquettes_sentence[i][j]
			if etiquette_sentence == etiquette:
				if etiquette_sentence.word_before_is_etiquette:
					etiquette_sentence.set_word_before(last_etiquette.best_word)


# Supprimer un mot deja utilise dans la phrase dans nos etiquettes pendant la recherche
def update_best_words(best_words, index):
	best_words_updated = dict(best_words)
	for j in range(len(list_of_etiquettes_sentence[index])):
		etiquette = list_of_etiquettes_sentence[index][j]
		if (etiquette.best_word in best_words_updated):
			del best_words_updated[etiquette.best_word]
			print("supprimer :", etiquette.best_word, ", phrase : ", index)

	return best_words_updated


# Recuperer le meilleur mot apres la correspondance avec les bi-grammes
def get_best_word_with_bi_grams(best_words, etiquette_sentence, index):
	update_etiquette(etiquette_sentence)
	if (index != 100):
		best_words = update_best_words(best_words, index)
	best_5_words = list(OrderedDict(sorted(best_words.items(), key=lambda x: x[1], reverse=True)))[:30]
	print("best : ", best_5_words)
	words_bi_grams = []
	for word in best_5_words:
		if check_bi_grams(word, etiquette_sentence):
			words_bi_grams.append(word)
	print("bi-gram :", words_bi_grams)
	if (len(words_bi_grams) == 0):
		best_notbigram_words = list(OrderedDict(sorted(best_words.items(), key=lambda x: x[1], reverse=True)))[:10]
		return random.choice(best_notbigram_words)
	else:
		return random.choice(words_bi_grams)


# Recuperer le meilleur mot pour la première etiquette
def get_best_word_from_first_etiquette(etiquette_sentence):
	sentence_words = []
	etiquette = etiquette_sentence.etiquette
	for theme in etiquette_sentence.theme:
		words = get_list_of_words_by_etiquette(etiquette)
		dict_theme = get_dictionnary_words_distance(words, theme)
		dict_query = get_dictionnary_words_distance(words, query)
		best_words = get_best_words_compared_to_query_and_theme(dict_theme, dict_query, theme, query)
		sentence_words.append(best_words)
		best_word = get_best_word_with_bi_grams(best_words, etiquette_sentence, 100)
		etiquette_sentence.set_best_word(best_word)
	# etiquette_sentence.set_best_word(best_word)
	return sentence_words


# Generer le dictionnaire des distances des mots avec le thème ou la query
def get_dictionnary_words_distance(words, theme):
	dict_words = {}
	for word in words:
		if word == 'gynecologie':
			exists = word in dict_vectors.keys()
			a = dict_vectors.keys()

		if word in dict_vectors.keys() and word != theme:
			if word == 'gynecologie':
				exists = word in dict_vectors.keys()
				a = dict_vectors.keys()
			tmp_d = angle_between(dict_vectors[word], dict_vectors[theme])
			dict_words[word] = tmp_d
	return dict_words


# Recuperer un dictionnaire des eventuelles meilleurs mots pour la première etiquette d'une phrase (filtration)
def get_best_words_compared_to_query_and_theme(dict_theme, dict_query, theme, query):
	best_moyenne = 0
	best_words = {}
	d_middle_theme = median(dict_theme.values())
	d_middle_query = median(dict_query.values())
	for word in dict_theme.keys():
		if word != theme and word != query:
			if theme == query:
				d_query = dict_query[word]
				if d_query < d_middle_query:
					best_words[word] = d_query
			else:
				d_theme = dict_theme[word]
				d_query = dict_query[word]
				tmp_moyenne = (d_query + d_theme) / 2
				if d_query < d_middle_query and d_theme > d_middle_theme:
					best_words[word] = tmp_moyenne
	return best_words


# Recuperer la liste des mots pour une etiquette_from_table
def get_list_of_words_by_etiquette(etiquette):
	for element in table_of_words:
		if element.etiquette == etiquette:
			return element.words


# Rechercher les meilleurs mots pour toutes les etiquettes dans les phrases
def get_all_words_for_sentence():
	for i in range(len(list_of_etiquettes_sentence)):
		for j in range(len(list_of_etiquettes_sentence[i])):
			if (j == 0):
				get_best_word_from_first_etiquette(list_of_etiquettes_sentence[i][j])
			else:
				get_best_word_from_etiquette(list_of_etiquettes_sentence[i][j], i, j)


# Rechercher les meilleurs mots pour toutes les phrases
def get_best_word_from_etiquette(etiquette_sentence, i, j):
	etiquette = etiquette_sentence.etiquette
	for theme in etiquette_sentence.theme:
		words = get_list_of_words_by_etiquette(etiquette)
		dict_theme = get_dictionnary_words_distance(words, theme)
		dict_query = get_dictionnary_words_distance(words, query)
		best_words = get_best_words_for_second_and_more_etiquette(dict_theme, dict_query, theme, query, i, j)
		best_word = get_best_word_with_bi_grams(best_words, etiquette_sentence, i)
		etiquette_sentence.set_best_word(best_word)


# Recuperer un dictionnaire des eventuelles meilleurs mots pour les etiquettes d'une phrase (sauf la premiere)(filtration)
def get_best_words_for_second_and_more_etiquette(dict_theme, dict_query, theme, query, i, j):
	best_moyenne = 0
	best_words = {}

	d_middle_theme = median(dict_theme.values())
	d_middle_query = median(dict_query.values())
	for word in dict_theme.keys():
		if word != theme and word != query:
			if theme == query:
				d_query = dict_query[word]
				if d_query < d_middle_query:
					best_words[word] = d_query
			else:
				d_theme = dict_theme[word]
				d_query = dict_query[word]
				d_last_etiquette = angle_between(dict_vectors[word],
												 dict_vectors[list_of_etiquettes_sentence[i][j - 1].best_word])
				tmp_moyenne = (d_query + d_theme + d_last_etiquette) / 3

				if d_query < d_middle_query and d_theme > d_middle_theme:
					best_words[word] = tmp_moyenne
	return best_words


# Verifier la correspondance avec les bi-grams
def check_bi_grams(best_word, etiquette):
	for bigram in dict_bi_gram:
		bigram = bigram.split(" ")
		if (bigram[0] == etiquette.word_before) and (bigram[1] == best_word):
			return True
	return False


# Affichage des phrases et execution de l'algorithme de recherche des mots
def display_sentence_from_templates():
	queries = ["tristesse", "amour", "joie", "haine", "bleu"]
	file = open('output/result_mediane.txt', 'w', encoding="utf8")
	global list_of_etiquettes_sentence
	global query
	for q in queries:
		query = q
		list_of_etiquettes_sentence = []
		templates_path = "tools/templates_basiques.txt"
		templates = get_sentence_templates(templates_path)
		for template in templates:
			list_of_etiquettes_sentence.append(get_etiquettes_from_template(template))
		get_all_words_for_sentence()

		index = 0
		for i in range(len(templates)):
			line = ""
			for word in templates[i]:
				if (word[0] == "*"):
					line += list_of_etiquettes_sentence[i][index].best_word + " "
					print(list_of_etiquettes_sentence[i][index].best_word, end=" ")
					index = index + 1
				else:
					if (word != ".\n"):
						line += word + " "
					print(word, end=" ")
			index = 0
			print()
			line += ".\t" + query + "\n"
			file.write(line)


if __name__ == '__main__':

	choice = 5
	print("- Tapez 1 pour le modele de langage ")
	print("- Tapez 2 pour le modele de neuronal ")
	print("- Tapez 0 pour Quitter ")
	while (choice != 0):
		choice = int(input("Entrez votre choix : "))

		if choice == 1:  ############    le modele de langage    #############

			data_dir = 'data'
			dict_bi_gram = model(data_dir, 2)
			dict_tri_gram = model(data_dir, 3)
			word = ""
			nb_words = 1
			word = input("Entrez le premier mot : ")
			nb_words = int(input("Entrez la longueur de la phrase (nombre de mots) : "))

			print("modele 2-grams : " + get_sentence(word, dict_bi_gram, nb_words))
			print("modele 3-grams : " + get_sentence(word, dict_tri_gram, nb_words))

			write_models = 0
			write_models = int(input("Tapez 1 si vous voulez sauvegarder les modeles: "))
			if write_models == 1:
				write_model(dict_bi_gram, 2)
				write_model(dict_tri_gram, 3)

		if choice == 2:  ############    le modele de neuronal    #############
			print("Traitement de donnees, cela peut prendre un moment ...")
			path = 'tools/embeddings-Fr.txt'
			dict_vectors = get_vector_model(path)
			table_of_words = get_list_words_etiquette("tools/TableAssociative.txt")
			display_sentence_from_templates()

		if (choice != 0):
			print("- Tapez 1 pour le modele de langage ")
			print("- Tapez 2 pour le modele de neuronal ")
			print("- Tapez 0 pour Quitter ")
