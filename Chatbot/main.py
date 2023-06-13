# Names: Ghaida Alshiddi & Bushra Rahman
# Class: CS 4395.001
# Assignment: Chatbot
# Due date: 4/15/23

import pathlib
from nltk.tokenize import word_tokenize, sent_tokenize
import random as rand
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords


# Global lists
GREETINGS = ['Hello', 'Hello there', 'Hi', 'Hi there', 'Hey', 'Greetings', 'Good to see you', 'Nice to meet you']
GOODBYES = ['Goodbye', 'Bye', 'Bye-bye', 'See you later', 'Talk to you later', 'Have a good day', 'Until next time', 'Thanks for talking', 'It was nice talking to you']
WELCOME = ['You\'re welcome', 'You\'re very welcome', 'No problem', 'Anytime', 'Of course', 'Sure thing', 'Happy to help', 'The pleasure is mine', 'Glad to be of service']
KNOWLEDGE = ['I can tell you something interesting about', 'In regards to', 'Well, when it comes to', 'About', 'If we\'re talking about', 'If you want to talk about']
SURE = ['Yes, I can', 'Of course I can', 'Will do', 'Sure thing', 'Definitely', 'Absolutely']
UNSURE = ['I\'m not so sure about that, but', 'Well, let me tell you this,', 'Let\'s just say that', 'Well, here\'s what I do know:', 'I just want to tell you that']


# This function is for calculating cosine similarity:
#               A · B       <-- dot product of 2 vectors A and B
# cos(θ) = _______________
#           ||A|| * ||B||   <-- normalization (division) by the vector norms
# The result is in the range [0, 1]; results close to 1 are better.
def cosine_similarity(A, B):
    cosine_similarity_value = float(dot(A, B)) / (norm(A) * norm(B))
    return cosine_similarity_value
# End of cosine_similarity()


# This function creates vector space models of every sentence in a list,
# and compares the cosine similarity value of each sentence to the first sentence in the list.
# The function returns the index of the sentence that has the highest cosine similarity to the first sentence.
def vector_space_modeler(sentences_list):
    # Create list of all tokens in all sentences
    tokens = []
    for sentence in sentences_list:
        tokens.extend(word_tokenize(sentence))

    # Preprocess tokens
    tokens_preprocessed = [tok for tok in tokens if tok not in stopwords.words('english') and tok.isalpha()]

    # Vocab = the set of all words in the corpus, sorted by alphabetization
    vocab = sorted(list(set(tokens_preprocessed)))

    # Create vector space model of each sentence
    vectors = []
    for sentence in sentences_list:
        # list of counts of each vocab word in the doc
        vec = [sentence.count(t) for t in vocab]
        vectors.append(vec)  # add list to vectors list

    max_cs = 0
    index = 0
    for i, vec in enumerate(vectors):
        # Get cosine similarity (cs) between first sentence (user input) and each vector
        cs = cosine_similarity(vectors[0], vec)
        # Get the doc with max cosine similarity to first doc
        if cs >= max_cs and i != 0:
            max_cs = cs
            index = i

    return index
# End of vector_space_modeler()


# This function generates a response for the chatbot.
def response_generator(user_response, knowledge_base_dict, user_name):
    # At the end we check whether this counter is still negative
    responses_counter = -1

    # Handle thankful messages
    if 'thank' in user_response.lower():
        print(rand.choice(WELCOME) + ', ' + user_name + '.')
        responses_counter += 1

    # Tokenize response
    tokens = [tok.lower() for tok in word_tokenize(user_response) if tok.isalpha() and tok not in stopwords.words('english')]

    # Search user response for terms in knowledge base (can handle multiple hits in one response)
    sure_counter = 0
    for token in tokens:
        # Pluralization used to check for singular or plural form of token in dict keys
        list_token = list(token)
        list_token.append('s')
        plural_token = ''.join(list_token)
        # Check for token (singular form, eg. 'sport') in dict keys
        if token in knowledge_base_dict.keys():
            if 'can' in user_response.lower() and sure_counter == 0:
                print(rand.choice(SURE) + '.')
                # Sure_counter makes sure a 'sure' response is only said once regardless of # of hits
                sure_counter += 1
            fact = rand.choice(knowledge_base_dict[token])
            print(rand.choice(KNOWLEDGE) + ' ' + token + ', ' + fact[0].lower() + fact[1:])
            responses_counter += 1
        # Check for token (plural form, eg. 'sports') in dict keys
        elif plural_token in knowledge_base_dict.keys():
            if 'can' in user_response.lower() and sure_counter == 0:
                print(rand.choice(SURE) + '.')
                # Sure_counter makes sure a 'sure' response is only said once regardless of # of hits
                sure_counter += 1
            fact = rand.choice(knowledge_base_dict[plural_token])
            print(rand.choice(KNOWLEDGE) + ' ' + plural_token + ', ' + fact[0].lower() + fact[1:])
            responses_counter += 1

    # If no hits via exact term, convert dict values to list and search entire list for relevant sentences
    sentences_list = [user_response]  # First element is current user response
    # Values() returns a list of the values, which here are themselves lists of sentences/facts/items
    for list_value in knowledge_base_dict.values():
        # Append each sentence/fact/item in the list to the overall sentences list
        for item in list_value:
            sentences_list.append(item)

    # Compare user input to sentences list to search for the sentence with the highest cosine similarity
    index = vector_space_modeler(sentences_list)
    sentence = sentences_list[index]
    # If nothing else has been output, this is the base case.
    if responses_counter < 0:
        print(rand.choice(UNSURE) + ' ' + sentence[0].lower() + sentence[1:])
        responses_counter += 1
# End of response_generator()


# main()
if __name__ == '__main__':
    # Open knowledge_base.txt
    with open(pathlib.Path.cwd().joinpath('knowledge_base.txt'), 'r', encoding='utf-8') as f:
        # Raw_text is a list split on newlines, where each element is a line in the file.
        raw_text = f.read().splitlines()

    # The knowledge_base.txt file is formatted to have an empty line separating term entries.
    while '' in raw_text:
        raw_text.remove('')

    # The knowledge base is a dict where {key:value} = {term:list of sentences}
    knowledge_base_dict = {}
    for line in raw_text:
        # Extract term from line
        term = word_tokenize(line)[0]
        # Initialize term as the dict key
        knowledge_base_dict[term] = sent_tokenize(line)
        # print(knowledge_base_dict[term][0])

    # Initialize output file for user models
    user_models_out = open('user_models.txt', 'a', encoding='utf-8', errors='ignore')
    user_name = input('I am a sports chatbot. What is your name?\n').capitalize()
    # User's name and input responses are separated using the backslash character '\'
    user_models_out.write(user_name + '\\')

    previous_user_flag = False
    with open(pathlib.Path.cwd().joinpath('user_models.txt'), 'r', encoding='utf-8') as f:
        # Read in users from user model file to find user's name if present
        users = f.read().splitlines()
    for line in users:
        if line.startswith(user_name):
            previous_user_flag = True

    # Print personalized greeting for returning users
    if previous_user_flag:
        print(rand.choice(GREETINGS) + ' again, ' + user_name + '. Let\'s talk about sports.')
    else:
        print(rand.choice(GREETINGS) + ', ' + user_name + '. Let\'s talk about sports.')

    # Begin conversation
    continue_flag = True
    while continue_flag:
        user_response = input()
        user_models_out.write(user_response + '\\')
        if 'bye' in user_response.lower() or user_response.capitalize() in GOODBYES:
            user_models_out.write('\n')
            print(rand.choice(GOODBYES) + ', ' + user_name + '.')
            continue_flag = False
        else:
            response_generator(user_response, knowledge_base_dict, user_name)
    # End of while loop
# End of main()
