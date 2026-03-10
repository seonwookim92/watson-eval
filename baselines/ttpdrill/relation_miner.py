import spacy
import helper
import json

class relation_miner():
    def __init__(self, nlp_model=None):
        if nlp_model:
            self.nlp = nlp_model
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                import os
                # Fallback if not loaded
                self.nlp = None

    def getProperText(self, text):
        # Keep original logic if needed, but Spacy handles stopwords too
        return text

    def depparse(self, text):
        if not self.nlp:
            return []
        doc = self.nlp(text)
        parsed = []
        current_parsed = []
        for token in doc:
            dep = token.dep_
            gov = token.head.text
            dep_text = token.text
            
            if dep == 'ROOT':
                current_parsed.append(('ROOT', 'ROOT', dep_text))
            elif dep == 'pobj' and token.head.dep_ == 'prep':
                # Map prep + pobj to nmod:prep_text
                # e.g., leveraged -> on (prep) -> systems (pobj) => ('nmod:on', 'leveraged', 'systems')
                prep_token = token.head
                new_dep = f"nmod:{prep_token.text.lower()}"
                new_gov = prep_token.head.text
                current_parsed.append((new_dep, new_gov, dep_text))
            elif dep == 'agent' and token.head.dep_ == 'prep':
                new_gov = token.head.head.text
                current_parsed.append(('nmod:agent', new_gov, dep_text))
            else:
                current_parsed.append((dep, gov, dep_text))
        
        parsed.append(current_parsed)
        return parsed

    def get_important_relations(self, dep_tree, sentence):
        extracted_words = dict()
        what_bagofwords = set()
        where_bagofwords = set()
        where_attribute_bagofwords = set()
        how_bagofwords = set()
        why_bagofwords = set()
        when_bagofwords = set()
        subject_bagofwords = set()
        action_bagofwords = set()

        for node in dep_tree[0]:
            # node is (dep, gov, text)
            rel, gov, text = node

            if rel == 'ROOT':
                what_bagofwords.add(text)
                action_bagofwords.add(text)

            # Extract syntactic subject (active and passive)
            if rel in ('nsubj', 'nsubjpass'):
                subject_bagofwords.add(text)

            self.get_relation(node, 'dobj', what_bagofwords, where_bagofwords)
            self.get_relation(node, 'nmod:through',
                              where_bagofwords,
                              where_bagofwords)
            self.get_relation(node, 'nmod:using',
                              where_bagofwords,
                              where_bagofwords)
            self.get_relation(node, 'nmod:into',
                              where_bagofwords,
                              where_bagofwords)

        #        what_bafofwords.append(verb)
        #        where_bagofwords.append(obj)
        extracted_words['what'] = helper.remove_stopwords(' '.join(list(what_bagofwords)))
        extracted_words['where'] = helper.remove_stopwords(' '.join(list(where_bagofwords)))
        extracted_words['where_attribute'] = helper.remove_stopwords(' '.join(list(where_attribute_bagofwords)))
        extracted_words['why'] = helper.remove_stopwords(' '.join(list(why_bagofwords)))
        extracted_words['when'] = helper.remove_stopwords(' '.join(list(when_bagofwords)))
        extracted_words['how'] = helper.remove_stopwords(' '.join(list(how_bagofwords)))
        extracted_words['subject'] = helper.remove_stopwords(' '.join(list(subject_bagofwords)))
        extracted_words['action'] = helper.remove_stopwords(' '.join(list(action_bagofwords)))
        extracted_words['text'] = sentence


        return extracted_words

    def get_relation(self, node, relation_type, *argv):
        #        print(node)
        if node[0] == relation_type:
            k = 1
            for arg in argv:
                #                print(arg)
                arg.add(node[k])
                k += 1
            #                print(arg)
            #            print(node[1], node[2])
            return node[1], node[2]

    def list_important_info(self, text):

        dep_parse_tree = self.depparse(text)
        #        print(dep_parse_tree)
        important_dict = self.get_important_relations(dep_parse_tree, text)
        return important_dict

    def all_imp_stuff(self, text):
        ourput_list = list()
        for sent in text:
            print(sent)
            dict_ = self.list_important_info(sent)
            print(dict_)
            ourput_list.append(dict_)

        return ourput_list

    def get_important_relations_new(self, list_of_tuples, sentence):
        list_of_forest = []
        for tuples in list_of_tuples:
            nodes = {}
            forest = []
            for count1 in tuples:
                print(count1)
                # count1:
                # ('ROOT', 'ROOT', 'eat')
                # ('nsubj', 'eat', 'I')
                # ('dobj', 'eat', 'chicken')
                # ('punct', 'eat', '.')
                rel, parent, child = count1
                # nodes[child]
                # {'Name': 'eat', 'Relationship': 'ROOT'}
                # {'Name': 'I', 'Relationship': 'nsubj'}
                # {'Name': 'chicken', 'Relationship': 'dobj'}
                # {'Name': '.', 'Relationship': 'punct'}

                # if rel in ['dobj','amod','compound']:
                    # print(count1)
                    # nodes[parent] = {'Name': parent, 'Relationship': rel}
                nodes[child] = {'Name': child, 'Relationship': rel}

            for count2 in tuples:
                # count2
                # ('ROOT', 'ROOT', 'eat')
                # ('nsubj', 'eat', 'I')
                # ('dobj', 'eat', 'chicken')
                # ('punct', 'eat', '.')
                rel, parent, child = count2
                # node
                # {'Name': 'eat', 'Relationship': 'ROOT'}
                # {'Name': 'I', 'Relationship': 'nsubj'}
                # {'Name': 'chicken', 'Relationship': 'dobj'}
                # {'Name': '.', 'Relationship': 'punct'}
                # if rel in ['dobj', 'amod', 'compound']:
                    # print(count2)
                node = nodes[child]

                if parent == 'ROOT':
                    # {'Name': 'eat', 'Relationship': 'ROOT'}
                    forest.append(node)
                else:
                    # parent
                    # {'Name': 'eat', 'Relationship': 'ROOT'}
                    # {'Name': 'eat', 'Relationship': 'ROOT', 'children': [{'Name': 'I', 'Relationship': 'nsubj'}]}
                    # {'Name': 'eat', 'Relationship': 'ROOT', 'children': [{'Name': 'I', 'Relationship': 'nsubj'}, {'Name': 'chicken', 'Relationship': 'dobj'}]}
                    parent = nodes[parent]
                    if not 'children' in parent:
                        parent['children'] = []
                    children = parent['children']
                    children.append(node)

            list_of_forest.append(forest)

        print('---------------------------------------')
        print(list_of_forest)
        print(list_of_tuples)
        # for relation in dep_tree[0]:
        #     if relation[0] == 'dobj':
        #
        #     print(relation)
        return

def test():
    from helper import FileReader
    from helper import StanfordServer
    isFile = True
    isStemmer = False
    isServerRestart = False
    report_name = 'reports/test.txt'
    preprocess_tools = FileReader(report_name)
    text = preprocess_tools.read_file()
    text_list = preprocess_tools.get_sent_tokenize(text)
    stanfordServer = StanfordServer()
    if isServerRestart:
        stanfordServer.startServer()
    stanfordNLP = stanfordServer.get_stanforcorenlp()
    print(text_list)
    # extracted_list = getReportExtraction(isFile, isStemmer, isServerRestart, report_name)
    # print(extracted_list)
    nlp_extract = relation_miner(stanfordNLP)
    extracted_list = nlp_extract.all_imp_stuff(text_list)
    print(extracted_list)

def tree_example():
    from anytree import Node, RenderTree

    udo = Node(name='nsubj')
    marc = Node(name='dobj', parent=udo)
    lian = Node(parent=marc, name='amod')
    dan = Node(parent=udo, name='nmod:for')
    jet = Node(parent=dan, name='nsubj')
    jan = Node(parent=dan, name='compound')
    joe = Node(parent=dan, name='det')

    print(udo)
    Node('/Udo')
    print(joe)
    Node('/Udo/Dan/Joe')

    for pre, fill, node in RenderTree(udo):
        print("%s%s%s" % (pre, node.name, fill))

    print(dan.children)
    (Node('/Udo/Dan/Jet'), Node('/Udo/Dan/Jan'), Node('/Udo/Dan/Joe'))

if __name__=='__main__':
    test()

