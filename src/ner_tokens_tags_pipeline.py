# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import regex
from pprint import pprint
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer


class NerTokes:
    def __doc__(self):
        """при инициализации подается путь до изображения"""

    def __init__(self, nlp_res, label_list, text='', conj='##', conf=0.8):
        self.text = text
        self.nlp_res = nlp_res
        self.detokenizer = NLTKMosesTokenizer(False, 'ru').detokenizer
        self.token_text = None
        self.indices = None
        self.label_list = label_list
        self.conj = conj
        self.joined_tokens = None
        self.norm_tokens = None
        self.joined_i = None
        self.joined_bi = None
        self.confidences = conf
        self.final_tags = None
        self.__make_features()

    def __make_features(self):
        self.token_text = ['[CLS]', ]
        self.indices = [0]
        self.positions = [(-1, -1)]
        self.scores = ['-1']

        for r in self.nlp_res:
            self.token_text.append(r['word'])
            self.indices.append(int(r['entity'].split('_')[-1]))
            self.positions.append((r['start'], r['end']))
            self.scores.append(r['score'])

        self.token_text.append('[SEP]')
        self.indices.append(0)
        self.positions.append((-1, -1))
        self.scores.append('-1')
        # print(self.positions)

    @staticmethod
    def get_most_common(enti):
        """
        Считаем наиболее частые встречи тегов в зависимости от приставки тега и в общем
        """
        result = dict(B=[], I=[], O=[], Overall=[])
        cnt = Counter(list(filter(lambda x: x[0] == 'B', enti))).most_common()
        max_v = cnt[0][1] if len(cnt) > 0 else 0
        result['B'] = list(filter(lambda x: x[1] >= max_v, cnt)) if max_v > 0 else [('', 0), ]

        cnt = Counter(list(filter(lambda x: x[0] == 'I', enti))).most_common()
        max_v = cnt[0][1] if len(cnt) > 0 else 0
        result['I'] = list(filter(lambda x: x[1] >= max_v, cnt)) if max_v > 0 else [('', 0), ]

        cnt = Counter(list(filter(lambda x: x[0] == 'O', enti))).most_common()
        max_v = cnt[0][1] if len(cnt) > 0 else 0
        result['O'] = list(filter(lambda x: x[1] >= max_v, cnt)) if max_v > 0 else [('', 0), ]

        cnt = Counter(enti).most_common()
        max_v = cnt[0][1] if len(cnt) > 0 else 0
        result['Overall'] = list(filter(lambda x: x[1] >= max_v, cnt)) if max_v > 0 else [('', 0), ]
        return result

    def normalize_tokens(self):
        """
        Приводим к единому значению тэги у слова, если оно состоит из нескольких токенов и нескольких тэгов
        """

        assert (self.joined_tokens is not None), "joined_tokens  еще не заполнена, сначала вызовите метод join_tokens()"

        spans = self.joined_tokens
        new_entities = []
        for i, enti in enumerate(spans):

            if i == 0:
                if len(enti['tags']) == 1:
                    enti['tags'] = [enti['tags'][0][0].replace('I-', 'B-'), enti['tags'][0][1]]
                    new_entities.append(enti)
                elif len(enti['tags']) > 1:
                    if enti['tags'][0][1] >= self.confidences:
                        enti['tags'] = [enti['tags'][0][0].replace('I-', 'B-'), enti['tags'][0][1]]
                        new_entities.append(enti)
                    else:
                        tags_1 = spans[i + 1]['tags']

                        new_ent = ""
                        for c in enti['tags']:
                            t0 = c[0].replace('I-', '').replace('B-', '')
                            for cc in tags_1:
                                tt0 = cc[0].replace('I-', '').replace('B-', '')
                                if t0 == tt0:
                                    new_ent = c[0].replace('I-', 'B-').replace('I-', 'B-')
                                    new_ent = [new_ent, c[1]]
                                    break
                        if new_ent != "":
                            enti['tags'] = new_ent
                            new_entities.append(enti)
                        else:
                            b = 0
                            for ii in enti['tags']:
                                if ii[0][0] == 'B':
                                    enti['tags'] = [ii[0], ii[1]]
                                    new_entities.append(enti)
                                    b = 1
                                    break
                            if b == 0:
                                for j in enti['tags']:
                                    if j[0][0] == 'I':
                                        enti['tags'] = [j[0], j[1]]
                                        new_entities.append(enti)
                                        b = 1
                                        break
                            if b == 0:
                                enti['tags'] = [enti['tags'][0][0].replace('I-', 'B-'), enti['tags'][0][1]]
                                new_entities.append(enti)
            else:
                if enti['tags'][0][1] >= self.confidences:
                    enti['tags'] = [enti['tags'][0][0], enti['tags'][0][1]]
                    new_entities.append(enti)
                    continue
                t00 = new_entities[i - 1]['tags'][0].replace('I-', '').replace('B-', '')
                if i < len(spans) - 1:
                    tags_1 = spans[i + 1]['tags']

                    new_ent = ""
                    if t00 != 'O':
                        for cc in tags_1:
                            t0 = cc[0].replace('I-', '').replace('B-', '')
                            if t00 == t0:
                                new_ent = cc[0].replace('B-', 'I-')
                                new_ent = [new_ent, cc[1]]
                                break
                        if new_ent != "":
                            enti['tags'] = new_ent
                            new_entities.append(enti)
                        else:
                            enti['tags'] = enti['tags'][0]
                            new_entities.append(enti)
                    else:
                        enti['tags'] = [enti['tags'][0][0].replace('I-', 'B-'), enti['tags'][0][1]]
                        new_entities.append(enti)
                else:
                    _ = True
                    for e in enti['tags']:
                        if t00 in e[0] and e[0][0] == 'I':
                            enti['tags'] = [e[0], e[1]]
                            new_entities.append(enti)
                            _ = False
                            break
                    if _:
                        e = enti['tags'][0]
                        enti['tags'] = e
                        new_entities.append(enti)
        self.norm_tokens = new_entities
        return new_entities

    def join_i_tag(self):
        """
        Объединяем последовательно идущие Тэги I-
        """
        assert (self.norm_tokens is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод normalize_tokens()"

        spans = self.norm_tokens
        st = {'word': '', 'coord': [], 'tags': ['', 0]}
        result = []
        for i, enti in enumerate(spans):
            e0 = enti['tags'][0][0]
            if e0 in ['I', 'O']:
                # if e0 == 'O':
                if (e0 == 'O') and (len(regex.findall(r"^(\индивидуальный\b){e<=4}", enti['word'].lower())) > 0 \
                        or len(regex.findall(r"^(\предприниматель\b){e<=4}", enti['word'].lower())) > 0 \
                        or enti['word'].lower() == 'ип'):
                    result.append(enti)
                else:
                    if st['tags'][0] == enti['tags'][0]:
                        st['word'] += ' ' + enti['word']
                        st['coord'] = [st['coord'][0], enti['coord'][1]]
                        st['tags'] = enti['tags']
                    else:
                        if len(st['word']) > 0:
                            st['word'] = st['word'].strip()
                            result.append(st)
                        st = enti
            if e0 in ['B']:
                if len(st['word']) > 0:
                    result.append(st)
                    st = enti
                else:
                    result.append(enti)
        if len(st['word']) > 0:
            if len(result) == 0:
                result.append(st)
            elif st['tags'][0] != result[-1]['tags'][0]:
                result.append(st)
        self.joined_i = sorted(result, key=lambda x: x['coord'][0])
        return self.joined_i

    def join_bi_tag(self):

        assert (self.joined_i is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод join_i_tag()"
        res_spans = self.joined_i
        # Нормализуем Теги "B-", если слово начинается с тега "i-"
        # и перед ним нет аналогичного тега "B-", то присвоить ему тег "B-"
        st = {'word': '', 'coord': [], 'tags': ['', 0]}
        result = []

        for i, enti in enumerate(res_spans):
            e0 = enti['tags'][0][0]
            cur_tag = enti['tags'][0]
            if e0 in ['B']:
                if len(st['word']) > 0:
                    # st['word'] = st['word'].strip()
                    result.append(st)
                    st = {'word': enti['word'], 'coord': [enti['coord'][0], enti['coord'][1]],
                          'tags': [enti['tags'][0].split('-')[-1], enti['tags'][1]]}
                else:
                    st['word'] = enti['word']
                    if len(st['coord']) > 0:
                        st['coord'] = [st['coord'][0], enti['coord'][1]]
                    else:
                        st['coord'] = [enti['coord'][0], enti['coord'][1]]
                    st['tags'] = [enti['tags'][0].split('-')[-1], enti['tags'][1]]  # enti['tags']
            elif e0 in ['I']:
                if len(st['word']) > 0:
                    if st['tags'][0] == cur_tag.split('-')[-1]:
                        st['word'] += ' ' + enti['word']
                        if len(st['coord']) > 0:
                            st['coord'] = [st['coord'][0], enti['coord'][1]]
                        else:
                            st['coord'] = [enti['coord'][0], enti['coord'][1]]
                        #                 st['tags'] = enti['tags']
                        st['tags'] = [enti['tags'][0].split('-')[-1], enti['tags'][1]]
                    else:
                        result.append(st)
                        st = {'word': enti['word'], 'coord': [enti['coord'][0], enti['coord'][1]],
                              'tags': ['O', enti['tags'][1]]}
                else:
                    st['tags'] = ['O', enti['tags'][1]]
                    st['word'] = enti['word']
                    st['coord'] = [enti['coord'][0], enti['coord'][1]]
                    result.append(st)
                    st = {'word': '', 'coord': [], 'tags': ['', 0]}
            elif e0 in ['O']:
                if len(st['word']) > 0:
                    st['word'] = st['word'].strip()
                    result.append(st)
                result.append({'word': enti['word'], 'coord': [enti['coord'][0], enti['coord'][1]],
                               'tags': ['O', enti['tags'][1]]})
                st = {'word': '', 'coord': [], 'tags': ['', 0]}

        if len(st['word']) > 0:
            result.append(st)
        self.joined_bi = sorted(result, key=lambda x: x['coord'][0])
        return self.joined_bi

    @staticmethod
    def __get_single_tag(tags, sc):
        """
        Предварительный выбор тэга
        """
        return tags[np.argmax(sc)]

    @staticmethod
    def __make_coord_tags(tags):
        coord = [tags[0][1][0], tags[-1][1][1]]
        ft = []
        sorted_tags = sorted(tags, key=lambda k: k[2], reverse=True)
        st = set(t[0] for t in sorted_tags)
        if len(st) > 1:
            ft = [[tt[0], tt[2]] for tt in sorted_tags]
        else:
            ft = [[sorted_tags[0][0], sorted_tags[0][2]]]
        return coord, ft

    def join_tokens(self):
        """
        Объединяем токены в слова и соединяем с тэгами
        """
        sp = ""
        spans = []
        tags = []
        i = 0
        word_coordinates = []

        for idx, t in enumerate(self.nlp_res):
            li = int(t['entity'].split('_')[-1])
            if t['word'][:2] == self.conj:
                sp += t['word'][2:]
                tags.append([self.label_list[li], [t['start'], t['end']], t['score']])
            else:
                sp += r' {}'.format(t['word'])
                tags.append([self.label_list[li], [t['start'], t['end']], t['score']])
            if idx < len(self.nlp_res) - 1:
                if self.nlp_res[idx + 1]['word'][:2] != self.conj:
                    c, ft = self.__make_coord_tags(tags)
                    if self.text != '':
                        word = self.text[c[0]:c[1]].strip()
                    else:
                        word = sp.strip()
                    spans.append(dict(word=word, coord=c, tags=ft))
                    sp = ""
                    tags = []
            else:
                c, ft = self.__make_coord_tags(tags)
                if self.text != '':
                    word = self.text[c[0]:c[1]].strip()
                else:
                    word = sp.strip()
                spans.append(dict(word=word, coord=c, tags=ft))
                sp = ""
                tags = []
        self.joined_tokens = spans
        return spans

    def trim_entity(self, ent):
        i = len(ent['word']) - 1
        stop_list = ['.', ',', '!', '?', ' ', '\\', '/', ]
        char_cnt = 0
        while i >= 0:
            if ent['word'][i] in stop_list:
                char_cnt += 1
            else:
                break
            i -= 1
        ent['coord'] = [ent['coord'][0], ent['coord'][1] - char_cnt]
        ent['word'] = self.text[ent['coord'][0]:ent['coord'][1]]
        return ent

    def get_final_tags(self):
        """
        Финальное объединение тэгов и токенов, убирание B-I- префиксов
        """
        assert (self.joined_bi is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод join_bi_tag()"
        # return self.joined_bi
        spans = self.joined_bi
        result = []
        i = -1
        while i < len(spans) - 1:
            i += 1
            sp = spans[i]
            s = sp
            if len(s['word']) > 0:
                s['word'] = self.text[s['coord'][0]:s['coord'][1]]
                if s['word'].lower() == 'ип' and i < len(spans) - 1:
                    if spans[i + 1]['tags'][0] == 'PER':
                        s['word'] = self.text[s['coord'][0]:spans[i + 1]['coord'][1]]
                        s['tags'] = ['ORG', spans[i + 1]['tags'][1]]
                        s['coord'] = [s['coord'][0], spans[i + 1]['coord'][1]]
                        i += 1

                if len(regex.findall(r"^(\индивидуальный предприниматель\b){e<=6}", s['word'].lower())) > 0 \
                        and i < len(spans) - 1:
                    if spans[i + 1]['tags'][0] == 'PER':
                        s['word'] = self.text[s['coord'][0]:spans[i + 1]['coord'][1]]
                        s['tags'] = ['ORG', spans[i + 1]['tags'][1]]
                        s['coord'] = [s['coord'][0], spans[i + 1]['coord'][1]]
                        i += 1

                if len(regex.findall(r"^(\индивидуальный\b){e<=4}", spans[i]['word'].lower())) > 0 and i < len(
                        spans) - 2:
                    if spans[i + 2]['tags'][0] == 'PER':
                        s['word'] = self.text[s['coord'][0]:spans[i + 2]['coord'][1]]
                        s['tags'] = ['ORG', spans[i + 2]['tags'][1]]
                        s['coord'] = [s['coord'][0], spans[i + 2]['coord'][1]]
                        i += 2
                if s['tags'][0] == 'PER' and i < len(spans) - 2:
                    if len(regex.findall(r"^(\индивидуальный\b){e<=4}", spans[i + 1]['word'].lower())) > 0:
                        s['word'] = self.text[s['coord'][0]:spans[i + 2]['coord'][1]]
                        s['tags'] = ['ORG', spans[i + 2]['tags'][1]]
                        s['coord'] = [s['coord'][0], spans[i + 2]['coord'][1]]
                        i += 2

                if s['tags'][0] == 'PER' and i < len(spans) - 1:
                    if spans[i + 1]['word'].lower() == 'ип':
                        if spans[i + 1]['tags'][0] == 'PER':
                            s['word'] = self.text[s['coord'][0]:spans[i + 1]['coord'][1]]
                            s['tags'] = ['ORG', spans[i + 1]['tags'][1]]
                            s['coord'] = [s['coord'][0], spans[i + 1]['coord'][1]]
                            i += 1
                s = self.trim_entity(s)
                result.append(s)
        self.final_tags = result
        return result


if __name__ == "__main__":
    text = 'ООО "ТД" ЭЛЕКОМ " адресу 197374 Санкт-Петербург город улица Стародеревенская дом 11 корпус 2 ЛИТЕР А ЭТАЖ 4 ОФИС 429 ИННКПП получатель 1814740712 / 781401001'
    token_text = ['[CLS]', 'инвестор', 'му', '##п', 'м', '##п', 'городского', 'округа', 'сама', '##ра', 'красног',
                  '##линс', '##кие', 'бани', 'адрес', 'филиала', 'банка', 'брян', '##ская', 'область', ',', 'город',
                  'ново', '##зыб', '##ков', 'банк', 'инвестор', 'инн', '/', 'к', '##пп', '1006', '##01', '##30', '##34',
                  '##68', '/', '632', '##501', '##001', 'тел', '.', '+', '7', '-', '830', '-', '199', '-', '803', '##1',
                  'корр', '.', 'счет', '303', '##10', '##41', '##13', '##61', '##95', '##57', '##97', '##44', '##2',
                  '[SEP]']
    indices = [7, 0, 0, 0, 3, 4, 4, 4, 6, 6, 4, 4, 4, 6, 0, 0, 0, 4, 5, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 7, 7, 7, 7,
               7, 0, 7, 7, 7, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]
    nlp_res = [{'word': 'о', 'score': 0.9998965859413147, 'entity': 'LABEL_3', 'index': 1, 'start': 0, 'end': 1}, {'word': '##оо', 'score': 0.9998134970664978, 'entity': 'LABEL_3', 'index': 2, 'start': 1, 'end': 3}, {'word': '"', 'score': 0.9999472498893738, 'entity': 'LABEL_4', 'index': 3, 'start': 4, 'end': 5}, {'word': 'тд', 'score': 0.9998692274093628, 'entity': 'LABEL_4', 'index': 4, 'start': 5, 'end': 7}, {'word': '"', 'score': 0.999881386756897, 'entity': 'LABEL_4', 'index': 5, 'start': 7, 'end': 8}, {'word': 'эл', 'score': 0.9996792674064636, 'entity': 'LABEL_4', 'index': 6, 'start': 9, 'end': 11}, {'word': '##еком', 'score': 0.9990493059158325, 'entity': 'LABEL_4', 'index': 7, 'start': 11, 'end': 15}, {'word': '"', 'score': 0.9995307326316833, 'entity': 'LABEL_4', 'index': 8, 'start': 16, 'end': 17}, {'word': 'адресу', 'score': 0.9941868185997009, 'entity': 'LABEL_0', 'index': 9, 'start': 18, 'end': 24}, {'word': '1973', 'score': 0.9890339374542236, 'entity': 'LABEL_5', 'index': 10, 'start': 25, 'end': 29}, {'word': '##74', 'score': 0.9952074885368347, 'entity': 'LABEL_6', 'index': 11, 'start': 29, 'end': 31}, {'word': 'санкт', 'score': 0.965875506401062, 'entity': 'LABEL_6', 'index': 12, 'start': 32, 'end': 37}, {'word': '-', 'score': 0.9900513291358948, 'entity': 'LABEL_6', 'index': 13, 'start': 37, 'end': 38}, {'word': 'петербург', 'score': 0.9798445701599121, 'entity': 'LABEL_6', 'index': 14, 'start': 38, 'end': 47}, {'word': 'город', 'score': 0.9998816847801208, 'entity': 'LABEL_6', 'index': 15, 'start': 48, 'end': 53}, {'word': 'улица', 'score': 0.9999939203262329, 'entity': 'LABEL_6', 'index': 16, 'start': 54, 'end': 59}, {'word': 'старо', 'score': 0.9999939799308777, 'entity': 'LABEL_6', 'index': 17, 'start': 60, 'end': 65}, {'word': '##дер', 'score': 0.9999908208847046, 'entity': 'LABEL_6', 'index': 18, 'start': 65, 'end': 68}, {'word': '##еве', 'score': 0.9999914169311523, 'entity': 'LABEL_6', 'index': 19, 'start': 68, 'end': 71}, {'word': '##нская', 'score': 0.9999916553497314, 'entity': 'LABEL_6', 'index': 20, 'start': 71, 'end': 76}, {'word': 'дом', 'score': 0.9999919533729553, 'entity': 'LABEL_6', 'index': 21, 'start': 77, 'end': 80}, {'word': '11', 'score': 0.9999897480010986, 'entity': 'LABEL_6', 'index': 22, 'start': 81, 'end': 83}, {'word': 'корпус', 'score': 0.999985933303833, 'entity': 'LABEL_6', 'index': 23, 'start': 84, 'end': 90}, {'word': '2', 'score': 0.9999898076057434, 'entity': 'LABEL_6', 'index': 24, 'start': 91, 'end': 92}, {'word': 'литер', 'score': 0.999981164932251, 'entity': 'LABEL_6', 'index': 25, 'start': 93, 'end': 98}, {'word': 'а', 'score': 0.9999885559082031, 'entity': 'LABEL_6', 'index': 26, 'start': 99, 'end': 100}, {'word': 'этаж', 'score': 0.9999849796295166, 'entity': 'LABEL_6', 'index': 27, 'start': 101, 'end': 105}, {'word': '4', 'score': 0.999992311000824, 'entity': 'LABEL_6', 'index': 28, 'start': 106, 'end': 107}, {'word': 'офис', 'score': 0.9999736547470093, 'entity': 'LABEL_6', 'index': 29, 'start': 108, 'end': 112}, {'word': '429', 'score': 0.999966561794281, 'entity': 'LABEL_6', 'index': 30, 'start': 113, 'end': 116}, {'word': 'инн', 'score': 0.9999902844429016, 'entity': 'LABEL_0', 'index': 31, 'start': 117, 'end': 120}, {'word': '##к', 'score': 0.9999526143074036, 'entity': 'LABEL_0', 'index': 32, 'start': 120, 'end': 121}, {'word': '##пп', 'score': 0.9999831318855286, 'entity': 'LABEL_0', 'index': 33, 'start': 121, 'end': 123}, {'word': 'получат', 'score': 0.9999962449073792, 'entity': 'LABEL_0', 'index': 34, 'start': 124, 'end': 131}, {'word': '##ель', 'score': 0.9999954700469971, 'entity': 'LABEL_0', 'index': 35, 'start': 131, 'end': 134}, {'word': '1814', 'score': 0.9999568462371826, 'entity': 'LABEL_7', 'index': 36, 'start': 135, 'end': 139}, {'word': '##74', 'score': 0.9998579621315002, 'entity': 'LABEL_7', 'index': 37, 'start': 139, 'end': 141}, {'word': '##0', 'score': 0.9999452233314514, 'entity': 'LABEL_7', 'index': 38, 'start': 141, 'end': 142}, {'word': '##71', 'score': 0.9999639987945557, 'entity': 'LABEL_7', 'index': 39, 'start': 142, 'end': 144}, {'word': '##2', 'score': 0.9999604225158691, 'entity': 'LABEL_7', 'index': 40, 'start': 144, 'end': 145}, {'word': '/', 'score': 0.9999924302101135, 'entity': 'LABEL_0', 'index': 41, 'start': 146, 'end': 147}, {'word': '781', 'score': 0.9999947547912598, 'entity': 'LABEL_7', 'index': 42, 'start': 148, 'end': 151}, {'word': '##401', 'score': 0.9999933838844299, 'entity': 'LABEL_7', 'index': 43, 'start': 151, 'end': 154}, {'word': '##001', 'score': 0.999994695186615, 'entity': 'LABEL_7', 'index': 44, 'start': 154, 'end': 157}]

    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-INNKPP', 'I-INNKPP', 'B-RSKS', 'I-RSKS',
                  'B-STAT', 'I-STAT']
    tokens = NerTokes(nlp_res, label_list, text)

    # получаем пары: слово-токены
    spans = tokens.join_tokens()
    # print(spans)

    # выбираем нужный токен для слова
    spans = tokens.normalize_tokens()
    # print(spans)
    spans = tokens.join_i_tag()
    # print(spans)

    spans = tokens.join_bi_tag()
    # print(spans)
    pprint(tokens.get_final_tags())
    # join_bi_tag(spans)
