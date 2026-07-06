# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
import regex
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

            #   cnt = {'B': [('B-LOC', 2)],
            #          'I': [('I-LOC', 1)],
            #          'O': [('O', 1)],
            #          'Overall': [('B-LOC', 2)]}
            # if '43209075037153306310' in tokens[i]:
            #     print(tokens[i])
            # if enti['word'] == '9076':
            #     print('1')

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
                            for i in enti['tags']:
                                if i[0][0] == 'B':
                                    enti['tags'] = [i[0], i[1]]
                                    new_entities.append(enti)
                                    b = 1
                                    break
                            if b == 0:
                                for i in enti['tags']:
                                    if i[0][0] == 'I':
                                        enti['tags'] = [i[0], i[1]]
                                        new_entities.append(enti)
                                        b = 1
                                        break
                            if b == 0:
                                enti['tags'] = [enti['tags'][0][0].replace('I-', 'B-'), enti['tags'][0][1]]
                                new_entities.append(enti)
            else:
                if enti['tags'][0][1] >= self.confidences: #or len(enti['tags']) == 1:
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
        self.joined_i = result
        return self.joined_i

    def join_bi_tag(self):

        assert (self.joined_i is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод join_i_tag()"
        spans = self.joined_i
        # Нормализуем Теги "B-", если слово начинается с тега "i-"
        # и перед ним нет аналогичного тега "B-", то присвоить ему тег "B-"
        st = {'word': '', 'coord': [], 'tags': ['', 0]}
        result = []

        for i, enti in enumerate(spans):
            # if enti['word'] == '9076':
            #     print(enti)
            e0 = enti['tags'][0][0]
            if e0 == 'I':
                if i == 0:
                    enti['tags'][0] = ['tags'][0].replace('I-', 'B-')
                    result.append(enti)
                else:
                    if spans[i - 1]['tags'][0][2:] == enti['tags'][0][2:]:
                        result.append(enti)
                    else:
                        enti['tags'][0] = enti['tags'][0].replace('I-', 'B-')
                        result.append(enti)
            else:
                result.append(enti)
        res_spans = result
        st = {'word': '', 'coord': [], 'tags': ['', 0]}
        result = []

        for i, enti in enumerate(res_spans):
            e0 = enti['tags'][0][0]
            if e0 in ['B']:
                if i < len(res_spans) - 1 and res_spans[i + 1]['tags'][0][0] == 'I':
                    if res_spans[i + 1]['tags'][0][2:] == enti['tags'][0][2:]:
                        st['word'] += ' ' + enti['word']
                        if len(st['coord']) > 0:
                            st['coord'] = [st['coord'][0], enti['coord'][1]]
                        else:
                            st['coord'] = [enti['coord'][0], enti['coord'][1]]
                        st['tags'] = enti['tags']
                        continue
                else:
                    if len(st['word']) > 0:
                        if st['tags'][0][2:] == enti[2:]:
                            st['word'] += ' ' + enti['word']
                            if len(st['coord']) > 0:
                                st['coord'] = [st['coord'][0], enti['coord'][1]]
                            else:
                                st['coord'] = [enti['coord'][0], enti['coord'][1]]
                            st['tags'] = enti['tags']
                            st = {'word': '', 'coord': [], 'tags': ['', 0]}
                    else:
                        result.append(enti)
            elif e0 in ['I']:
                if len(st['word']) > 0:
                    if st['tags'][0][2:] == enti['tags'][0][2:]:
                        st['word'] += ' ' + enti['word']
                        if len(st['coord']) > 0:
                            st['coord'] = [st['coord'][0], enti['coord'][1]]
                        else:
                            st['coord'] = [enti['coord'][0], enti['coord'][1]]
                        #                 st['tags'] = enti['tags']
                        result.append(st)
                        st = {'word': '', 'coord': [], 'tags': ['', 0]}
                    else:
                        st['word'] = st['word'].strip()
                        result.append(st)
                        st = {'word': '', 'coord': [], 'tags': ['', 0]}
                    continue
                else:
                    if i <= len(res_spans) - 2:
                        if res_spans[i + 1]['tags'][0][0] == 'B':
                            if res_spans[i + 1]['tags'][0][2:] == enti['tags'][0][2:]:
                                st['word'] += ' ' + enti['word']

                                if len(st['coord']) > 0:
                                    st['coord'] = [st['coord'][0], res_spans[i + 1]['coord'][1]]
                                else:
                                    st['coord'] = [res_spans[i + 1]['coord'][0], res_spans[i + 1]['coord'][1]]
                                st['tags'] = enti['tags']
                                st['tags'] = res_spans[i + 1]['tags']
                                continue
                    else:
                        if res_spans[i - 1]['tags'][0][2:] == enti['tags'][0][2:]:
                            st['word'] += ' ' + enti['word']

                            if len(st['coord']) > 0:
                                st['coord'] = [st['coord'][0], res_spans[i]['coord'][1]]
                            else:
                                st['coord'] = [res_spans[i - 1]['coord'][0], res_spans[i]['coord'][1]]

                            st['tags'] = res_spans[i - 1]['tags']
                            result.append(st)
                            st = {'word': '', 'coord': [], 'tags': ['', 0]}
                            continue
                        else:
                            st['word'] = enti['word']
                            st['tags'] = enti['tags']
                            st['coord'] = [res_spans[i]['coord'][0], res_spans[i]['coord'][1]]
                            result.append(st)
                            st = {'word': '', 'coord': [], 'tags': ['', 0]}
                            print(res_spans[i])

            elif e0 in ['O']:
                result.append(enti)
                continue
        self.joined_bi = result
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

    def get_final_tags(self):
        """
        Финальное объединение тэгов и токенов, убирание B-I- префиксов
        """
        assert (self.joined_bi is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод join_bi_tag()"

        spans = self.joined_bi
        result = []
        st = {'word': '', 'coord': [], 'tags': ['', 0]}
        i = -1
        while i < len(spans) - 1:
            i += 1
            enti = spans[i]['tags'][0]
            if i <= len(spans) - 2:
                if enti == 'B-ORG' and (
                        len(regex.findall(r"^(\bиндивидуальный\b){e<=4}", spans[i]['word'].lower())) > 0 or spans[i][
                    'word'].lower() == 'ип'):
                    if spans[i + 1]['tags'][0] == 'B-PER':
                        st['coord'] = [spans[i]['coord'][0], spans[i + 1]['coord'][1]]
                        st['word'] = self.text[st['coord'][0]:st['coord'][1]]
                        st['tags'] = [enti[2:], spans[i]['tags'][1]]
                        result.append(st)
                        i += 1
                        st = {'word': '', 'coord': [], 'tags': ['', 0]}
                else:
                    if len(st['word']) <= 2 and enti == spans[i + 1]['tags'][0]:
                        st['coord'] = [spans[i]['coord'][0], spans[i + 1]['coord'][1]]
                        st['word'] = self.text[st['coord'][0]:st['coord'][1]]
                        st['tags'] = [enti[2:] if enti != 'O' else 'O', spans[i]['tags'][1]]
                        result.append(st)
                        st = {'word': '', 'coord': [], 'tags': ['', 0]}
                        i += 1
                    else:
                        st['coord'] = spans[i]['coord']
                        st['word'] = self.text[st['coord'][0]:st['coord'][1]]
                        st['tags'] = [enti[2:] if enti != 'O' else 'O', spans[i]['tags'][1]]
                        result.append(st)
                        st = {'word': '', 'coord': [], 'tags': ['', 0]}
            else:
                st['coord'] = spans[i]['coord']
                st['word'] = self.text[st['coord'][0]:st['coord'][1]]
                st['tags'] = [enti[2:] if enti != 'O' else 'O', spans[i]['tags'][1]]
                result.append(st)
        self.final_tags = result
        return result


if __name__ == "__main__":
    text = 'Заказчик АО ТОРГТЕХНИКА 7703405973 540601001 Банк Поставщик ООО "ЭСИДБАНК" ИНН КПП Получатель 7713586119 / 691301001 190098, г Санкт-Петербург, Адмиралтейский р-н, пл Труда, д 4 тел. +8 (958) 392 21 72 тел. +7-341-105-9076'
    token_text = ['[CLS]', 'инвестор', 'му', '##п', 'м', '##п', 'городского', 'округа', 'сама', '##ра', 'красног',
                  '##линс', '##кие', 'бани', 'адрес', 'филиала', 'банка', 'брян', '##ская', 'область', ',', 'город',
                  'ново', '##зыб', '##ков', 'банк', 'инвестор', 'инн', '/', 'к', '##пп', '1006', '##01', '##30', '##34',
                  '##68', '/', '632', '##501', '##001', 'тел', '.', '+', '7', '-', '830', '-', '199', '-', '803', '##1',
                  'корр', '.', 'счет', '303', '##10', '##41', '##13', '##61', '##95', '##57', '##97', '##44', '##2',
                  '[SEP]']
    indices = [7, 0, 0, 0, 3, 4, 4, 4, 6, 6, 4, 4, 4, 6, 0, 0, 0, 4, 5, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 7, 7, 7, 7,
               7, 0, 7, 7, 7, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]
    nlp_res = [{'word': 'заказчик', 'score': 0.9996680617332458, 'entity': 'LABEL_0', 'index': 1, 'start': 0, 'end': 8}, {'word': 'а', 'score': 0.9741144776344299, 'entity': 'LABEL_3', 'index': 2, 'start': 9, 'end': 10}, {'word': '##о', 'score': 0.8248963356018066, 'entity': 'LABEL_3', 'index': 3, 'start': 10, 'end': 11}, {'word': 'торг', 'score': 0.7867106795310974, 'entity': 'LABEL_4', 'index': 4, 'start': 12, 'end': 16}, {'word': '##техника', 'score': 0.8334603905677795, 'entity': 'LABEL_0', 'index': 5, 'start': 16, 'end': 23}, {'word': '770', 'score': 0.9999595284461975, 'entity': 'LABEL_7', 'index': 6, 'start': 24, 'end': 27}, {'word': '##34', 'score': 0.9996466040611267, 'entity': 'LABEL_7', 'index': 7, 'start': 27, 'end': 29}, {'word': '##0', 'score': 0.9981418251991272, 'entity': 'LABEL_7', 'index': 8, 'start': 29, 'end': 30}, {'word': '##59', 'score': 0.9964417219161987, 'entity': 'LABEL_7', 'index': 9, 'start': 30, 'end': 32}, {'word': '##73', 'score': 0.9973393082618713, 'entity': 'LABEL_7', 'index': 10, 'start': 32, 'end': 34}, {'word': '540', 'score': 0.9996638894081116, 'entity': 'LABEL_7', 'index': 11, 'start': 35, 'end': 38}, {'word': '##601', 'score': 0.9999940395355225, 'entity': 'LABEL_7', 'index': 12, 'start': 38, 'end': 41}, {'word': '##001', 'score': 0.9999386072158813, 'entity': 'LABEL_7', 'index': 13, 'start': 41, 'end': 44}, {'word': 'банк', 'score': 0.9905232191085815, 'entity': 'LABEL_0', 'index': 14, 'start': 45, 'end': 49}, {'word': 'поставщик', 'score': 0.9774523973464966, 'entity': 'LABEL_0', 'index': 15, 'start': 50, 'end': 59}, {'word': 'о', 'score': 0.7250691056251526, 'entity': 'LABEL_3', 'index': 16, 'start': 60, 'end': 61}, {'word': '##оо', 'score': 0.5354593396186829, 'entity': 'LABEL_4', 'index': 17, 'start': 61, 'end': 63}, {'word': '"', 'score': 0.9864862561225891, 'entity': 'LABEL_4', 'index': 18, 'start': 64, 'end': 65}, {'word': 'эс', 'score': 0.9981212615966797, 'entity': 'LABEL_4', 'index': 19, 'start': 65, 'end': 67}, {'word': '##ид', 'score': 0.9978916049003601, 'entity': 'LABEL_4', 'index': 20, 'start': 67, 'end': 69}, {'word': '##банк', 'score': 0.9996762275695801, 'entity': 'LABEL_4', 'index': 21, 'start': 69, 'end': 73}, {'word': '"', 'score': 0.9993371367454529, 'entity': 'LABEL_4', 'index': 22, 'start': 73, 'end': 74}, {'word': 'инн', 'score': 0.9940195083618164, 'entity': 'LABEL_0', 'index': 23, 'start': 75, 'end': 78}, {'word': 'к', 'score': 0.980684220790863, 'entity': 'LABEL_0', 'index': 24, 'start': 79, 'end': 80}, {'word': '##пп', 'score': 0.9790560007095337, 'entity': 'LABEL_0', 'index': 25, 'start': 80, 'end': 82}, {'word': 'получат', 'score': 0.9991772174835205, 'entity': 'LABEL_0', 'index': 26, 'start': 83, 'end': 90}, {'word': '##ель', 'score': 0.9980520009994507, 'entity': 'LABEL_0', 'index': 27, 'start': 90, 'end': 93}, {'word': '771', 'score': 0.9999130368232727, 'entity': 'LABEL_7', 'index': 28, 'start': 94, 'end': 97}, {'word': '##35', 'score': 0.9994072318077087, 'entity': 'LABEL_7', 'index': 29, 'start': 97, 'end': 99}, {'word': '##86', 'score': 0.9993639588356018, 'entity': 'LABEL_7', 'index': 30, 'start': 99, 'end': 101}, {'word': '##11', 'score': 0.9988710284233093, 'entity': 'LABEL_7', 'index': 31, 'start': 101, 'end': 103}, {'word': '##9', 'score': 0.998989999294281, 'entity': 'LABEL_7', 'index': 32, 'start': 103, 'end': 104}, {'word': '/', 'score': 0.9988253712654114, 'entity': 'LABEL_0', 'index': 33, 'start': 105, 'end': 106}, {'word': '691', 'score': 0.9997431039810181, 'entity': 'LABEL_7', 'index': 34, 'start': 107, 'end': 110}, {'word': '##301', 'score': 0.9985347390174866, 'entity': 'LABEL_7', 'index': 35, 'start': 110, 'end': 113}, {'word': '##001', 'score': 0.9995242953300476, 'entity': 'LABEL_7', 'index': 36, 'start': 113, 'end': 116}, {'word': '1900', 'score': 0.7706076502799988, 'entity': 'LABEL_5', 'index': 37, 'start': 117, 'end': 121}, {'word': '##98', 'score': 0.9992072582244873, 'entity': 'LABEL_5', 'index': 38, 'start': 121, 'end': 123}, {'word': ',', 'score': 0.9834540486335754, 'entity': 'LABEL_6', 'index': 39, 'start': 123, 'end': 124}, {'word': 'г', 'score': 0.9983919858932495, 'entity': 'LABEL_6', 'index': 40, 'start': 125, 'end': 126}, {'word': 'санкт', 'score': 0.9919406175613403, 'entity': 'LABEL_6', 'index': 41, 'start': 127, 'end': 132}, {'word': '-', 'score': 0.997170090675354, 'entity': 'LABEL_6', 'index': 42, 'start': 132, 'end': 133}, {'word': 'петербург', 'score': 0.9994142055511475, 'entity': 'LABEL_6', 'index': 43, 'start': 133, 'end': 142}, {'word': ',', 'score': 0.9999721050262451, 'entity': 'LABEL_6', 'index': 44, 'start': 142, 'end': 143}, {'word': 'адмиралт', 'score': 0.9974466562271118, 'entity': 'LABEL_6', 'index': 45, 'start': 144, 'end': 152}, {'word': '##еи', 'score': 0.9975752830505371, 'entity': 'LABEL_6', 'index': 46, 'start': 152, 'end': 154}, {'word': '##ски', 'score': 0.992323100566864, 'entity': 'LABEL_6', 'index': 47, 'start': 154, 'end': 157}, {'word': '##и', 'score': 0.9969121217727661, 'entity': 'LABEL_6', 'index': 48, 'start': 157, 'end': 158}, {'word': 'р', 'score': 0.9999811053276062, 'entity': 'LABEL_6', 'index': 49, 'start': 159, 'end': 160}, {'word': '-', 'score': 0.9999118447303772, 'entity': 'LABEL_6', 'index': 50, 'start': 160, 'end': 161}, {'word': 'н', 'score': 0.9999024868011475, 'entity': 'LABEL_6', 'index': 51, 'start': 161, 'end': 162}, {'word': ',', 'score': 0.9999988079071045, 'entity': 'LABEL_6', 'index': 52, 'start': 162, 'end': 163}, {'word': 'пл', 'score': 0.9999964237213135, 'entity': 'LABEL_6', 'index': 53, 'start': 164, 'end': 166}, {'word': 'труда', 'score': 0.9999191164970398, 'entity': 'LABEL_6', 'index': 54, 'start': 167, 'end': 172}, {'word': ',', 'score': 0.9999983906745911, 'entity': 'LABEL_6', 'index': 55, 'start': 172, 'end': 173}, {'word': 'д', 'score': 0.9999991059303284, 'entity': 'LABEL_6', 'index': 56, 'start': 174, 'end': 175}, {'word': '4', 'score': 0.9997225999832153, 'entity': 'LABEL_6', 'index': 57, 'start': 176, 'end': 177}, {'word': 'тел', 'score': 0.9784663915634155, 'entity': 'LABEL_0', 'index': 58, 'start': 178, 'end': 181}, {'word': '.', 'score': 0.9814109206199646, 'entity': 'LABEL_0', 'index': 59, 'start': 181, 'end': 182}, {'word': '+', 'score': 0.9999148845672607, 'entity': 'LABEL_11', 'index': 60, 'start': 183, 'end': 184}, {'word': '8', 'score': 0.999568521976471, 'entity': 'LABEL_11', 'index': 61, 'start': 184, 'end': 185}, {'word': '(', 'score': 0.9470010995864868, 'entity': 'LABEL_12', 'index': 62, 'start': 186, 'end': 187}, {'word': '958', 'score': 0.9977148175239563, 'entity': 'LABEL_12', 'index': 63, 'start': 187, 'end': 190}, {'word': ')', 'score': 0.9997919201850891, 'entity': 'LABEL_12', 'index': 64, 'start': 190, 'end': 191}, {'word': '392', 'score': 0.9997766613960266, 'entity': 'LABEL_12', 'index': 65, 'start': 192, 'end': 195}, {'word': '21', 'score': 0.9998475909233093, 'entity': 'LABEL_12', 'index': 66, 'start': 196, 'end': 198}, {'word': '72', 'score': 0.9444999098777771, 'entity': 'LABEL_12', 'index': 67, 'start': 199, 'end': 201}, {'word': 'тел', 'score': 0.9899634718894958, 'entity': 'LABEL_0', 'index': 68, 'start': 202, 'end': 205}, {'word': '.', 'score': 0.8844606280326843, 'entity': 'LABEL_0', 'index': 69, 'start': 205, 'end': 206}, {'word': '+', 'score': 0.9999736547470093, 'entity': 'LABEL_11', 'index': 70, 'start': 207, 'end': 208}, {'word': '7', 'score': 0.9999555945396423, 'entity': 'LABEL_11', 'index': 71, 'start': 208, 'end': 209}, {'word': '-', 'score': 0.9915403723716736, 'entity': 'LABEL_11', 'index': 72, 'start': 209, 'end': 210}, {'word': '341', 'score': 0.994165301322937, 'entity': 'LABEL_11', 'index': 73, 'start': 210, 'end': 213}, {'word': '-', 'score': 0.9959147572517395, 'entity': 'LABEL_11', 'index': 74, 'start': 213, 'end': 214}, {'word': '105', 'score': 0.9964866638183594, 'entity': 'LABEL_11', 'index': 75, 'start': 214, 'end': 217}, {'word': '-', 'score': 0.8581590056419373, 'entity': 'LABEL_11', 'index': 76, 'start': 217, 'end': 218}, {'word': '907', 'score': 0.5228016972541809, 'entity': 'LABEL_12', 'index': 77, 'start': 218, 'end': 221}, {'word': '##6', 'score': 0.6314551830291748, 'entity': 'LABEL_12', 'index': 78, 'start': 221, 'end': 222}]

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
    print(tokens.get_final_tags())
    # join_bi_tag(spans)
