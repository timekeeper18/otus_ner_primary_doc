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
                if enti['tags'][0][1] >= self.confidences:  # or len(enti['tags']) == 1:
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
                if enti == 'O' and (
                        len(regex.findall(r"^(\bпредприниматель\b){e<=4}", spans[i]['word'].lower())) > 0 or spans[i][
                    'word'].lower() == 'ип'):
                    if spans[i + 1]['tags'][0] == 'B-PER':
                        st['coord'] = [spans[i]['coord'][0], spans[i + 1]['coord'][1]]
                        st['word'] = self.text[st['coord'][0]:st['coord'][1]]
                        st['tags'] = ['ORG', spans[i]['tags'][1]]
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
    text = 'ИП Панькова В. Б. ООО "Гарант-СВ"'
    token_text = ['[CLS]', 'инвестор', 'му', '##п', 'м', '##п', 'городского', 'округа', 'сама', '##ра', 'красног',
                  '##линс', '##кие', 'бани', 'адрес', 'филиала', 'банка', 'брян', '##ская', 'область', ',', 'город',
                  'ново', '##зыб', '##ков', 'банк', 'инвестор', 'инн', '/', 'к', '##пп', '1006', '##01', '##30', '##34',
                  '##68', '/', '632', '##501', '##001', 'тел', '.', '+', '7', '-', '830', '-', '199', '-', '803', '##1',
                  'корр', '.', 'счет', '303', '##10', '##41', '##13', '##61', '##95', '##57', '##97', '##44', '##2',
                  '[SEP]']
    indices = [7, 0, 0, 0, 3, 4, 4, 4, 6, 6, 4, 4, 4, 6, 0, 0, 0, 4, 5, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 7, 7, 7, 7,
               7, 0, 7, 7, 7, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]
    nlp_res = [{'word': 'ип', 'score': 0.9960789084434509, 'entity': 'LABEL_3', 'index': 1, 'start': 0, 'end': 2}, {'word': 'пан', 'score': 0.9990836977958679, 'entity': 'LABEL_1', 'index': 2, 'start': 3, 'end': 6}, {'word': '##ько', 'score': 0.9988061189651489, 'entity': 'LABEL_1', 'index': 3, 'start': 6, 'end': 9}, {'word': '##ва', 'score': 0.9999367594718933, 'entity': 'LABEL_1', 'index': 4, 'start': 9, 'end': 11}, {'word': 'в', 'score': 0.9999921321868896, 'entity': 'LABEL_2', 'index': 5, 'start': 12, 'end': 13}, {'word': '.', 'score': 0.9999943971633911, 'entity': 'LABEL_2', 'index': 6, 'start': 13, 'end': 14}, {'word': 'б', 'score': 0.9999589323997498, 'entity': 'LABEL_2', 'index': 7, 'start': 15, 'end': 16}, {'word': '.', 'score': 0.9999861717224121, 'entity': 'LABEL_2', 'index': 8, 'start': 16, 'end': 17}, {'word': 'о', 'score': 0.8441886901855469, 'entity': 'LABEL_3', 'index': 9, 'start': 18, 'end': 19}, {'word': '##оо', 'score': 0.5218430161476135, 'entity': 'LABEL_2', 'index': 10, 'start': 19, 'end': 21}, {'word': '"', 'score': 0.9994028806686401, 'entity': 'LABEL_4', 'index': 11, 'start': 22, 'end': 23}, {'word': 'гарант', 'score': 0.9933186769485474, 'entity': 'LABEL_4', 'index': 12, 'start': 23, 'end': 29}, {'word': '-', 'score': 0.9995629787445068, 'entity': 'LABEL_4', 'index': 13, 'start': 29, 'end': 30}, {'word': 'св', 'score': 0.7986558079719543, 'entity': 'LABEL_4', 'index': 14, 'start': 30, 'end': 32}, {'word': '"', 'score': 0.754765510559082, 'entity': 'LABEL_4', 'index': 15, 'start': 32, 'end': 33}]

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
