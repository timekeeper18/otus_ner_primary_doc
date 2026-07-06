# -*- coding: utf-8 -*-
from collections import Counter

import regex
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer


class NerTokes:
    def __doc__(self):
        """при инициализации подается путь до изображения"""

    def __init__(self, token_text, indices, label_list, conj='##'):
        self.detokenizer = NLTKMosesTokenizer(False, 'ru').detokenizer
        self.token_text = token_text
        self.indices = indices
        self.label_list = label_list
        self.conj = conj
        self.joined_tokens = None
        self.norm_tokens = None
        self.joined_i = None
        self.joined_bi = None

    # def normalize_tokens(self):
    #     """
    #     Приводим к единому значению тэги у слова, если оно состоит из нескольких токенов и нескольких тэгов
    #     """
    #
    #     assert (self.joined_tokens is not None), "joined_tokens  еще не заполнена, сначала вызовите метод join_tokens()"
    #
    #     spans = self.joined_tokens
    #     tokens, entities = zip(*spans)
    #     new_entities = []
    #     for i, enti in enumerate(entities):
    #         if 'литлтон' in tokens[i]:
    #             print(tokens[i])
    #         if i == 0:
    #             cnt = Counter(list(filter(lambda x: x[0] == 'B', enti))).most_common(1)
    #             if len(cnt) > 0:
    #                 new_entities.append(cnt[0][0])
    #             else:
    #                 cnt = Counter(enti).most_common(1)
    #                 new_entities.append(cnt[0][0])
    #         else:
    #             if enti[0] == "O":
    #                 new_entities.append(enti[0])
    #             else:
    #
    #
    #                 if i < len(entities)-1:
    #                     if new_entities[i - 1][2:] == entities[i + 1][0][2:]:
    #                         enti[0] = f"I-{new_entities[i - 1][2:]}"
    #                         new_entities.append(enti[0])
    #                 else:
    #                     new_entities.append(enti[0])
    #                 if new_entities[i - 1][2:] in enti[0] and enti[0][0] == 'I':
    #                     new_entities.append(enti[0])
    #                 else:
    #                     new_entities.append(Counter(enti).most_common(1)[0][0])
    #
    #
    #     self.norm_tokens = list(zip(tokens, new_entities))
    #     return self.norm_tokens

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
        tokens, entities = zip(*spans)
        new_entities = []
        for i, enti in enumerate(entities):
            cnt = self.get_most_common(enti)
            #   cnt = {'B': [('B-LOC', 2)],
            #          'I': [('I-LOC', 1)],
            #          'O': [('O', 1)],
            #          'Overall': [('B-LOC', 2)]}
            # if '43209075037153306310' in tokens[i]:
            #     print(tokens[i])
            if i == 0:

                if len(cnt['Overall']) == 1:
                    new_entities.append([tokens[i], cnt['Overall'][0][0].replace('I-', 'B-')])
                elif len(cnt['Overall']) > 1:

                    cnt_1 = self.get_most_common(entities[i + 1])
                    new_ent = ""
                    for c in cnt['Overall']:
                        t0 = c[0].replace('I-', '').replace('B-', '')
                        for cc in cnt_1['Overall']:
                            tt0 = cc[0].replace('I-', '').replace('B-', '')
                            if t0 == tt0:
                                new_ent = c[0].replace('I-', 'B-').replace('I-', 'B-')
                                break
                    if new_ent != "":
                        new_entities.append([tokens[i], new_ent])
                    else:
                        if cnt['B'][0][1] > 0:
                            new_entities.append([tokens[i], cnt['B'][0][0]])
                        elif cnt['I'][0][1] > 0:
                            new_entities.append([tokens[i], cnt['I'][0][0]])
                        else:
                            new_entities.append([tokens[i], 'O'])
            else:
                if enti[0] == "O":
                    new_entities.append([tokens[i], enti[0]])
                else:
                    if i < len(entities) - 1:
                        t00 = new_entities[i - 1][1].replace('I-', '').replace('B-', '')

                        cnt_1 = self.get_most_common(entities[i + 1])
                        new_ent = ""
                        if t00 != 'O':
                            for c in cnt_1['Overall']:
                                t0 = c[0].replace('I-', '').replace('B-', '')
                                if t00 == t0:
                                    new_ent = f"I-{t00}"
                            if new_ent != "":
                                new_entities.append([tokens[i], new_ent])
                            else:
                                new_entities.append([tokens[i], enti[0]])
                        else:
                            new_entities.append([tokens[i], enti[0]])
                    else:
                        if new_entities[i - 1][1][2:] in enti[0] and enti[0][0] == 'I':
                            new_entities.append([tokens[i], enti[0]])
                        else:
                            new_entities.append([tokens[i], cnt['Overall'][0][0]])

        #     norm_tokens = list(zip(tokens, new_entities))
        self.norm_tokens = new_entities
        return new_entities

    def join_i_tag(self):
        """
        Объединяем последовательно идущие Тэги I-
        """
        assert (self.norm_tokens is not None),\
            "joined_tokens  еще не заполнена, сначала вызовите метод normalize_tokens()"

        spans = self.norm_tokens
        tokens, entities = zip(*spans)
        st = ['', '']
        ent = []
        result = []
        for i, enti in enumerate(entities):
            enti = entities[i]
            e0 = enti[0]

            if e0 in ['I', 'O']:
                if st[1] == enti:
                    st[0] += f' {tokens[i]}'
                    st[1] = enti
                else:
                    if len(st[0]) > 0:
                        st[0] = st[0].strip()
                        result.append(st)
                    st = [tokens[i], enti]
            if e0 in ['B']:
                if len(st[0]) > 0:
                    result.append(st)
                    st = [tokens[i], enti]
                else:
                    result.append([tokens[i].strip(), enti])
        if len(st[0]) > 0:
            if len(result) == 0:
                result.append(st)
            elif st[1] != result[-1][0]:
                result.append(st)
        self.joined_i = result
        return self.joined_i

    def join_bi_tag(self):

        assert (self.joined_i is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод join_i_tag()"
        spans = self.joined_i
        tokens, entities = zip(*spans)
        result = []

        # Нормализуем Теги "B-", если слово начинается с тега "i-"
        # и перед ним нет аналогичного тега "B-", то присвоить ему тег "B-"
        for i, enti in enumerate(entities):
            e0 = enti[0]
            # if 'литлтон' in tokens[i]:
            #     print(tokens[i])
            if e0 == 'I':
                if i == 0:
                    result.append([tokens[i], enti.replace('I-', 'B-')])
                else:
                    if entities[i - 1][2:] == enti[2:]:
                        result.append([tokens[i], enti])
                    else:
                        result.append([tokens[i], enti.replace('I-', 'B-')])
            else:
                result.append([tokens[i], enti])

        tokens, entities = zip(*result)
        st = ['', '']
        result = []

        for i, enti in enumerate(entities):
            e0 = enti[0]
            if e0 in ['B']:
                if i < len(entities)-1 and entities[i + 1][0] == 'I':
                    if entities[i + 1][2:] == enti[2:]:
                        st[0] += ' ' + tokens[i]
                        st[1] = enti
                        continue
                else:
                    if len(st[0]) > 0:
                        if st[1][2:] == enti[2:]:
                            st[0] += ' ' + tokens[i]
                            result.append(st)
                            st = ['', '']
                    else:
                        result.append([tokens[i], enti])
            elif e0 in ['I']:
                if len(st[0]) > 0:
                    if st[1][2:] == enti[2:]:
                        st[0] += ' ' + tokens[i]
                        result.append(st)
                        st = ['', '']
                    else:
                        st[0] = st[0].strip()
                        result.append(st)
                        st = ['', '']
                    continue
                else:
                    if entities[i + 1][0] == 'B':
                        if entities[i + 1][2:] == enti[2:]:
                            st[0] += ' ' + tokens[i]
                            st[1] = entities[i + 1]
                            continue

            elif e0 in ['O']:
                result.append([tokens[i], enti])
                continue
        self.joined_bi = result
        return self.joined_bi

    @staticmethod
    def __get_single_tag(tags):
        """
        Предварительный выбор тэга
        """
        if len(set(tags)) == 1:
            return [tags[0]] #if tags[0] != 'O' else [tags[0]]
        # if 'O' in tags:
        #     return ['O']
        return tags  # list(map(lambda x: x[2:] if x[0] != 'O' else x[0], tags)) #Counter(tags).most_common(1)[0][0]

    def join_tokens(self):
        """
        Объединяем токены в слова и соединяем с тэгами
        """
        sp = ""
        spans = []
        tags = []
        i = 0

        for t, idx in zip(self.token_text, self.indices):
            if t not in ['[CLS]', '[SEP]']:
                if t[:2] == self.conj:
                    sp += t[2:]
                    tags.append(self.label_list[idx])
                else:
                    sp += f' {t}'
                    tags.append(self.label_list[idx])
                if i < len(self.token_text) - 1:
                    if self.token_text[i + 1][:2] != self.conj:
                        spans.append([sp.strip(), self.__get_single_tag(tags)])
                        sp = ""
                        tags = []
            i += 1
        self.joined_tokens = spans
        return spans

    def get_final_tags(self):
        """
        Финальное объединение тэгов и токенов, убирание B-I- префиксов
        """
        assert (self.joined_bi is not None), \
            "joined_tokens  еще не заполнена, сначала вызовите метод join_bi_tag()"

        spans = self.joined_bi
        tokens, entities = zip(*spans)
        result = []
        i = -1

        while i < len(entities) - 1:
            i += 1
            enti = entities[i]
            if enti == 'B-ORG' and (
                    len(regex.findall(r"^(\bиндивидуальный\b){e<=4}", tokens[i])) > 0 or tokens[i] == 'ип'):
                if i < len(entities) - 2:
                    if entities[i + 1] == 'B-PER':
                        st = tokens[i] + tokens[i + 1]
                        result.append([enti[2:], self.detokenizer.detokenize([''.join(c) for c in st.split()])])
                        i += 1
            else:
                if i < len(entities) - 2:
                    if len(tokens[i]) <= 2 and enti == entities[i + 1]:
                        st = tokens[i] + tokens[i + 1]
                        result.append([enti[2:] if enti != 'O' else 'O',
                                       self.detokenizer.detokenize([''.join(c) for c in st.split()])])
                        i += 1
                    else:
                        result.append([enti[2:] if enti != 'O' else 'O',
                                       self.detokenizer.detokenize([''.join(c) for c in tokens[i].split()])])
                else:
                    result.append([enti[2:] if enti != 'O' else 'O',
                                   self.detokenizer.detokenize([''.join(c) for c in tokens[i].split()])])
        return result


if __name__ == "__main__":
    token_text = ['[CLS]', 'as', '##df', '##as', '##df', '##s', '[SEP]']
    indices = [7, 0, 0, 0, 0, 0, 4]

    label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-INNKPP', 'I-INNKPP', 'B-RSKS', 'I-RSKS',
                  'B-STAT', 'I-STAT']
    tokens = NerTokes(token_text, indices, label_list)

    # получаем пары: слово-токены
    spans = tokens.join_tokens()
    print(spans)

    # выбираем нужный токен для слова
    spans = tokens.normalize_tokens()
    # print(spans)
    spans = tokens.join_i_tag()
    # print(spans)

    spans = tokens.join_bi_tag()
    # print(spans)

    print(tokens.get_final_tags())
    # join_bi_tag(spans)
