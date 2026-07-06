# -*- coding: utf-8 -*-
from web_service.src import NerTokes_Pipe
from transformers import AutoModelForTokenClassification, AutoTokenizer
from fuzzysearch import find_near_matches
from transformers import pipeline
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer
detokenizer = NLTKMosesTokenizer('ru').detokenizer
import re
import operator


class Ner:
    def __doc__(self):
        """Получение данных от модели NER и их обработка"""

    def __init__(self, model_path, label_list, conj='##', model_conf=0.3):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.conj = conj
        self.model_conf = model_conf
        self.label_list = label_list
        self.tokens = None
        self.indices = None
        self.token_text = None
        self.spans = None
        self.text = None
        self.pipe_res = None
        self.spans_coord = []
        self.tags = []
        self.confidence = []
        self.pipe = pipeline('ner', model=self.model, tokenizer=self.tokenizer)

    def pipeline(self, text):
        self.tokens = None
        self.indices = None
        self.token_text = None
        self.spans = None
        self.spans_coord = []
        self.tags = []
        self.confidence = []

        self.text = text
        self.pipe_res = self.pipe(self.text)
        return self.pipe_res

    def get_spans(self, text, dict=False):
        """
        Получаем спаны: соответствие подстроки и тега
        :param text: тестируемый текст
        :return: [['O', 'подстрока 1'], ['ORG', 'подстрока 3'], ['O', 'подстрока 3']]
        """
        self.text = text
        self.pipeline(text)
        tokens = NerTokes_Pipe(self.pipe_res, self.label_list, self.text, '##',self.model_conf)

        # получаем пары: слово-токены
        tokens.join_tokens()
        # выбираем нужный токен для слова
        tokens.normalize_tokens()
        # объединяем I-токены
        tokens.join_i_tag()
        # объединяем BI-токены
        tokens.join_bi_tag()
        # финальные теги
        self.spans = tokens.get_final_tags()

        self.__spans_decomposition()
        if dict:
            return self.spans
        else:
            sp = []
            for s in self.spans:
                sp.append([s['tags'][0], s['word']])
            return sp

    def __spans_decomposition(self):
        res = [list(sp.values()) for sp in self.spans]
        self.tokens, spans_coord, tags = zip(*res)
        for t in zip(spans_coord, tags):
            if t[1][0] != 'O':
                self.confidence.append(t[1][1])
                self.tags.append(t[1][0])
                self.spans_coord.append((int(t[0][0]), int(t[0][1]), t[1][0]))

    @staticmethod
    def get_span_coordinates_any(spans, text, dp=False):
        """
        Получаем координаты спанов в исходном тексте, необходимо для посветки в тетрадке методом show_box_markup
        (смотри ipymarkup от natasha)
        :return: [(24, 40, 'ORG')]
        """
        spans_coord = []
        if dp:
            txt, _ = prepare_txt(text.lower())
        else:
            txt = text.lower().replace('й', 'и')
        for t in filter(lambda x: x[0] != 'O', spans):
            if dp:
                tok, _ = prepare_txt(t[1].lower())
            else:
                tok = t[1].lower().replace('й', 'и')

            for m in find_near_matches(tok, txt, max_l_dist=2):  # , max_deletions=1, max_insertions=1
                #             if "".join(m.matched.split()).replace('й', 'и') == "".join(t[1].lower().replace('й', 'и').split()):
                if dp:
                    mm, _ = prepare_txt("".join(m.matched.split()).lower())
                else:
                    mm = "".join(m.matched.split()).replace('й', 'и')
                #             print(mm)
                if mm == "".join(tok.split()):
                    spans_coord.append((m.start, m.end, t[0]))
        return spans_coord


def check_case(txt):
    s = ''
    for w in txt.split():
        if w.lower() in ['ооо', 'пао', 'ао', 'зао', 'ип', 'оао', 'нко', 'инн', 'кпп', 'иннкпп', 'огрн', 'тд', 'тфд']:
            s += f' {w.upper()}'
        else:
            s += f' {w.title()}'
    return detokenizer.detokenize([s])  # ' '.join(detokenizer.detokenize([s]))


def replace_position(txt, re_comp, re_repl, replace_list):
    ignorecase = re.compile(re_comp, re.IGNORECASE)
    r_all = re.finditer(ignorecase, txt)
    if r_all is not None:
        for r in r_all:
            replace_list.append([re_repl.strip(), r.group(0), r.start(), r.end()])
    # txt = re.sub(ignorecase, re_repl, txt)
    return replace_list


def replace_org_forms(txt):
    if len(txt) == 0:
        return txt, []
    else:
        replace_list = []
        replace_list = replace_position(txt, '(закрытое)', 'З', replace_list)
        replace_list = replace_position(txt, '(общество)|(с ограниченной)|(ответственностью)|(открытое)', 'О',
                                             replace_list)
        replace_list = replace_position(txt, '(публичное)', 'П', replace_list)
        replace_list = replace_position(txt, '(акционерное)', 'А', replace_list)
        replace_list = replace_position(txt, '(индивидуальный)', 'И', replace_list)
        replace_list = replace_position(txt, '(предприниматель)', 'П', replace_list)

        if len(replace_list) > 0:
            replace_list = sorted(replace_list, key=operator.itemgetter(2))

            R = []
            R_ind = []
            for r in replace_list:
                if len(R_ind) == 0 or R_ind[-1][3] + 1 == r[2]:
                    R_ind.append(r)
                else:
                    short = ''.join([i[0] for i in R_ind])
                    long = ' '.join([i[1] for i in R_ind])
                    start_pos = R_ind[0][2]
                    R_ind_str = [short, long, start_pos, start_pos + len(long)]
                    R.append(R_ind_str)
                    R_ind = [r]
            short = ''.join([i[0] for i in R_ind])
            long = ' '.join([i[1] for i in R_ind])
            start_pos = R_ind[0][2]
            R_ind_str = [short, long, start_pos, start_pos + len(long)]
            R.append(R_ind_str)

            txt_new = ''
            for r in range(len(R)):
                if r != len(R) - 1:
                    R[r].extend([len(txt_new), len(txt_new) + len(R[r][0])])
                    txt_new += R[r][0] + txt[R[r][3]:R[r + 1][2]]
                else:
                    R[r].extend([len(txt_new), len(txt_new) + len(R[r][0])])
                    txt_new += R[r][0] + txt[R[r][3]:]

            return txt_new, R
        else:
            return txt, []

def prepare_txt(sl):
    sl = sl.replace('.', '. ')
    sl = sl.replace('  ', ' ')
    ignorecase = re.compile('[”«»]', re.IGNORECASE)
    sl = re.sub(ignorecase, '\"', sl)
    sl = sl.replace('©', 'с')
    sl = sl.replace('®', '0')
    sl = sl.replace('й', 'и').replace('ь', '').replace('ъ', '')
    sl = sl.replace('ообщество', 'общество').replace('аакционерное', 'акционерное').replace('ппубличное', 'публичное')
    sl = sl.replace('ооткрытое', 'открытое').replace('ззакрытое', 'закрытое')
#     sl = sl.replace(' - ', '-')
    sl = "".join(re.findall(r'\w+|\s+|[\"-\'-”«»]', sl)).strip()
    sl, sl_replace = replace_org_forms(sl)
    s = check_case(sl)
    return s.lower(), sl_replace