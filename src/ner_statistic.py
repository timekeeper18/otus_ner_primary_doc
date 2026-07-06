from sklearn import metrics
import pandas as pd
import numpy as np
import ast
import re
from nervaluate import Evaluator
import operator
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer
detokenizer = NLTKMosesTokenizer('ru').detokenizer

def nomalize_spans(correct_result, dp=False):
    new_corret_result = []
    for row in correct_result:
        row_cor = ast.literal_eval(row)
        if dp:

            row_cor = [[''.join(re.findall('[A-Za-zА-Яа-я0-9]+', prepare_txt(i[1])))]
                       for i in row_cor if 'ORG' in i or 'PER' in i]
        else:
            row_cor = [[''.join(re.findall('[A-Za-zА-Яа-я0-9]+', i[1].lower().strip().replace('й', 'и')))]
                       for i in row_cor if 'ORG' in i or 'PER' in i]
        new_corret_result.append(row_cor)
    return new_corret_result


def get_tag_info_dict(org_string, tag_correct, tag_ner):
    true = []
    pred = []

    org_string = ''.join(re.findall('[A-Za-zА-Яа-я0-9]+', org_string.lower().strip()))

    for i in tag_correct:
        if isinstance(i, list):
            i = ' '.join(i)
        try:
            start_label = org_string.index(i.lower().strip())
            stop_label = start_label + len(i)
            true.append({'label': 'ORG', 'start': start_label, 'end': stop_label})
        except:
            continue

    for j in tag_ner:
        if isinstance(j, list):
            j = ' '.join(j)
        j = re.sub('\s{1,}', '', j)
        try:
            start_label = org_string.index(j.lower().strip())
            stop_label = start_label + len(j)
            pred.append({'label': 'ORG', 'start': start_label, 'end': stop_label})
        except:
            continue

    return true, pred


def get_stat(df_ner, correct_result, ner_result, name='ner'):
    org_text = df_ner['TEXT']
    true_tag_all = []
    pred_tag_all = []

    a1 = []
    for a in correct_result:
        a1.append(np.array(a).flatten())
    total_tegs = []
    for i in a1:
        total_tegs.extend(i)

    for index in range(0, len(df_ner)):
        true_pos, pred_pos = get_tag_info_dict(org_text[index], correct_result[index], ner_result[index])
        true_tag_all.append(true_pos)
        pred_tag_all.append(pred_pos)
    evaluator = Evaluator(true_tag_all, pred_tag_all, tags=['ORG', 'PER'])
    results, results_per_tag = evaluator.evaluate()
    res = results['exact']
    res['name'] = name
    # TP = correct
    # FN = possible-correct
    # FP = actual-correct
    #
    # Кол-во тегов - TP-FN-FP
    tn = len(total_tegs) - res['correct']-(res['possible'] - res['correct'])-(res['actual'] - res['correct'])
    res['acuracy'] = (tn+res['correct'])/len(total_tegs)
    return results['exact']


##############################################
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
    return s.lower()