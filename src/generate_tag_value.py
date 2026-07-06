import random
import string
import numpy as np
import pandas as pd
import re


class GenerateTagValue:
    def __init__(self, companies_list):
        self.tag_dict = {'[ORG]': ['RBK_NAME', 'DD_FNAME', 'DD_SNAME'],
                         '[BANK]': ['CBR_NAME', 'CBR_FULL_NAME', 'DD_FNAME'],
                         '[INN]': ['RBK_INN'],
                         '[KPP]': ['DD_KPP'],
                         '[LOC]': ['RBK_ADDR', 'DD_ADDR'],
                         '[PER]': ['DD_MANAGER_NAME']}
        self.opf = {'RBK_NAME': 'DD_OPF_SHORT',
                    'DD_FNAME': 'DD_OPF_FULL',
                    'DD_SNAME': 'DD_OPF_SHORT'}
        self.path_excel_requsites = companies_list
        self.requsites_org = pd.read_excel(self.path_excel_requsites,
                                           sheet_name='COMPANIES',
                                           engine='openpyxl').fillna('')

        self.requsites_org.loc[self.requsites_org['RBK_INN'] != "", 'RBK_INN'] = self.requsites_org['RBK_INN'].apply(str).apply(lambda x: x.split('.')[0])
        self.requsites_org.loc[self.requsites_org['DD_KPP'] != "", 'DD_KPP'] = self.requsites_org['DD_KPP'].apply(str).apply(lambda x: x.split('.')[0])
        self.requsites_org.loc[self.requsites_org['DD_INN'] != "", 'DD_INN'] = self.requsites_org['DD_INN'].apply(str).apply(lambda x: x.split('.')[0])

        self.requsites_ip = self.requsites_org[self.requsites_org['DD_OPF_SHORT'] == 'ИП'].reset_index(drop=True)

        self.requsites_bank = pd.read_excel(self.path_excel_requsites, sheet_name='BANKS', engine='openpyxl').fillna('')
        self.count_row_org = len(self.requsites_org)
        self.count_row_ip = len(self.requsites_ip)
        self.count_row_bank = len(self.requsites_bank)

    @staticmethod
    def generate_rs():
        """
        Функция генерации расчётного счёта
        """
        rs_bank = '4' + "".join(random.choice(string.digits) for _ in range(19))
        return rs_bank

    @staticmethod
    def generate_ks():
        """
        Функция генерации корреспондентского счёта
        """
        ks_bank = '30' + "".join(random.choice(string.digits) for _ in range(18))
        return ks_bank

    @staticmethod
    def generate_bik():
        """
        Функция генерации БИК Банка
        """
        bik_bank = '04' + "".join(random.choice(string.digits) for _ in range(7))
        return bik_bank

    @staticmethod
    def generate_tel():
        """
        Функция генерации телефонного номера
        """
        first_char = str(random.choice([7, 8]))
        code_reg = str(random.choice([3, 8, 9])) + str(random.randint(0, 99))
        code_reg = code_reg if len(code_reg) == 3 else code_reg + '0' * (3 - len(code_reg))
        second_char = "".join(random.choice(string.digits) for _ in range(7))

        short_tel = np.random.choice([1, 0], p=[0.02, 0.98])  # Для короткого номера вероятность 2 % , для полного 98%

        if short_tel:
            first_char = str(random.randint(20, 99))
            second_char = "".join(random.choice(string.digits) for _ in range(4))
            tel_num = random.choice([f'{first_char} {second_char[:2:]} {second_char[2::]}',
                                     f'{first_char}-{second_char[:2:]}-{second_char[2::]}'])
        else:
            tel_num = random.choice([f'+{first_char} {code_reg} {second_char}',
                                     f'{first_char} {code_reg} {second_char[:3:]} {second_char[3::]}',
                                     f'+{first_char} {code_reg} {second_char[:2:]} {second_char[2:4:]} {second_char[4:7:]}',
                                     f'{first_char}-{code_reg}-{second_char}',
                                     f'+{first_char}-{code_reg}-{second_char[:3:]}-{second_char[3::]}',
                                     f'{first_char}-{code_reg}-{second_char[:2:]}-{second_char[2:4:]}-{second_char[4:7:]}',
                                     f'+{first_char} ({code_reg}) {second_char}',
                                     f'{first_char} ({code_reg}) {second_char[:3:]} {second_char[3::]}',
                                     f'+{first_char} ({code_reg}) {second_char[:2:]} {second_char[2:4:]} {second_char[4:7:]}',
                                     f'{first_char}-({code_reg})-{second_char}',
                                     f'+{first_char}-({code_reg})-{second_char[:3:]}-{second_char[3::]}',
                                     f'{first_char}-({code_reg})-{second_char[:2:]}-{second_char[2:4:]}-{second_char[4:7:]}',
                                     f'+{first_char} {code_reg} {second_char[:3:]} {second_char[3:5]} {second_char[5:7:]}',
                                     f'{first_char}-{code_reg}-{second_char[:3:]}-{second_char[3:5]}-{second_char[5:7:]}',
                                     f'+{first_char} ({code_reg}) {second_char[:3:]} {second_char[3:5]} {second_char[5:7:]}',
                                     f'+{first_char}{code_reg}{second_char}'])
        return tel_num

    @staticmethod
    def generate_okpo():
        """
        Функция генерации ОКПО
        """
        return "".join(random.choice(string.digits) for _ in range(random.choice([8, 10, 14])))

    @staticmethod
    def generate_okato():
        """
        Функция генерации ОКАТО
        """
        return "".join(random.choice(string.digits) for _ in range(5))

    def get_org(self, column, dfrow):
        rnd = random.randint(1, 3)
        if rnd == 1:
            opf_behind = [re.sub(fr'{dfrow[self.opf[column]].values[0]} ', '', dfrow[column].values[0],
                                 flags=re.I) + f', {dfrow[self.opf[column]].values[0]}',
                          re.sub(fr'{dfrow[self.opf[column]].values[0]} ', '', dfrow[column].values[0],
                                 flags=re.I) + f',{dfrow[self.opf[column]].values[0]}',
                          re.sub(fr'{dfrow[self.opf[column]].values[0]} ', '', dfrow[column].values[0],
                                 flags=re.I) + f' {dfrow[self.opf[column]].values[0]}']
            return opf_behind[random.randint(0, len(opf_behind) - 1)]
        else:
            return dfrow[column].values[0]

    @staticmethod
    def generate_oktmo():
        """
        Функция генерации ОКТМО
        """
        return "".join(random.choice(string.digits) for _ in range(11))

    def get_tag_value(self, tag):
        if tag in self.tag_dict:

            if tag not in ['[BANK]', '[PER]']:
                if tag == '[ORG]':
                    column = random.choice(self.tag_dict[tag])
                    if random.randint(1, 2) > 1:

                        tag_value = self.requsites_org[
                            self.requsites_org.index == random.randint(1, self.count_row_org - 1)]
                        tag_value = self.get_org(column, tag_value)
                    else:
                        tag_value = self.requsites_ip[
                            self.requsites_ip.index == random.randint(1, self.count_row_ip - 1)]
                        tag_value = self.get_org(column, tag_value)
                else:
                    column = random.choice(self.tag_dict[tag])
                    tag_value = self.requsites_org[column][random.randint(1, self.count_row_org - 1)]

                tag_value = "" if tag_value == '-' else tag_value
                return str(tag_value) if tag_value is not None else ''
            else:
                column = random.choice(self.tag_dict[tag])
                tag_value = self.requsites_bank[column][random.randint(1, self.count_row_bank - 1)]
                tag_value = "" if tag_value == '-' else tag_value
                return str(tag_value) if tag_value is not None else ''
        else:
            if tag == '[RS]':
                return self.generate_rs()
            elif tag == '[KS]':
                return self.generate_ks()
            elif tag == '[TEL]':
                return self.generate_tel()
            elif tag == '[OKPO]':
                return self.generate_okpo()
            elif tag == '[OKATO]':
                return self.generate_okato()
            elif tag == '[OKTMO]':
                return self.generate_oktmo()
            elif tag == '[BIK]':
                return self.generate_bik()
            else:
                return ''


if __name__ == "__main__":
    entities = {'[ORG]': 'ГАРАЖНО-СТРОИТЕЛЬНЫЙ КООПЕРАТИВ "АСС"',
                '[BANK]': 'ПАО АКБ "1Банк"',
                '[INN]': '7609017469',
                '[KPP]': '760901001',
                '[RS]': '40702810124000011658',
                '[KS]': '30101810500000000976',
                '[LOC]': '191144, г Санкт-Петербург, Центральный р-н, Дегтярный пер, д 11 литер а',
                '[PER]': 'Янова Оксана Вячеславовна',
                '[BIK]': '044525976',
                '[TEL]': '+7 (812) 748-2777',
                '[OKPO]': '89041828',
                '[OKATO]': '40278562000',
                '[OKTMO]': '40349000'
                }
    GTV = GenerateTagValue(r'../data/interim/ner_tag_companies_and_banks.xlsx')
    for k, v in entities.items():
        get_tag = GTV.get_tag_value(k)
        print(k, get_tag)
