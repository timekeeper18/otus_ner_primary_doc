try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML
import json
import re

import regex
import requests
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def delete_org_forms(txt):
    """
    удаляем организационную форму предприятия для сверки только наименования
    :param txt:
    :return:
    """
    # регулярка с учетом количества опечаток
    r = regex.compile(
        r"(\bзакрытое\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    r = regex.compile(
        r"(\bакционерное\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(\bоткрытое\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(\bобщество\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(\bкоммерческий\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(\bбанк\b){e<=1}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(с ограниченной){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(\bответственностью\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(
        r"(\bпубличное\b){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    # print(txt)
    r = regex.compile(
        r"(торговый дом){e<=2}",
        regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    #     print(txt)
    r = regex.compile(r'(\bООО\b){e<=2}', regex.IGNORECASE)
    txt = regex.sub(r, '', txt)
    txt = txt.replace('  ', ' ')
    return txt.strip()


def sum_inn(inn):
    if len(inn) not in (10, 12):
        return False
    if len(inn) == 10:
        return inn[-1] == inn_csum(inn[:-1])
    else:
        return inn[-2:] == inn_csum(inn[:-2]) + inn_csum(inn[:-1])


def inn_csum(inn):
    k = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    pairs = zip(k[11 - len(inn):], [int(x) for x in inn])
    return str(sum([k * v for k, v in pairs]) % 11 % 10)


def check_inn(inn):
    r = regex.compile(r'[0-9]+')
    inn = regex.findall(r, inn)
    inn = ''.join(inn)

    if len(inn) <= 8:
        return 'False', inn

    if len(inn) not in (10, 12):
        if len(inn) == 9:
            return 'True', inn_csum(inn)
        else:
            return 'False', inn
    if len(inn) == 10:
        if inn[-1] == inn_csum(inn[:-1]):
            return 'True', inn
        else:
            inn = inn_csum(inn[:-1])
            return 'True', inn
    else:
        if inn[-2:] == inn_csum(inn[:-2]) + inn_csum(inn[:-1]):
            return 'True', inn
        else:
            return 'False', inn


def get_dadata_suggestion(query, resource, apy_key, count=15):
    BASE_URL = "https://suggestions.dadata.ru/suggestions/api/4_1/rs/suggest/{}"
    # API_KEY = '806ef7579701a1cc83665906adef64d72c05d8dc'

    url = BASE_URL.format(resource)
    headers = {"Authorization": "Token {}".format(apy_key), "Content-Type": "application/json"}
    data = {"query": query, "count": count}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    return r.json()


def get_verification_data(inp, apy_key):
    """
    Getting full data by org. INN, KPP or NAME
    """
    orgs_data = []
    result = get_dadata_suggestion(inp, "party", apy_key, count=15)

    # print('result', result)
    for item in result['suggestions']:
        orgs_data_element = {'Name1': None, 'INN': None, 'KPP': None, 'Address': None, 'Full_Name': None,
                             'Short_Name': None,
                             'City': None, 'postal_code': None}
        for key in item:
            if key == 'value':
                orgs_data_element['Name1'] = item[key]
            if key == 'data':
                orgs_data_element['INN'] = item[key]['inn']
                if 'kpp' not in list(item['data'].keys()):
                    orgs_data_element['KPP'] = 'IP'
                else:
                    orgs_data_element['KPP'] = item[key]['kpp']
            if key == 'data':
                orgs_data_element['Address'] = item[key]['address']['unrestricted_value']
            if key == 'data':
                orgs_data_element['Full_Name'] = item[key]['name']['full_with_opf']
                orgs_data_element['Short_Name'] = item[key]['name']['short_with_opf']
                orgs_data_element['City'] = item[key]['address']['data']['city']
                orgs_data_element['postal_code'] = item[key]['address']['data']['postal_code']
                orgs_data_element['full_with_opf'] = item[key]['name']['full_with_opf']
        orgs_data.append(orgs_data_element)

    return orgs_data


def del_extra_chars(ent):
    # r = regex.compile(r'(\bООО\b){e<=2}', regex.IGNORECASE)
    # ent = regex.split(r, ent)[-1]
    ents = list(re.split(r"[(|{]", ent))
    ent = sorted(ents, key=lambda k: len(k), reverse=True)[0]
    # print(ent)
    ent = delete_org_forms(ent)

    clear_string = re.compile(r'[A-Za-zА-ЯЁа-яё0-9]+')
    return ' '.join(re.findall(clear_string, ent)).strip()


def get_correct_comp_details(ent_dict, ent, apy_key='2e443721385453b3bd84679ce85d4a07a735d9c5', ratio=89):
    """
    Получаем скоректированные внешним справочником данные по реквизитам компании
    :param ent_dict:
    :param ent:
    :param apy_key:
    :param ratio:
    :return:
    """
    c_kpp = ''
    new_inn = ''
    data_serv = ent_dict
    inn = data_serv['{}_INN'.format(ent)].strip()
    name_st = data_serv['{}_NAME'.format(ent)].strip()
    if not sum_inn(inn) and len(' '.join(inn.split())) >= 11:
        if not name_st.upper().startswith('ИП') and not name_st.upper().startswith('ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ'):
            new_inn = inn[:10:]
            c_kpp = inn[10::]
    if new_inn != '':
        inn = new_inn
    kpp = data_serv['{}_KPP'.format(ent)].strip()
    if c_kpp != '':
        kpp = str(c_kpp) + kpp
    name_org = data_serv['{}_NAME'.format(ent)]
    name_org = del_extra_chars(name_org)
    new_dict = {'{}_INN'.format(ent): inn,
                '{}_NAME'.format(ent): name_org,
                '{}_KPP'.format(ent): kpp,
                '{}_ADDRESS'.format(ent): data_serv['{}_ADDRESS'.format(ent)]}

    # если ИНН/КПП распознались слитно
    if len(new_dict['{}_INN'.format(ent)]) >= 15:
        inn, kpp = get_separated_innkpp(new_dict['{}_INN'.format(ent)])
        new_dict.update({'{}_INN'.format(ent): inn})
        new_dict.update({'{}_KPP'.format(ent): kpp})
    if len(new_dict['{}_KPP'.format(ent)]) >= 15:
        inn, kpp = get_separated_innkpp(new_dict['{}_INN'.format(ent)])
        new_dict.update({'{}_INN'.format(ent): inn})
        new_dict.update({'{}_KPP'.format(ent): kpp})
    ####################################################

    new_par = get_verification_data(inn, apy_key)
    data_serv.update(new_dict)
    # Сверка с результатами справочника: запрос по ИНН
    if not new_par:
        # Сверка с результатами справочника: запрос по Наименованию организации
        new_par = get_verification_data(name_org, apy_key)

    return check_similarity_vit(new_par, new_dict, ent, name_org, data_serv, name_st, ratio)


def get_separated_innkpp(innkpp):
    inn = innkpp[:10:]
    kpp = innkpp[10::]
    return inn, kpp


def check_similarity(new_par, new_dict, ent, name_org, data_serv, name_st, ratio):
    inn = data_serv['{}_INN'.format(ent)]
    # try:
    #     if check_inn(inn)[0] == 'True':
    #         inn = check_inn(inn)[1]
    # except:
    #     pass
    kpp = data_serv['{}_KPP'.format(ent)]
    address = data_serv['{}_ADDRESS'.format(ent)]
    # собираем кандидатов от для сравнения в один массив
    addr_split = address.split(' ')
    name_org = ' '.join(name_org.lower().split())
    inn = ' '.join(inn.split())
    kpp = ' '.join(kpp.split())
    name = name_org
    if len(new_par) == 1:
        serv_res = new_par[0]
        new_inn = serv_res.get('INN')
        new_kpp = serv_res.get('KPP')
        new_name = del_extra_chars(serv_res.get('Full_Name').lower())
        new_full_name = del_extra_chars(serv_res.get('Name1').lower())
        new_address = serv_res.get('Address')
        new_opf_name = serv_res.get('full_with_opf')
        ratio_name1 = fuzz.ratio(name_org, new_name)
        ratio_full_name = fuzz.ratio(name_org, new_full_name)
        if fuzz.ratio(inn, new_inn) == 100 and fuzz.ratio(kpp, new_kpp) == 100:
            return {'{}_INN'.format(ent): new_inn,
                    '{}_KPP'.format(ent): new_kpp,
                    '{}_NAME'.format(ent): new_opf_name,
                    '{}_ADDRESS'.format(ent): new_address}

        if fuzz.ratio(inn, new_inn) >= ratio and (ratio_name1 >= ratio or ratio_full_name >= ratio):
            new_req = {'{}_INN'.format(ent): new_inn,
                       '{}_NAME'.format(ent): new_opf_name,
                       '{}_ADDRESS'.format(ent): new_address}
            if fuzz.ratio(kpp, new_kpp) >= ratio:
                new_req.update({'{}_KPP'.format(ent): new_kpp})
            else:
                new_req.update({'{}_KPP'.format(ent): kpp})
            return new_req

    for z in new_par:
        serv_name = del_extra_chars(z.get('Name1').lower())
        serv_inn = z.get('INN')
        serv_kpp = z.get('KPP')
        full_name = del_extra_chars(z.get('Full_Name').lower())
        new_address = z.get('Address')
        ratio_name1 = fuzz.ratio(name_org, serv_name)
        ratio_inn = fuzz.ratio(serv_inn, inn)
        ratio_kpp = fuzz.ratio(serv_kpp, kpp)
        ratio_full_name = fuzz.ratio(name_org, full_name)

        if z.get('City') is not None:
            city = z.get('City').lower()
        else:
            city = ' '

        if ratio_inn == 100 and ratio_kpp == 100:
            name = z.get('full_with_opf') + '(!!!)' if ratio_name1 <= ratio else z.get('full_with_opf')
            return {'{}_INN'.format(ent): serv_inn,
                    '{}_NAME'.format(ent): name,
                    '{}_KPP'.format(ent): serv_kpp,
                    '{}_ADDRESS'.format(ent): new_address}

        # if ratio_name1 >= ratio and ratio_inn >= ratio or ratio_full_name >= ratio:
        if (ratio_name1 >= ratio or ratio_full_name >= ratio) and ratio_inn >= ratio:

            inn = z.get('INN')
            if ratio_kpp <= ratio:
                kpp = kpp + '(!!!)'
            else:
                kpp = serv_kpp

            if ratio_name1 >= ratio:
                name = z.get('full_with_opf')
            elif ratio_full_name >= ratio:
                name = z.get('full_with_opf')
            else:
                name += '(!!!)'
            if ratio_inn == 100 and ratio_kpp == 100:
                name = z.get('full_with_opf') + '(!!!)' if ratio_name1 <= ratio else z.get('full_with_opf')
            if process.extract(city, addr_split)[0][1] >= ratio:
                address = z.get('Address')
                new_dict.update({'{}_INN'.format(ent): inn,
                                 '{}_NAME'.format(ent): name,
                                 '{}_KPP'.format(ent): kpp,
                                 '{}_ADDRESS'.format(ent): address})
                return new_dict
            else:
                new_dict.update({'{}_INN'.format(ent): inn,
                                 '{}_NAME'.format(ent): name,
                                 '{}_KPP'.format(ent): kpp,
                                 '{}_ADDRESS'.format(ent): address})
                return new_dict

        elif (ratio_name1 <= ratio or ratio_full_name <= ratio) and ratio_inn >= ratio and ratio_kpp >= ratio:

            inn = z.get('INN')
            if ratio_kpp <= ratio:
                kpp = kpp + '(!!!)'
            else:
                kpp = serv_kpp
            if ratio_name1 >= ratio:
                name = z.get('full_with_opf')
            elif ratio_full_name >= ratio:
                name = z.get('full_with_opf')
            else:
                name += z.get('full_with_opf') + '(!!!)'

            # жесткая хамена результатами сервиса если ИНН и КПП совпали на 100%
            if ratio_inn == 100 and ratio_kpp == 100:
                name = z.get('full_with_opf') + '(!!!)' if ratio_name1 <= ratio else z.get('full_with_opf')

            if process.extract(city, addr_split)[0][1] >= ratio:

                address = z.get('Address')
                new_dict.update({'{}_INN'.format(ent): inn,
                                 '{}_NAME'.format(ent): name,
                                 '{}_KPP'.format(ent): kpp,
                                 '{}_ADDRESS'.format(ent): address})
                return new_dict
            else:
                # inn = z.get('INN')
                # if fuzz.ratio(z.get('KPP'), ' '.join(kpp.split())) <= 90:
                #     kpp = kpp + '(!!!)'
                # else:
                #     kpp = z.get('KPP')
                # name = z.get('Name1')
                new_dict.update({'{}_INN'.format(ent): inn,
                                 '{}_NAME'.format(ent): name,
                                 '{}_KPP'.format(ent): kpp,
                                 '{}_ADDRESS'.format(ent): address})
        else:
            continue
    if not sum_inn(' '.join(inn.split())):
        inn = inn + '(!!!)'
    new_dict = {'{}_INN'.format(ent): inn,
                '{}_NAME'.format(ent): name_st + '(!!!)',
                '{}_KPP'.format(ent): data_serv['{}_KPP'.format(ent)].strip() + '(!!!)',
                '{}_ADDRESS'.format(ent): data_serv['{}_ADDRESS'.format(ent)] + '(!!!)'}
    return new_dict


def check_similarity_vit(new_par, new_dict, ent, name_org, data_serv, name_st, ratio):
    # inn = data_serv['{}_INN'.format(ent)]
    # kpp = data_serv['{}_KPP'.format(ent)]
    # address = data_serv['{}_ADDRESS'.format(ent)]
    #
    # # собираем кандидатов от для сравнения в один массив
    # addr_split = address.split(' ')
    # name_org = ' '.join(name_org.lower().split())
    # inn = ' '.join(inn.split())
    # kpp = ' '.join(kpp.split())
    # # name = name_org

    if len(new_par) == 0:
        return {'{}_INN'.format(ent): data_serv['{}_INN'.format(ent)] + '(!!!)',
                '{}_NAME'.format(ent): name_org + '(!!!)',
                '{}_KPP'.format(ent): data_serv['{}_KPP'.format(ent)] + '(!!!)',
                '{}_ADDRESS'.format(ent): data_serv['{}_ADDRESS'.format(ent)] + '(!!!)'}
    max_ratio_name = 0
    final_name = ''
    for z in new_par:
        inn = data_serv['{}_INN'.format(ent)]
        kpp = data_serv['{}_KPP'.format(ent)]
        address = data_serv['{}_ADDRESS'.format(ent)]

        # собираем кандидатов от для сравнения в один массив
        addr_split = address.split(' ')
        name_org = ' '.join(name_org.lower().split())
        inn = ' '.join(inn.split())
        kpp = ' '.join(kpp.split())
        # name = name_org

        serv_name = del_extra_chars(z.get('Name1', '').lower())
        serv_inn = z.get('INN')
        serv_kpp = z.get('KPP')
        full_name = del_extra_chars(z.get('Full_Name', '').lower())
        short_name = del_extra_chars(z.get('Short_Name', '').lower() if z.get('Short_Name', '') is not None else '')
        new_address = z.get('Address')
        ratio_name1 = fuzz.ratio(name_org, serv_name)
        ratio_short_name = fuzz.ratio(name_org, short_name)
        ratio_inn = fuzz.ratio(serv_inn, inn)
        ratio_kpp = fuzz.ratio(serv_kpp, kpp)
        ratio_full_name = fuzz.ratio(name_org, full_name)

        # максимальное значение меры близости имен от сервиса по сравнению наименований без орг. форм
        # для сравнения с порогговым значением
        ratio_name, _ = sorted([[ratio_name1, z.get('Name1', '')],
                                [ratio_short_name, z.get('Short_Name', '')],
                                [ratio_full_name, z.get('Full_Name', '')]],
                               key=lambda k: k[0], reverse=True)[0]

        # находим наиболее близкой наименование от справочника с учетом орг. форм для уточнения наименования организации
        _, closest_name = sorted([[fuzz.ratio(name_st, z.get('Name1', '')), z.get('Name1', '')],
                                  [fuzz.ratio(name_st, z.get('full_with_opf', '')), z.get('full_with_opf', '')]],
                                 key=lambda k: k[0], reverse=True)[0]

        city = z.get('City').lower() if z.get('City') is not None else ' '
        if len(addr_split) > 0:
            final_addr = z.get('Address') if process.extract(city, addr_split)[0][1] >= ratio else address
        if ratio_inn == 100 and ratio_kpp == 100:
            name = closest_name + '(!!!)' if ratio_name < ratio else closest_name
            return {'{}_INN'.format(ent): serv_inn,
                    '{}_NAME'.format(ent): name,
                    '{}_KPP'.format(ent): serv_kpp,
                    '{}_ADDRESS'.format(ent): final_addr}

        if ratio_inn >= ratio:
            kpp = kpp + '(!!!)' if ratio_kpp < ratio else serv_kpp
            if ratio_name >= ratio:
                inn = z.get('INN')
                name = closest_name
                new_dict.update({'{}_INN'.format(ent): inn,
                                 '{}_NAME'.format(ent): name,
                                 '{}_KPP'.format(ent): kpp,
                                 '{}_ADDRESS'.format(ent): final_addr})
                return new_dict

            elif ratio_name < ratio <= ratio_kpp:
                inn = z.get('INN')
                name = closest_name + '(!!!)'

                new_dict.update({'{}_INN'.format(ent): inn,
                                 '{}_NAME'.format(ent): name,
                                 '{}_KPP'.format(ent): kpp,
                                 '{}_ADDRESS'.format(ent): final_addr})

                return new_dict

        if max_ratio_name < ratio_name > ratio:
            max_ratio_name = ratio_name
            final_name = closest_name

    if not sum_inn(' '.join(inn.split())):
        inn = inn + '(!!!)'
    new_dict = {'{}_INN'.format(ent): inn,
                '{}_NAME'.format(ent): final_name + '(!!!)' if final_name != '' else name_st,
                '{}_KPP'.format(ent): data_serv['{}_KPP'.format(ent)].strip() + '(!!!)',
                '{}_ADDRESS'.format(ent): data_serv['{}_ADDRESS'.format(ent)] + '(!!!)'}
    return new_dict


def web_verify_data(all_ent):
    buyer_new = get_correct_comp_details(all_ent, 'BUYER')
    all_ent.update(buyer_new)

    seller_new = get_correct_comp_details(all_ent, 'SELLER')
    all_ent.update(seller_new)
    return all_ent


if __name__ == '__main__':
    all_ent = {'SELLER_NAME': 'ПАО "ВЫМПЕЛ"КОМ")',
               'SELLER_ADDRESS': '127083, РФ МОСКВА УЛ. ВОСЬМОГО МАРТА Д 10 СТР 14', 'SELLER_INN': '7713076301',
               'SELLER_KPP': '997750001', 'BUYER_NAME': 'ГАРАНТ СВ',
               'BUYER_ADDRESS': '298685, РФ РЕСП. КРЫМ ГОР. ЯЛТА УЛ. ГЕНЕРАЛА ОСТРЯКОВА Д 9', 'BUYER_INN': '9103007830',
               'BUYER_KPP': '910301001', 'NUMBER': 'LX#P#7006202', 'DATE': '(!!!)',
               'SELLER_COORD': {'NAME': [2727, 616, 3607, 87], 'ADDRESS': [2729, 755, 2280, 81],
                                'INNKPP': [2724, 880, 1004, 107]},
               'BUYER_COORD': {'NAME': [2722, 1454, 841, 77], 'ADDRESS': [2720, 1591, 2763, 84],
                               'INNKPP': [2723, 1737, 1004, 76]}, 'HEAD_SHAPE': (2000, 9972)}

    print(web_verify_data(all_ent))