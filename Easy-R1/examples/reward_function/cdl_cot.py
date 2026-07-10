# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List

formalgeo_train_whole_circle = ['188', '253', '349', '419', '476', '560', '760', '767', '1045', '1170', '1517', '1521', '1522', '1545', '1693', '1820', '1821', '1884', '1974', '2020', '2112', '2607', '2846', '4640', '4668', '4686', '4736', '4769', '4850', '5308', '5348', '5349', '5351']
formalgeo_test_whole_circle = ['876', '1833', '4915', ]
formalgeo_train_whole_circle = formalgeo_train_whole_circle + formalgeo_test_whole_circle

def can_rotate(value1, value2, isList=False, isstr=False):
    # The length is different and cannot be rotated
    # 判断 value2 是否可以通过旋转得到 value1
    if len(value1) != len(value2):
        return False

    if isList:
        for _ in range(len(value1)):
            if value1 == value2:
                return True
            value2 = [value2[-1]] + value2[:-1]
        return False
    elif isstr:
        for _ in range(len(value1)):
            if value1 == value2:
                return True
            value2 = value2[-1] + value2[:-1]
        return False
    else:
        value1 = value1[0]
        value2 = value2[0]
        extended_value1 = value1 + value1

        if value2 in extended_value1:
            return True
        else:
            return False

def split_str(s):
    result = []
    stack = []
    start = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
        elif char == ',' and not stack:
            result.append(s[start:i].strip())
            start = i + 1

    result.append(s[start:].strip())
    return result

def consCDL_acc_reward(response: str, ground_truth: str) -> float:
    try:
        match_gt = re.search(r"consCDL:\[(.*?)\]", ground_truth)
        gt_dict = {'cons_cdl':split_str(match_gt.group(1).replace(' ', ''))}
        target = gt_dict['cons_cdl']
        match_ref = re.search(r"consCDL:\[(.*?)\]", response)
        ref_dict = {'cons_cdl':split_str(match_ref.group(1).replace(' ', ''))}
        prediction = ref_dict['cons_cdl']
        prediction = list(set(prediction))
    except:
        # print(response)
        # print(ground_truth)
        return 0.0, 0.0

    if len(target) == 1 and target == ['']:
        return 0.0, 0.0

    target_types = []
    target_elems = []
    target_mark = [0] * len(target)
    prediction_mark = [0] * len(prediction)

    for i in range(len(target)):
        try:
            matches = re.search(r'^(.*?)\((.*?)\)', target[i].strip("'"))
            preset_type = matches.group(1)
            values = matches.group(2).split(',')
            target_types.append(preset_type)
            target_elems.append(values)
        except:
            target_types.append('')
            target_elems.append([''])
            continue
    
    for i in range(len(prediction)):
        matches = re.search(r'^(.*?)\((.*?)\)', prediction[i].strip("'"))
        try:
            preset_type = matches.group(1)
            values = matches.group(2).split(',')
        except:
            continue

        for j in range(len(target)):
            if preset_type == target_types[j] and target_mark[j] == 0:   
                if preset_type == 'Shape':
                    if can_rotate(values, target_elems[j], isList=True):
                        prediction_mark[i] = 1
                        target_mark[j] = 1
                        break
                    else:
                        # new_values = []
                        # # if proID=='49' and preset_type == 'Shape' and values == ['WA','AZ','ZW']:
                        # #     import pdb;pdb.set_trace()
                        if len(values[0]) == 2:
                            new_values = []
                            new_values.append(values[0][1] + values[0][0])
                            for value_idx in range(1, len(values)):
                                take_value = values[len(values)-value_idx]
                                if len(take_value) == 2:
                                    new_values.append(take_value[1] + take_value[0])
                                elif len(take_value) == 3:
                                    new_values.append(take_value[0] + take_value[1] + take_value[2])
                                else:
                                    break
                                    # import pdb;pdb.set_trace()
                            if can_rotate(new_values, target_elems[j], isList=True):
                                prediction_mark[i] = 1
                                target_mark[j] = 1
                                break
                        elif len(values[0]) == 3:
                            new_values = []
                            new_values.append(values[0][0] + values[0][1] + values[0][2])
                            for value_idx in range(1, len(values)):
                                take_value = values[len(values)-value_idx]
                                if len(take_value) == 2:
                                    new_values.append(take_value[1] + take_value[0])
                                elif len(take_value) == 3:
                                    new_values.append(take_value[0] + take_value[1] + take_value[2])
                                else:
                                    break
                            if can_rotate(new_values, target_elems[j], isList=True):
                                prediction_mark[i] = 1
                                target_mark[j] = 1
                                break
                            # import pdb;pdb.set_trace()
                        elif len(values[0]) > 3:
                            break                            
                elif preset_type == 'Collinear':
                    if len(values) != 1:
                        break
                    elif ((values[0] == target_elems[j][0]) or (values[0] == target_elems[j][0][::-1])):
                        prediction_mark[i] = 1
                        target_mark[j] = 1
                        break
                elif preset_type == 'Cocircular':
                    try:
                        if len(target_elems[j]) == 1 and values[0] == target_elems[j][0]:
                            prediction_mark[i] = 1
                            target_mark[j] = 1
                            break
                        elif values[0] == target_elems[j][0] and can_rotate(values[1],target_elems[j][1],isstr=True):
                            prediction_mark[i] = 1
                            target_mark[j] = 1
                            break
                    except:
                        continue

    
    return sum(target_mark) / len(target_mark), sum(prediction_mark) / len(prediction_mark)

def textCDL_acc_reward(response: str, ground_truth: str) -> float:
    try:
        match_gt = re.search(r"textCDL:\[(.*?)\]", ground_truth)
        gt_dict = {'text_cdl':split_str(match_gt.group(1).replace(' ', ''))}
        target = gt_dict['text_cdl']
        match_ref = re.search(r"textCDL:\[(.*?)\]", response)
        ref_dict = {'text_cdl':split_str(match_ref.group(1).replace(' ', ''))}
    except:
        # print(response)
        # print(ground_truth)
        return 0.0, 0.0
    if len(target) == 1 and target == ['']:
        return 0.0, 0.0

    target_types = []
    target_elems = []
    target_mark = [0] * len(target)

    for i in range(len(target)):
        try:
            gt_dict['text_cdl'][i] = gt_dict['text_cdl'][i].strip("'")
            matches = re.search(r'^([^(]+)\((.*)\)$', gt_dict['text_cdl'][i])
            preset_type = matches.group(1)
            values = matches.group(2)
        except:
            target_types.append('')
            target_elems.append([''])
            continue

        pattern = r'Add|Sub|Mul|Div|Sin|Cos|Tan|RatioOfSimilarTriangle|RatioOfMirrorSimilarTriangle|RatioOfSimilarQuadrilateral|RatioOfMirrorSimilarQuadrilateral'
        if preset_type in ('PerpendicularBetweenLine','ParallelBetweenLine'):
            values = values.split(',')
        elif preset_type == 'Equal' and not re.search(pattern, values):
            preset_type = 'Equal-a'
            values = values.split(',')
        target_types.append(preset_type)
        target_elems.append(values)

    prediction = ref_dict['text_cdl']
    prediction = list(set(prediction))
    prediction_mark = [0] * len(prediction)
    if prediction == ['']:
        return 0.0, 0.0
    
    for i in range(len(prediction)):
        prediction[i] = prediction[i].strip("'")

        try:
            matches = re.search(r'^([^(]+)\((.*)\)$', prediction[i])
            preset_type = matches.group(1)
            values = matches.group(2)
        except:
            pass

        if preset_type == 'Equal' and not re.search(r'Add|Sub|Mul|Div|Sin|Cos|Tan', values):
            preset_type = 'Equal-a'
        
        for j in range(len(target)):
            if preset_type == target_types[j] and target_mark[j] == 0:
                if preset_type == 'Equal-a':
                    if isinstance(values, list):
                        values = values
                    elif isinstance(values, set):
                        values = list(values)
                    else:
                        values = values.split(',')
                    # values = values if isinstance(values, list) else values.split(',')
                    p_lineNum = sum('LengthOfLine' in e for e in values)
                    t_lineNum = sum('LengthOfLine' in e for e in target_elems[j])
                    p_angNum = sum('MeasureOfAngle' in e for e in values)
                    t_angNum = sum('MeasureOfAngle' in e for e in target_elems[j])
                    # The target and prediction types are different
                    if (p_lineNum != t_lineNum) or (p_angNum != t_angNum):
                        continue
                    # Judgment of the numerical condition of the line segment length
                    elif p_lineNum == 1 and t_lineNum == 1:
                        try:
                            p_line = re.split(r'[()]', values[0])[1]
                            t_line = re.split(r'[()]', target_elems[j][0])[1]

                            if (p_line == t_line or p_line[::-1] == t_line) and values[1] == target_elems[j][1]:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break   
                        except:
                            continue
                    # Judgment of the numerical condition of the angle length
                    elif p_angNum == 1 and t_angNum == 1:
                        # if idx == 62:
                        #     import pdb;pdb.set_trace()
                        try:
                            p_ang = list(re.split(r'[()]', values[0])[1])
                            t_ang = list(re.split(r'[()]', target_elems[j][0])[1])
                            if values[1] == target_elems[j][1] and p_ang == t_ang:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break
                        except:
                            break
                    # Judgment of equality condition of two line segments
                    elif p_lineNum == 2 and t_lineNum == 2:
                        try:
                            p_l1, p_l2 = re.split(r'[()]', values[0])[1], re.split(r'[()]', values[1])[1]
                            t_l1, t_l2 = re.split(r'[()]', target_elems[j][0])[1], re.split(r'[()]', target_elems[j][1])[1]
                        except:
                            break
                        set1 = {p_l1, p_l1[::-1], p_l2, p_l2[::-1]}
                        set2 = {t_l1, t_l1[::-1], t_l2, t_l2[::-1]}
                        if set1 == set2:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                    # Judgment of equality condition of two angles
                    elif p_angNum == 2 and t_angNum == 2:
                        try:
                            p_a1, p_a2 = list(re.split(r'[()]', values[0])[1]), list(re.split(r'[()]', values[1])[1])
                            t_a1, t_a2 = list(re.split(r'[()]', target_elems[j][0])[1]), list(re.split(r'[()]', target_elems[j][1])[1])
                        except:
                            break
                        if p_a1[1] == t_a1[1] and p_a1==t_a1 and p_a2==t_a2:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                        elif p_a1[1] == t_a2[1] and p_a1==t_a2 and p_a2==t_a1:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                    # Judgment of  conditions other than lines and angles
                    else:
                        values = set(values)
                        target_con = set(target_elems[j])
                        if values == target_con:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                        else:
                            values = list(values)
                elif preset_type == 'Equal' and values == target_elems[j]:
                    target_mark[j] = 1
                    prediction_mark[i] = 1
                    break
                elif preset_type == 'PerpendicularBetweenLine':
                    values = values if isinstance(values, list) else values.split(',')
                    try:
                        p1 = list(values[0])
                        p2 = list(values[1])
                        t1 = list(target_elems[j][0])
                        t2 = list(target_elems[j][1])
                        if p1[1] == p2[1] and p1 == t1 and p2 == t2:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                        elif p1[1] == p2[1] and p1 == t2 and p2 == t1:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                    except:
                        break
                elif preset_type == 'ParallelBetweenLine':
                    try:
                        values = values if isinstance(values, list) else values.split(',')
                        p1 = values[0]
                        p2 = values[1]
                        t1 = target_elems[j][0]
                        t2 = target_elems[j][1]
                    except:
                        break
                    if (p1 == t1 and p2 == t2) or (p1 == t2 and p2 == t1):
                        target_mark[j] = 1
                        prediction_mark[i] = 1
                        break
                else:
                    if values == target_elems[j]:
                        target_mark[j] = 1
                        prediction_mark[i] = 1
                        break
    return sum(target_mark) / len(target_mark), sum(prediction_mark) / len(prediction_mark)                   
            
def imgCDL_acc_reward(response: str, ground_truth: str) -> float:
    try:
        match_gt = re.search(r"imgCDL:\[(.*?)\]", ground_truth)
        gt_dict = {'img_cdl':split_str(match_gt.group(1).replace(' ', ''))}
        target = gt_dict['img_cdl']
        match_ref = re.search(r"imgCDL:\[(.*?)\]", response)
        ref_dict = {'img_cdl':split_str(match_ref.group(1).replace(' ', ''))}
        prediction = ref_dict['img_cdl']
        prediction = list(set(prediction))
        prediction_mark = [0] * len(prediction)
    except:
        return 0.0, 0.0
    
    if len(target) == 1 and target == ['']:
        return 0.0, 0.0
    target_types = []
    target_elems = []
    target_mark = [0] * len(target)
    for i in range(len(target)):
        try:
            matches = re.search(r'^([^(]+)\((.*)\)$', target[i].strip("'"))
            preset_type = matches.group(1)
            values = matches.group(2)

            pattern = r'Add|Sub|Mul|Div|Sin|Cos|Tan|RatioOfSimilarTriangle|RatioOfMirrorSimilarTriangle|RatioOfSimilarQuadrilateral|RatioOfMirrorSimilarQuadrilateral'
            if preset_type == 'Equal' and re.search(pattern, values):
                preset_type = 'Equal-a'
            else:
                values = values.split(',')
            target_types.append(preset_type)
            target_elems.append(values)
        except:
            target_types.append('')
            target_elems.append([''])
            continue
    for i in range(len(prediction)):
        try:
            matches = re.search(r'^([^(]+)\((.*)\)$', prediction[i].strip("'"))
            preset_type = matches.group(1)
            values = matches.group(2)
        except:
            continue
        if preset_type == 'Equal' and re.search(r'Add|Sub|Mul|Div|Sin|Cos|Tan', values):
            preset_type = 'Equal-a'

        # Different types of judgment methods are considered in the case of collinearity
        for j in range(len(target)):
            if preset_type == target_types[j] and target_mark[j] == 0:
                if preset_type == 'Equal':
                    if isinstance(values, list):
                        values = values
                    elif isinstance(values, set):
                        values = list(values)
                    else:
                        values = values.split(',')
                    # values = values if isinstance(values, list) else values.split(',')
                    p_lineNum = sum('LengthOfLine' in e for e in values)
                    t_lineNum = sum('LengthOfLine' in e for e in target_elems[j])
                    p_angNum = sum('MeasureOfAngle' in e for e in values)
                    t_angNum = sum('MeasureOfAngle' in e for e in target_elems[j])
                    # The target and prediction types are different
                    if (p_lineNum != t_lineNum) or (p_angNum != t_angNum):
                        continue
                    # Judgment of the numerical condition of the line segment length
                    elif p_lineNum == 1 and t_lineNum == 1:
                        try:
                            p_line = re.split(r'[()]', values[0])[1]
                            t_line = re.split(r'[()]', target_elems[j][0])[1]
                            if (p_line == t_line or p_line[::-1] == t_line) and values[1] == target_elems[j][1]:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break
                        except:
                            break
                    # Judgment of the numerical condition of the angle length
                    elif p_angNum == 1 and t_angNum == 1:
                        try:
                            p_ang = list(re.split(r'[()]', values[0])[1])
                            t_ang = list(re.split(r'[()]', target_elems[j][0])[1])
                        
                            if values[1] == target_elems[j][1] and p_ang[1] == t_ang[1] and p_ang == t_ang:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break
                        except:
                            break
                    # Judgment of equality condition of two line segments
                    elif p_lineNum == 2 and t_lineNum == 2:
                        try:
                            p_l1, p_l2 = re.split(r'[()]', values[0])[1], re.split(r'[()]', values[1])[1]
                            t_l1, t_l2 = re.split(r'[()]', target_elems[j][0])[1], re.split(r'[()]', target_elems[j][1])[1]
                            set1 = {p_l1, p_l1[::-1], p_l2, p_l2[::-1]}
                            set2 = {t_l1, t_l1[::-1], t_l2, t_l2[::-1]}
                            if set1 == set2:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break
                        except:
                            break
                    # Judgment of equality condition of two angles
                    elif p_angNum == 2 and t_angNum == 2:
                        try:
                            p_a1, p_a2 = list(re.split(r'[()]', values[0])[1]), list(re.split(r'[()]', values[1])[1])
                            t_a1, t_a2 = list(re.split(r'[()]', target_elems[j][0])[1]), list(re.split(r'[()]', target_elems[j][1])[1])
                            if p_a1[1] == t_a1[1] and p_a1 == t_a1 and p_a2[1] == t_a2[1] and p_a2 == t_a2:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break
                            elif p_a1[1] == t_a2[1] and p_a1 == t_a2 and p_a2[1] == t_a1[1] and p_a2 == t_a1:
                                target_mark[j] = 1
                                prediction_mark[i] = 1
                                break
                        except:
                            continue
                    # Judgment of arc condition
                    else:
                        values = set(values)
                        target_arc = set(target_elems[j])
                        if values == target_arc:
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                        else:
                            values = list(values)
                elif preset_type == 'Equal-a' and values == target_elems[j]:
                    target_mark[j] = 1
                    prediction_mark[i] = 1
                    break
                elif preset_type == 'PerpendicularBetweenLine':
                    values = values if isinstance(values, list) else values.split(',')
                    try:
                        p1 = list(values[0])
                        p2 = list(values[1])
                        t1 = list(target_elems[j][0][::-1])
                        t2 = list(target_elems[j][1][::-1])
                        if p1[1] == p2[1] and set(p1) == set(t1) and set(p2) == set(t2):
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                    except:
                        break
                elif preset_type == 'ParallelBetweenLine':
                    try:
                        values = values if isinstance(values, list) else values.split(',')
                        p1 = values[0]
                        p2 = values[1]
                        t1 = target_elems[j][0]
                        t2 = target_elems[j][1]
                        if (p1==t1 and p2==t2) or (p1==t2 and p2==t1):
                            target_mark[j] = 1
                            prediction_mark[i] = 1
                            break
                    except:
                        break
                    

    return sum(target_mark) / len(target_mark), sum(prediction_mark) / len(prediction_mark)

# def consCDL_format_reward(response: str, proID: str) -> float:
#     whole_circle = formalgeo_train_whole_circle
#     # Step 1: parse consCDL from the given content
#     match = re.search(r"consCDL:\[(.*?)\]", response)
#     try:
#         target_dict = {'cons_cdl':split_str(match.group(1).replace(' ', ''))}
#     except:
#         return 0.0
    
#     target = target_dict['cons_cdl']
#     target = list(dict.fromkeys(target))
#     target_mark = [0] * len(target)
#     circle_center = set()
#     isolate_points = []

#     if len(target) == 0:
#         return 0.0

#     # Step 2: parse preset and values for all consCDL
#     for i in range(len(target)): 
#         try:
#             matches = re.search(r'^(.*?)\((.*?)\)', target[i].strip("'"))
#             preset_type = matches.group(1)
#             values = matches.group(2).split(',')
#         except:
#             continue

#         # Step 3: check syntax for each preset
#         # values should not be empty for any preset
#         if len(values) == 0:
#             continue

#         # Rules for Shape: 
#         # Shape is used to describe angles composed of two lines or closed figures composed of many lines and arcs, e.g., Shape(EC,OCD,DP,PE). 
#         # We need to declare all the smallest edges. 
#         # When declaring lines (EC,DP,PE), they should be arranged in a counterclockwise direction, with the pointsconnecting end to end.
#         # When declaring arcs (OCD, arc with points CD on circle O), it's important to note that only represent them in a counterclockwise direction.
#         if preset_type == 'Shape':
#             if len(values) == 1:
#                 # 点不会孤立出现，非法：
#                 if len(values[0])== 1:
#                     isolate_points.append(values[0][0])
#                     continue
#                 # 声明一条线段 Line segment
#                 elif len(values[0]) == 2:
#                     target_mark[i] = 1
#                     continue
#                 # 当弧单独出现时，不需要使用Shape来声明，因为弧必伴随Cocircular谓词
#                 else:
#                     continue
#             elif len(values) == 2:
#                 # 声明一个角 Angle或是有弧参与组成的封闭图形
#                 # 检查是否是两条线段，是两条线段，则是角
#                 if len(values[0])== 2 and len(values[1])== 2:
#                     # 检查角所在的点
#                     angle = set(values[0]) & set(values[1])
#                     angle = list(angle)
#                     if len(angle) != 1: #多点/无点重合，不是角
#                         continue
#                     # 检查是否首尾相接
#                     if values[0][-1]==angle[0] and values[1][0]==angle[0]:
#                         target_mark[i] = 1
#                         continue
#                     else:
#                         continue
#                 # 两个弧组成的封闭图形
#                 elif len(values[0])== 3 and len(values[1])== 3:
#                     c_1 = values[0][0]
#                     c_2 = values[1][0]
#                     if c_1 == c_2:
#                         continue
#                     else: #剩下两个点完全相同，顺序相反
#                         if (values[0][1]==values[1][2]) and (values[0][2]==values[1][1]):
#                             target_mark[i] = 1
#                             continue
#                         elif (values[0][1]==values[1][1]) and (values[0][2]==values[1][2]):
#                             target_mark[i] = 1
#                             continue
#                         else:
#                             continue
#                 # 一条弧一条线段
#                 elif len(values[0])==3 and len(values[1])==2:
#                     c = values[0][0]
#                     v_1 = set(values[0][1:])
#                     v_2 = set(values[1])
#                     if (v_1 == v_2):
#                         if (values[1][0]==values[0][-1]):
#                             target_mark[i] = 1
#                             continue
#                         else:
#                             continue
#                     else:
#                         continue
#                 elif len(values[0])==2 and len(values[1])==3:
#                     c = values[1][0]
#                     v_1 = set(values[1][1:])
#                     v_2 = set(values[0])
#                     if (v_1 == v_2):
#                         target_mark[i] = 1
#                     else:
#                         continue
#                 else:
#                     continue
#             elif len(values) > 2:
#                 # 检查线段/弧是否有序
#                 last_start = None
#                 last_end = None
#                 last_arc = False
#                 wrong_flag = False
#                 for j, value in enumerate(values):
#                     if len(value) == 2: #线段
#                         cur_start = value[0]
#                         cur_end = value[1]
#                         cur_arc = False
#                     elif len(value) == 3: #弧
#                         cur_start = value[1]
#                         cur_end = value[2]
#                         cur_arc = True
#                     else:
#                         # print(f'wrong Shape format {target[i]} for proID {proID}: wrong values in Shape.')
#                         wrong_flag = True
#                         break
#                     if last_end is not None:
#                         if not last_arc and not cur_arc and (last_end != cur_start):
#                             # print(f'wrong Shape format {target[i]} for proID {proID}: wrong rotated values in Shape.')
#                             wrong_flag = True
#                             break
#                         elif last_arc and not cur_arc and cur_start!=last_start and cur_start!=last_end:
#                             wrong_flag = True
#                             break
#                         elif cur_arc and not last_arc and cur_start!=last_end and cur_end!=last_end:
#                             wrong_flag = True
#                             break
#                         elif last_arc and cur_arc:
#                             edge_1 = set(value) & set(last_start)
#                             edge_2 = set(value) & set(last_end)
#                             if len(edge_1)==0 and len(edge_2)==0:
#                                 wrong_flag = True
#                                 break
#                     last_end = cur_end
#                     last_start = cur_start
#                     last_arc = cur_arc
#                 if not wrong_flag:
#                     target_mark[i] = 1
#                     continue
#         elif preset_type == 'Collinear': 
#             if len(values) != 1:
#                 continue
#             else:
#                 # 只有三个或三个以上的点共线需要声明
#                 if len(values[0]) <= 2:
#                     continue
#                 else:
#                     target_mark[i] = 1
#                     continue
#         elif preset_type == 'Cocircular':
#             if len(values) == 1:    #表示一个圆，只有圆心，圆上没有其他点
#                 c = values[0]
#                 # if c in isolate_points and proID not in whole_circle:
#                 #     continue
#                 if c in circle_center:
#                     continue
#                 if len(c) != 1:
#                     continue
#                 else:
#                     target_mark[i] = 1
#                     circle_center.add(c)
#                     continue
#             elif len(values) == 2:  #表示一个圆，圆上有其他点
#                 c = values[0]
#                 # if c in isolate_points and proID not in whole_circle:
#                 #     continue
#                 if c in circle_center:
#                     continue
#                 if len(c) != 1:
#                     continue
#                 else:
#                     target_mark[i] = 1
#                     circle_center.add(c)
#                     continue
#             else:
#                 continue
#         else:
#             continue
                    
#     return sum(target_mark) / len(target_mark)  

def format_reward(response: str) -> float:
    pattern = r"<think>([\s\S]*?)<\/think>\s*<CDL>([\s\S]*?)<\/CDL>"
    match = re.search(pattern, response)
    if match:
        think_content = match.group(1)
        cdl_content = match.group(2)
        return 1.0, cdl_content
    else:
        return 0.0, response

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.5) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        # print(f'key of reward_input:{reward_input.keys()}')
        # response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        # format_score = format_reward(reward_input["response"])
        format_score, cdl = format_reward(reward_input["response"])
        textCDL_recall_score, textCDL_precision_score = textCDL_acc_reward(reward_input["response"], reward_input["ground_truth"])
        consCDL_recall_score, consCDL_precision_score = consCDL_acc_reward(reward_input["response"], reward_input["ground_truth"])
        imgCDL_recall_score, imgCDL_precision_score = imgCDL_acc_reward(reward_input["response"], reward_input["ground_truth"])

        # # Average Score
        # consCDL_score = (consCDL_recall_score + consCDL_precision_score) / 2
        # imgCDL_score = (imgCDL_precision_score + imgCDL_recall_score) / 2
        # textCDL_score = (textCDL_precision_score + textCDL_recall_score) / 2

        # F1 Score
        try:
            consCDL_score = (2 * consCDL_recall_score * consCDL_precision_score) / (consCDL_recall_score + consCDL_precision_score)
        except:
            consCDL_score = 0
        try:
            imgCDL_score =  (2 * imgCDL_recall_score * imgCDL_precision_score)/ (imgCDL_precision_score + imgCDL_recall_score)
        except:
            imgCDL_score = 0
        try:
            textCDL_score =  (2 * textCDL_recall_score * textCDL_precision_score)/ (textCDL_precision_score + textCDL_recall_score)
        except:
            textCDL_score = 0


        # consCDL_format_score = consCDL_format_reward(reward_input["response"], reward_input["proID"])
        # overall_score = 0.1 * format_score + (1 - format_weight) * (consCDL_recall_score + consCDL_precision_score) / 2 + format_weight * (consCDL_format_score + imgCDL_precision_score + imgCDL_recall_score + textCDL_precision_score + textCDL_recall_score) / 5
        overall_score = 0.1 * format_score + (1 - format_weight) *  consCDL_score+ format_weight * (imgCDL_score + textCDL_score) / 2
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "textCDL_precision": textCDL_precision_score,
                "consCDL_precision": consCDL_precision_score,
                "imgCDL_precision": imgCDL_precision_score,
                "textCDL_recall": textCDL_recall_score,
                "consCDL_recall": consCDL_recall_score,
                "imgCDL_recall": imgCDL_recall_score,
                # "consCDL_format": consCDL_format_score,
                
            }
        )
    return scores
