import os
import re
import pickle


with open("/home/itachi/A/VulDet/CDDFVUL/pdg/sensitive_func.pkl", "rb") as fin:
    list_sensitive_funcname = pickle.load(fin)


def get_CVPC_node_list(node_dict):
    pointer_node_list = []
    array_node_list = []
    param_node_list = []
    call_node_list = []
    identifier_node_type = ['IDENTIFIER', 'FIELD_IDENTIFIER', 'assignment']
    array_node_type = ['indirectIndexAccess']
    call_type = ["CALL"]
    param_type = ['PARAM']
    for node in node_dict:
        node_type = node_dict[node].node_type
        node_code = node_dict[node].code
        if node_type == param_type:
            param_node_list.append(node_dict[node])
            
        if node_type in identifier_node_type:
            # 不存在指针则返回-1
            indx_1 = node_code.find("*")
            if indx_1 != -1:
                pointer_node_list.append(node_dict[node])
        
        if node_type in array_node_type:
            if node_code.find("[") != -1:
                array_node_list.append(node_dict[node])
        
        if node_type in call_type:
            for c in node_code.strip().split("("):
                # print("code:",c)
                if c.strip() in list_sensitive_funcname:
                    
                    call_node_list.append(node_dict[node])
                    
        # if node_type in call_type:
        #     for func_name in list_sensitive_funcname:     
        #         for c in node_code.strip().split("("):
        #             if func_name in c:
        #                 call_node_list.append(node_dict[node])
    param_node_list = list(set(param_node_list))
    pointer_node_list = list(set(pointer_node_list))
    array_node_list = list(set(array_node_list))
    call_node_list = list(set(call_node_list))
    return list(set(param_node_list + pointer_node_list + array_node_list + call_node_list))


def get_param_node(node_dict):
    param_node_list = []
    param_type = 'Param'
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type == param_type:
            param_node_list.append(node_dict[node])
    return param_node_list


def get_pointers_node(node_dict):
    pointer_node_list = []
    # identifier_list = []
    identifier_node_type = ['Identifier', 'Field_Identifier']
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type in identifier_node_type:
            node_code = node.code
            # 不存在指针则返回-1
            indx_1 = node_code.find("*")
            if indx_1 != -1:
                pointer_node_list.append(node_dict[node])
        
    pointer_node_list = list(set(pointer_node_list))
    return pointer_node_list

'''
<operator>.indirectIndexAccess
'''

def get_all_array(node_dict):
    array_node_list = []
    identifier_list = []
    identifier_node_type = ['indirectIndexAccess']
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type in identifier_node_type:
            
            identifier_list.append(node_dict[node])
    for node in identifier_list:
        node_code = node.code
        if node_code.find("[") != -1:
            array_node_list.append(node)
    array_node_list = list(set(array_node_list))
    return array_node_list
    
    
def get_all_sensitiveAPI(node_dict):
    with open("/home/itachi/A/VulDet/CDDFVUL/pdg/sensitive_func.pkl", "rb") as fin:
        list_sensitive_funcname = pickle.load(fin)
    call_node_list = []
    call_type = "Call"   
    for func_name in list_sensitive_funcname:
        for node in node_dict:
            node_type = node_dict[node].node_type
            node_code = node_dict[node].code.split("(")
            if node_type == call_type:
                for c in node_code:
                    if func_name in c:
                        call_node_list.append(node_dict[node])
                        
    return call_node_list


def get_all_integeroverflow_point(node_dict):
    interoverflow_list = []
    exp_type = 'assignment'
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type == exp_type:
            node_code = node_dict[node].code
            if node_code.find("="):
                code = node_code.split('=')[-1].strip()
                pattern = re.compile("((?:_|[A-Za-z])\w*(?:\s(?:\+|\-|\*|\/)\s(?:_|[A-Za-z])\w*)+)")
            else:
                code = node_code
                pattern = re.compile("(?:\s\/\s(?:_|[A-Za-z])\w*\s)")
            results = re.search(pattern, code)
            if results != None:
                interoverflow_list.append(node_dict[node])
            
    return interoverflow_list


def get_all_control_structure(node_dict):
    control_structure_list = []
    # control_structure_type = ['CONTROL_STRUCTURE']
    control_structure_type = ['equals', 'greaterEqualsThan', 'greaterThan', 'lessEqualsThan',
             'lessThan', 'logicalAnd', 'logicalNot', 'logicalOr', 'not', 'notEquals', 'or' ]
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type in control_structure_type:
            # 'if, else, do, while, for, switch, goto'
            # if node_dict[node].code.find("if") != -1 or node_dict[node].code.find("else") != -1 or node_dict[node].code.find("do") != -1 or node_dict[node].code.find("while") != -1 or node_dict[node].code.find("for") != -1 or node_dict[node].code.find("switch") != -1 or node_dict[node].code.find("goto") != -1:
            control_structure_list.append(node_dict[node])
            
    return control_structure_list

