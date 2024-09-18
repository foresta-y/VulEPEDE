import networkx as nx

def slice_pointers(edges, pointer_node_list):
    def get_adjacent_nodes(node_id, edges, direction):
        """获取指定方向（'in' 或 'out'）上的相邻节点"""
        adjacent_nodes = []
        for edge in edges:
            if direction == 'in' and edge.node_out == node_id:
                adjacent_nodes.append(edge.node_in)
            elif direction == 'out' and edge.node_in == node_id:
                adjacent_nodes.append(edge.node_out)
        return adjacent_nodes

    def traverse(node_id, visited, edges, direction):
        """递归遍历相邻节点，生成切片"""
        if node_id in visited:
            return
        visited.add(node_id)
        adjacent_nodes = get_adjacent_nodes(node_id, edges, direction)
        for adj_node in adjacent_nodes:
            traverse(adj_node, visited, edges, direction)

    pointer_slices = []
    for pointer_node in pointer_node_list:
        # 遍历前驱节点，生成向后的切片
        backward_visited = set()
        traverse(pointer_node.id, backward_visited, edges, 'in')

        # 遍历后继节点，生成向前的切片
        forward_visited = set()
        traverse(pointer_node.id, forward_visited, edges, 'out')

        # 合并前驱和后继节点，形成完整切片
        full_slice = backward_visited.union(forward_visited)
        pointer_slices.append(full_slice)

    return pointer_slices


def slice_pointers_to_subgraph(edges, pointer_node_list, node_dict):
    def get_adjacent_nodes(node_id, edges, direction):
        """获取指定方向（'in' 或 'out'）上的相邻节点"""
        adjacent_nodes = []
        for edge in edges:
            if direction == 'in' and edge.node_out == node_id:
                adjacent_nodes.append(edge.node_in)
            elif direction == 'out' and edge.node_in == node_id:
                adjacent_nodes.append(edge.node_out)
        return adjacent_nodes

    def traverse(node_id, visited, edges, direction):
        """递归遍历相邻节点，生成切片"""
        if node_id in visited:
            return
        visited.add(node_id)
        adjacent_nodes = get_adjacent_nodes(node_id, edges, direction)
        for adj_node in adjacent_nodes:
            traverse(adj_node, visited, edges, direction)

    pointer_subgraphs = []
    for pointer_node in pointer_node_list:
        # 遍历前驱节点，生成向后的切片
        backward_visited = set()
        traverse(pointer_node.id, backward_visited, edges, 'in')

        # 遍历后继节点，生成向前的切片
        forward_visited = set()
        traverse(pointer_node.id, forward_visited, edges, 'out')

        # 合并前驱和后继节点，形成完整切片
        full_slice = backward_visited.union(forward_visited)

        # 创建子图
        subgraph = nx.DiGraph()
        for node_id in full_slice:
            subgraph.add_node(node_id, **vars(node_dict[node_id]))
        for edge in edges:
            if edge.node_in in full_slice and edge.node_out in full_slice:
                subgraph.add_edge(edge.node_in, edge.node_out, **edge.attrs)
        
        pointer_subgraphs.append(subgraph)

    return pointer_subgraphs


def slice_nodes_to_single_subgraph(edges, node_list, node_dict, direction='both'):
    """
    生成包含指定节点及其相关边的单一子图。
    
    参数:
    edges: list[Edge] - 图中的边列表。
    node_list: list[Node] - 需要切片的节点列表。
    node_dict: dict - 包含所有节点对象的字典。
    direction: str - 遍历方向，可以是 'in'（前驱），'out'（后继）或 'both'（前驱和后继）。
    
    返回:
    nx.DiGraph - 包含指定节点及其相关边的单一子图。
    """
    def get_adjacent_nodes(node_id, edges, direction):
        """获取指定方向（'in' 或 'out'）上的相邻节点"""
        adjacent_nodes = []
        for edge in edges:
            if direction == 'in' and edge.node_out == node_id:
                adjacent_nodes.append(edge.node_in)
            elif direction == 'out' and edge.node_in == node_id:
                adjacent_nodes.append(edge.node_out)
        return adjacent_nodes

    def traverse(node_id, visited, edges, direction):
        """递归遍历相邻节点，生成切片"""
        if node_id in visited:
            return
        visited.add(node_id)
        adjacent_nodes = get_adjacent_nodes(node_id, edges, direction)
        for adj_node in adjacent_nodes:
            traverse(adj_node, visited, edges, direction)

    visited_nodes = set()
    for node in node_list:
        if direction in ['in', 'both']:
            traverse(node.id, visited_nodes, edges, 'in')
        if direction in ['out', 'both']:
            traverse(node.id, visited_nodes, edges, 'out')

    # 创建子图
    subgraph = nx.DiGraph()
    for node_id in visited_nodes:
        node_data = node_dict[node_id]
        subgraph.add_node(node_id, type=node_data.node_type, label=node_data.label, code=node_data.code)
    for edge in edges:
        if edge.node_in in visited_nodes and edge.node_out in visited_nodes:
            # label为CDG或者DDG
            subgraph.add_edge(edge.node_in, edge.node_out, label=edge.edge_type)

    return subgraph