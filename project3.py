
#https://github.com/ChuntaoLu/Algorithms-Design-and-
#Analysis/blob/master/week4%20Graph%20search%20and%20SCC/scc.py#L107

#The original script was written in Python 2. It computes the strong
# connected components(SCC) of a given graph.

#I'm not sure what sys accomplishes
#time is used to measure the time for the function call
#heapq is used to order the components
#groupby seems to be for gathering similar elements
#I'm not sure what defaultdict is for
import sys
import time
import heapq
#import resource
from itertools import groupby
from collections import defaultdict
import pdb
from heapq import *

#set rescursion limit and stack size limit
sys.setrecursionlimit(10 ** 6)
#resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, 2 ** 30))

#Indirected graph
class Graph:
    #@param isDirected affects how the edges will be added to the graph
    def __init__(self, isDirected):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
        self.isDirected = isDirected

    def add_node(self, value):
        self.nodes.add(value)
    
    def add_nodes(self, values = None):
        if values is None:
            return
        for value in values:
            self.add_node(value)
    
    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance
        if self.isDirected:
            self.edges[to_node].append(from_node)
            self.distances[(to_node, from_node)] = distance

    #
    def tpl_order(self,start):
        self.n = 0
        return self.dfs_tpl_order(start,[])
                    
    def dfs_tpl_order(self,start,path):
        path = list(path)
        path = path + [start]
        for edge in self.edges[start]:
            if edge not in path:
                path = self.dfs_tpl_order(edge,path)
        print (self.n, start)
        self.n -= 1
        return path
        
        
    def bfs(self, start):
        visited, queue = set(), [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(set(self.edges[vertex]) - visited)
        return visited
    
    #No comment
    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        for key in set(self.edges[start]) - visited:
            self.dfs(key, visited)
        return visited
    
    #No comment
    def dfs_paths(self, start, goal):
        stack = [(start, [start])]
        while stack:
            (vertex, path) = stack.pop()
            for next in set(self.edges[vertex]) - set(path):
                if next == goal:
                    yield path + [next]
                else:
                    stack.append((next, path + [next]))

    def bfs_paths(self, start, goal):
        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next in set(self.edges[vertex]) - set(path):
                if next == goal:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))
    
#We start by growing a cloud of vertices beginnign with a node 'v' and eventualy covering all the #vertices. Each vertex will have a label representing the distance from the candidate node to 'v'
    def dijkstra(self, initial):
        visited = {initial: 0}
        path = {}

        nodes = set(self.nodes)

        while nodes:
            min_node = None
            for node in nodes:
              if node in visited:
                if min_node is None:
                  min_node = node
                elif visited[node] < visited[min_node]:
                  min_node = node

            if min_node is None:
              break

            nodes.remove(min_node)
            current_weight = visited[min_node]

            for edge in self.edges[min_node]:
              weight = current_weight + self.distances[(min_node, edge)]
              if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

        return visited, path


#Although similar to Djkstra's algorithm, Prim's algorithm start with a random vertex 'v' and grow #a cloud of vertices (minimum spanning tree) starting from the vertex 'v' by adding the least #weighted edge into the cloud.

    def prim( self):
        nodes = list(self.nodes)
        edges = self.edges
        
        conn = defaultdict( list )
        for from_node in nodes:
            for to_node in self.edges[from_node]:
                c = self.distances[(from_node, to_node)]
                conn[ from_node ].append( (c, from_node, to_node) )
                conn[ to_node ].append( (c, to_node, from_node) )
     
        mst = []
        used = set( [nodes[ 0 ]] )
        usable_edges = conn[ nodes[0] ][:]
        heapify( usable_edges )
     
        while usable_edges:
            cost, n1, n2 = heappop( usable_edges )
            if n2 not in used:
                used.add( n2 )
                mst.append( ( n1, n2, cost ) )
     
                for e in conn[ n2 ]:
                    if e[ 2 ] not in used:
                        heappush( usable_edges, e )
        return mst

    # Step 1: For each node prepare the destination and predecessor
    # Initialize takes in the graph and the source node and creates two dictionaries, one for destination and one for predecessor
    # It starts with the assumption that the nodes are infinitely far away, and then rewrites them based on their weights
    # Initialize traverses the connected nodes of the graph and organize them into their respective dictionary of either destination or predecessor
    def initialize(self, source):
        d = {} # destination
        p = {} # predecessor
        for node in self.nodes:
            d[node] = float('Inf') # assumption is they are very far
            p[node] = None
        d[source] = 0 # starting node
        return d, p

    # Records the lowest distance between a node and neighbour
    def relax(self, node, neighbour, d, p):
        if d[neighbour] > d[node] + self.distances[(node, neighbour)]:
            d[neighbour]  = d[node] + self.distances[(node, neighbour)]
            p[neighbour] = node
            
    # The recursive function to find the shortest distances
    def bellman_ford(self, source):
        d, p = self.initialize(source)
        for i in range(len(self.edges)-1): #Run this until is converges
            for u in self.nodes:
                for v in self.edges[u]: #For each neighbour of u
                    self.relax(u, v, d, p) #Lets relax it
        
        # Step 3: check for negative-weight cycles
        for u in self.nodes:
            for v in self.edges[u]:
                assert d[v] <= d[u] + self.distances[(u, v)]

        return d, p
############################################################################

#Start of task 6

############################################################################


#Tracker is a class that tracks time, source, leader, finish time and all nodes explored
class Tracker(object):

    def __init__(self):
        self.current_time = 0
        self.current_source = None
        self.leader = {}
        self.finish_time = {}
        self.explored = set()

#dfs is set up within SCC function as a recursive call that explores a graph as represented by dictionary
def dfs(graph_dict, node, tracker):
    tracker.explored.add(node)
    tracker.leader[node] = tracker.current_source
    for head in graph_dict[node]:
        if head not in tracker.explored:
            dfs(graph_dict, head, tracker)
    tracker.current_time += 1
    tracker.finish_time[node] = tracker.current_time

#dfs_loop is the outer loop of the SCC function within the recursive call that checks all Strongly Connected Components. Current source node changes after inner loop ends
def dfs_loop(graph_dict, nodes, tracker):
    for node in nodes:
        if node not in tracker.explored:
            tracker.current_source = node
            dfs(graph_dict, node, tracker)

#reverses the direction of the edges in the directed graph
def graph_reverse(graph):
    reversed_graph = defaultdict(list)
    for tail, head_list in graph.items():
        for head in head_list:
            reversed_graph[head].append(tail)
    return reversed_graph

#Runs through the graph twice, first backward, then forwards. Returns a dictionary with the Strongly Connected Components.
def scc(graph):
    out = defaultdict(list)
    tracker1 = Tracker()
    tracker2 = Tracker()
    nodes = set()
    reversed_graph = graph_reverse(graph)
    for tail, head_list in graph.items():
        nodes |= set(head_list)
        nodes.add(tail)
    nodes = sorted(list(nodes), reverse=True)
    dfs_loop(reversed_graph, nodes, tracker1)
    sorted_nodes = sorted(tracker1.finish_time,
                          key=tracker1.finish_time.get, reverse=True)
    dfs_loop(graph, sorted_nodes, tracker2)
    for lead, vertex in groupby(sorted(tracker2.leader, key=tracker2.leader.get),
                                key=tracker2.leader.get):
        out[lead] = list(vertex)
    return out


def question1():
    g = Graph(True)
    
    g.add_nodes({'A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'})
    g.add_edge('A','B',0)
    g.add_edge('A','E',0)
    g.add_edge('A','F',0)
    g.add_edge('B','A',0)
    g.add_edge('B','C',0)
    g.add_edge('B','F',0)
    g.add_edge('C','B',0)
    g.add_edge('C','D',0)
    g.add_edge('C','G',0)
    g.add_edge('D','C',0)
    g.add_edge('D','G',0)
    g.add_edge('E','A',0)
    g.add_edge('E','F',0)
    g.add_edge('E','I',0)
    g.add_edge('F','A',0)
    g.add_edge('F','B',0)
    g.add_edge('F','E',0)
    g.add_edge('F','I',0)
    g.add_edge('G','C',0)
    g.add_edge('G','D',0)
    g.add_edge('G','J',0)
    g.add_edge('H','K',0)
    g.add_edge('H','L',0)
    g.add_edge('I','E',0)
    g.add_edge('I','F',0)
    g.add_edge('I','J',0)
    g.add_edge('I','M',0)
    g.add_edge('J','G',0)
    g.add_edge('J','I',0)
    g.add_edge('K','H',0)
    g.add_edge('K','O',0)
    g.add_edge('K','L',0)
    g.add_edge('L','K',0)
    g.add_edge('L','H',0)
    g.add_edge('L','P',0)
    g.add_edge('M','I',0)
    g.add_edge('M','N',0)
    g.add_edge('N','M',0)
    g.add_edge('O','K',0)
    g.add_edge('P','L',0)

    run1 = g.bfs('A')
    print('Question 1 results: ')
    print('BFS: ', run1)
    run2 = g.dfs('A')
    print('DFS: ', run2)
    paths = list(g.dfs_paths('A', 'F'))
    print(paths)

def question3():
    start = time.time()
    graph = Graph(False)

    graph.edges = {
        1 : set([3]),
        2 : set([1]),
        3 : set([2, 5]),
        4 : set([1, 2, 12]),
        5 : set([6, 8]),
        6 : set([7, 8, 10]),
        7 : set([10]),
        8 : set([9, 10]),
        9 : set([5, 11]),
        10 : set([11, 9]),
        11 : set([12]),
        12 : set ()
        }
        
    groups = scc(graph.edges)
    t1 = time.time() - start
    top_5_1 = heapq.nlargest(5, groups, key=lambda x: len(groups[x]))
    result1 = []
    for i in range(5):
        try:
            result1.append(len(groups[top_5_1[i]]))
        except:
            result1.append(0)
        
    print('Question 3, Strongly connected components are: ')
    for key in groups:
        print(groups[key])
	
def question4():
    graph = Graph(True)
    #Vertices(nodes)
    graph.add_nodes({'A', 'B', 'C', 'D', 'E'})
    graph.add_edge('A', 'B', 22)
    graph.add_edge('A', 'C', 9)
    graph.add_edge('A', 'D', 12)
    graph.add_edge('B', 'H', 34)
    graph.add_edge('B', 'F', 36)
    graph.add_edge('B', 'C', 35)
    graph.add_edge('C', 'F', 42)
    graph.add_edge('C', 'E', 65)
    graph.add_edge('C', 'D', 4)
    graph.add_edge('D', 'E', 33)
    graph.add_edge('D', 'I', 30)
    graph.add_edge('E', 'F', 18)
    graph.add_edge('E', 'G', 23)
    graph.add_edge('F', 'H', 24)
    graph.add_edge('F', 'G', 39)
    graph.add_edge('G', 'H', 25)
    graph.add_edge('G', 'I', 21)
    graph.add_edge('H', 'I', 19)
    v, path = graph.dijkstra('A')
    
    print('Question 4, shortest path tree starting at \'A\'')
    print('Visited: ', v)
    print('Path :', path)
	
def question5():
    graph = Graph(True)
    
    graph.add_nodes({'A', 'B', 'C', 'D', 'E'})
    
    graph.add_edge('A', 'B', 22)
    graph.add_edge('A', 'C', 9)
    graph.add_edge('A', 'D', 12)
    graph.add_edge('B', 'H', 34)
    graph.add_edge('B', 'F', 36)
    graph.add_edge('B', 'C', 35)
    graph.add_edge('C', 'F', 42)
    graph.add_edge('C', 'E', 65)
    graph.add_edge('C', 'D', 4)
    graph.add_edge('D', 'E', 33)
    graph.add_edge('D', 'I', 30)
    graph.add_edge('E', 'F', 18)
    graph.add_edge('E', 'G', 23)
    graph.add_edge('F', 'H', 24)
    graph.add_edge('F', 'G', 39)
    graph.add_edge('G', 'H', 25)
    graph.add_edge('G', 'I', 21)
    graph.add_edge('H', 'I', 19)
    
    print('Question 5, minimum spanning tree: ')
    print(graph.prim())

def main():
    question1()
    #The answer t question 2 lies in our report
    question3()
    question4()
    question5()
    
if __name__ == "__main__":
    main()

