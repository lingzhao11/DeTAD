import numpy as np
class TreeNode():
    def __init__(self, start, end, indicator="F"):
        self.val = [start, end]
        self.info = None    
        self.child = []
        self.parent = []
        self.indicator = indicator


class MultiChildTree():
    def __init__(self, start, end, indicator="F"):
        self.root = TreeNode(start, end, indicator)
        self.node_list = []
    
        
    def insert(self, start, end, indicator="F"):
        self.treenode = TreeNode(start, end, indicator)
        self.add()

    def add(self):
        if len(self.root.child) ==0:
            self.root.child.append(self.treenode)
            self.treenode.parent.append(self.root)
        else:
            self.temp_parent_list = []
            self.pre(self.root)
            for each_parent in self.temp_parent_list:
                self.treenode.parent.append(each_parent)
                each_parent.child.append(self.treenode)
        
    def pre(self, tree_root):  
        sin = 0
        for i in range(0, len(tree_root.child), 1):
            
            temp_child = tree_root.child[i]
            if temp_child in self.temp_parent_list:
                continue
            if (temp_child.val[0] < self.treenode.val[0] and self.treenode.val[1] <= temp_child.val[1]) or (temp_child.val[0] <= self.treenode.val[0] and self.treenode.val[1] < temp_child.val[1]):
                sin =1
                if len(temp_child.child)==0:
                    self.temp_parent_list.append(temp_child)
                 
                else:
                    self.pre(temp_child)
            
            else:
                if temp_child.val[0] < self.treenode.val[0] and abs(self.treenode.val[1] - temp_child.val[1]) <= 2:
                    self.treenode.val[1] = temp_child.val[1]
                    sin =1
                    if len(temp_child.child)==0:
                        self.temp_parent_list.append(temp_child)
                   
                    else:
                        self.pre(temp_child)
                elif self.treenode.val[1] < temp_child.val[1] and abs(self.treenode.val[0]-temp_child.val[0]) <= 2:
                    self.treenode.val[0] = temp_child.val[0]
                    sin = 1
                    if len(temp_child.child) == 0:
                        self.temp_parent_list.append(temp_child)
                        
                    else:
                        self.pre(temp_child)
            
   
        if sin == 0:
            self.temp_parent_list.append(tree_root)
            return

    def pre_traversal(self, tree_root):  # traverse in pre-order
       
        if (tree_root != None):
            if tree_root != self.root:
                self.node_list.append(tree_root)
            
            for i in range(0, len(tree_root.child), 1):
                if tree_root.child[i].val[0] == tree_root.val[0] and tree_root.child[i].val[1] == tree_root.val[1]:
                    continue
               
                self.pre_traversal(tree_root.child[i])
                
    def acquire_list(self):
        self.pre_traversal(self.root)
        return self.node_list





