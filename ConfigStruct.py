import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


"""
Christopher MacKinnon 2021

The objective of this class is a layer of abstraction on top of the ConfigSpace package, allowing
parameters to be defined in a hierarchy rather than absolutely defining each parameter. This is 
aimed at allowing the hyperparameter for each layer in a model to be dynamically constructed once 
a layer is defined 
    
"""

class Parameter:
   
    def __init__(self, name : str , hyperparameter_type : str, lower_or_constant_value , upper_value = None ,normal = False, log : bool = False):
        self.name = name
        self.type = hyperparameter_type
        self.lower = lower_or_constant_value 
        self.upper = upper_value
        self.normal = normal 
        self.log = log 

        if self.type == "Constant":
            self.config = CSH.Constant(name = name, value = lower_or_constant_value)
        if normal == False:
            if self.type == "Integer":
                self.config = CSH.UniformIntegerHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)

            if self.type == "Float":
                self.config = CSH.UniformFloatHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)
         
        if normal == True:
            if self.type == "Integer":
                self.config = CSH.NormalIntegerHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)

            if self.type == "Float":
                self.config = CSH.NormalFloatHyperparameter(name = name, lower = lower_or_constant_value, upper = upper_value, log = log)
         


    def get_args(self):
        return self.type, self.lower, self.upper, self.normal, self.log
class Integer_Struct(Parameter):
    """
    Generates 1 of each child parameter for each integer between lower and upper and adds conditions
    
    name of children follows format parent_name + _ + number + _ + child_name
    """
    
    def __init__(self, config_space,  children : list,struct_name : str , name : str , hyperparameter_type : str, lower_or_constant_value , upper_value = None ,normal = False, log :  bool = False ):
        
        super().__init__(name, hyperparameter_type, lower_or_constant_value, upper_value, normal, log)
        self.children_template = children
        self.children = []
        self.children_dict = dict()
        self.conditions = []
        self.config_space = config_space
        self.struct_name = struct_name 

    def add_children_to_config_space(self):
        self.config_space.add_hyperparameters( [child.config for child in self.children] )
        self.config_space.add_hyperparameter(self.config)
    def generate_name(self,parent_name : str, child_name : str, num : int):
        return parent_name + "_" + str(num) +"_" + child_name  

    def _generate_child_set(self, i ):
        children = list()
        for child_template in self.children_template:
            new_child_name = self.generate_name(self.struct_name, child_template.name , i )
            print(new_child_name , child_template.get_args())
            children.append( Parameter( new_child_name, *child_template.get_args() ))
        return children

    def _generate_conditions(self):
        pass

    def generate_children(self):
        #Generate child parameter for each value of the parent class
        for i in range(self.lower, self.upper + 1): 
            children = self._generate_child_set(i)
            self.children += children
            self.children_dict[i] = children
        self.add_children_to_config_space()

class Cumulative_Integer_Struct(Integer_Struct):
    def __init__(self,config_space ,  children : list,struct_name, name : str , hyperparameter_type : str, lower_or_constant_value , upper_value = None ,normal = False, log :bool = False ):
        super().__init__(config_space ,children, struct_name ,name, hyperparameter_type, lower_or_constant_value, upper_value, normal, log)



     
    def batch_add_greater_than_cond(self, a_list, b , num):
        for a in a_list:
            if num != self.lower: 
                cond = CS.GreaterThanCondition(a.config,b,num -1 )
                self.config_space.add_condition(cond)  
    
    def generate_conditions(self):
        for i in range(self.lower, self.upper + 1):
            self.batch_add_greater_than_cond( self.children_dict[i] , self.config , i )
    def init(self):
        self.generate_children()
        self.generate_conditions()
if __name__ == "__main__":
    pass        
