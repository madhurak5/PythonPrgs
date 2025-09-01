import json
class StudentDB:
    def __init__(self):
        self.__data = None

    def connect(self, data_file):
        with open(data_file) as json_file:
            self.__data = json.load(json_file)

    def get_data(self, name):
        for stud in self.__data['students']:
            if stud['name'] == name:
                return stud
    def close(self):
        pass

    def add(self, x, y):
        return x+y

    def prod(self, x, y):
        return x*y


# {[{{'$.qualification': {'condition': 'AnyOf', 'values': [{'condition': 'Equals', 'value': 'B.E'},
# 			                                            {'condition': 'Equals', 'value': 'M.Tech'},
# 			                                            {'condition': 'Equals', 'value': 'Ph.D'}]
#                         }
#     },
#     {'$.designation': {'condition': 'Equals', 'value': {'condition': 'Equals', 'value': 'Assoc. Professor'}}}
#    },
# {'$.department': {'condition': 'AnyOf', 'values': [{'condition': 'Equals', 'value': 'Computer Science'},
# 			{'condition': 'Equals', 'value': 'Electronics'},
# 			 {'condition': 'Equals', 'value': 'Mechanical'},
# 			{'condition': 'Equals', 'value': 'Biomedical'}]}}],
# {'$.role': {'condition': 'Equals', 'value': {'condition': 'Equals', 'value': 'Superuser'}}}}
