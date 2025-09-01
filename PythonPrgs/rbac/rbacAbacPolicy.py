from py_abac.storage import MongoStorage
from py_abac.policy import Policy
from pymongo import MongoClient
class rbacAbacPolicyStore(MongoStorage):
    def add(policy):
        db = "admin"
        col = "books"
        client = MongoClient()
        storage = MongoStorage(client[db][col])
        storage.add(policy)