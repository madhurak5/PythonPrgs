import pytest

from py_abac.pdp import PDP, EvaluationAlgorithm
from py_abac.policy import Policy
from py_abac.provider.base import AttributeProvider
from py_abac.request import AccessRequest
from py_abac.storage.mongo import MongoStorage
# from ..test_storage.test_mongo import create_client
# from mongoClient import create_client
from pymongo import MongoClient
import os
from py_abac.policy.rules import Rules

def create_Cl():
    DEFAULT_MONGODB_HOST = "127.0.0.1:27017"
    host = os.getenv("MONGODB_HOST",DEFAULT_MONGODB_HOST)
    return MongoClient(host)
    connect('admin')

DB_NAME = 'db_test'
COLLECTION = 'policies_test'
SUBJECT_IDS = {"Meera": "user:1", "Pooja": "user:2", "Madhura": "user:3", "Henry": "user:4"}
POLICIES = [
    {
        "uid": "1",
        "description": "A Female Professor from any department can create, update or delete any file ",
        "effect": "allow",
        "rules": {
            "subject": [ {"$.id": {"condition": "Equals", "value": "Madhura"}},
                        {"$.gender": {"condition": "Equals", "value": "Female"}},
                        {"$.designation": {"condition": "Equals", "value": "Professor"}},
                        {"$.department": {"condition": "AnyOf",
                                    "values": [{"condition": "Equals", "value": "Computer Science"},
                                               {"condition": "Equals", "value": "Electronics"}]}}],
            "resource": {"$.name": {"condition": "RegexMatch", "value": ".*"}},
            "action": [{"$.method": {"condition": "AnyOf",
                                     "values": [{"condition": "Equals", "value": "create"},
                                                {"condition": "Equals", "value": "update"},
                                                {"condition": "Equals", "value": "delete"}]}}],
            "context": {}
        },
        "targets": {},
        "priority": 0
    },
    {
        "uid": "2",
        "description": "A Female Professor from any department can create, update or delete any file ",
        "effect": "deny",
        "rules": {
            "subject": [{"$.id": {"condition": "Equals", "value": "Poorvi"}},
                        ],
            "resource": {"$.name": {"condition": "RegexMatch", "value": ".*"}},
            "action": [{"$.method": {"condition": "AnyOf",
                                     "values": [{"condition": "Equals", "value": "create"},
                                                {"condition": "Equals", "value": "update"},
                                                {"condition": "Equals", "value": "delete"}]}}],
            "context": {}
        },
        "targets": {},
        "priority": 0
    }

]

@pytest.fixture
def st():
    client = create_Cl()
    storage = MongoStorage(client, DB_NAME, collection=COLLECTION)
    for policy_json in POLICIES:
        storage.add(Policy.from_json(policy_json))
    yield storage
    client[DB_NAME][COLLECTION].drop()
    client.close()

#
# @pytest.mark.parametrize('desc, request_json, should_be_allowed', [
#    (
#             'Female Prof from any department can update, delete, or create any file',
#             {
#                 "subject": {"id": "", "attributes": {"gender": "Female",  "designation":"Professor", "department":'Electronics' }},
#                 "resource": {"id": "", "attributes": {"name": ".*"}},
#                 "action": {"id": "", "attributes": {"method": "update"}},
#                 "context": {}
#             },
#             True,
#     )
# ])
# def test_is_allowed_deny_overrides(st, desc, request_json, should_be_allowed):
#     pdp = PDP(st, EvaluationAlgorithm.DENY_OVERRIDES)
#     request = AccessRequest.from_json(request_json)
#     assert should_be_allowed == pdp.is_allowed(request)

# @pytest.mark.parametrize('desc, request_json, should_be_allowed', [
#     (
#             'Max is allowed to update anything, even empty one',
#             {
#                 "subject": {"id": SUBJECT_IDS["Madhura"], "attributes": {"name": "Madhura"}},
#                 "resource": {"id": "", "attributes": {"name": ""}},
#                 "action": {"id": "", "attributes": {"method": "update"}},
#                 "context": {}
#             },
#             True,
#     )])
# def test_is_allowed_allow_overrides(st, desc, request_json, should_be_allowed):
#     pdp = PDP(st, EvaluationAlgorithm.ALLOW_OVERRIDES)
#     request = AccessRequest.from_json(request_json)
#     assert should_be_allowed == pdp.is_allowed(request)

@pytest.mark.parametrize('desc, request_json, should_be_allowed', [
    (
            'Madhura is allowed to update anything, even empty one',
            {
                "subject": {"id": "Madhura", "attributes": {"gender": "Female",  "designation":"Professor", "department":'Electronics'}},
                "resource": {"id": "", "attributes": {"name": ".*"}},
                "action": {"id": "", "attributes": {"method": "update"}},
                "context": {}
            },
            True,
    ),
    (
            'Poorvi is allowed to update anything, even empty one',
            {
                "subject": {"id": "Poorvi",
                            "attributes": {"gender": "Female", "designation": "Professor", "department": 'Electronics'}},
                "resource": {"id": "", "attributes": {"name": ".*"}},
                "action": {"id": "", "attributes": {"method": "update"}},
                "context": {}
            },
            True,
    )
])
def test_is_allowed_highest_priority(st, desc, request_json, should_be_allowed):
    pdp = PDP(st)
    request = AccessRequest.from_json(request_json)
    print("Allowed ----------------------------------------------------------------------------------")
    assert should_be_allowed != pdp.is_allowed(request)

request_json = {
                "subject": {
                    "id": "Poorvi",
                    "attributes":{"gender":"Female", "designation":"Professor", "department":'Electronics'}
                },
                "resource": {"id": "", "attributes": {"name": ".*"}},
                "action": {"id": "", "attributes": {"method": "update"}},
                "context": {}
            }
request = AccessRequest.from_json(request_json)

def test_pdp_create_error(st):
    with pytest.raises(TypeError):
        PDP(None)
    with pytest.raises(TypeError):
        PDP(st, None)
    with pytest.raises(TypeError):
        PDP(st, EvaluationAlgorithm.DENY_OVERRIDES, [None])


def test_is_allowed_error(st):
    g = PDP(st)
    with pytest.raises(TypeError):
        g.is_allowed(None)