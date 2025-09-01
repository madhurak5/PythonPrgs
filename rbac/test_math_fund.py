from math_func import StudentDB
import pytest
db = None
# def setup_module(module):
#     print("Setting up...")
#     global db
#     db = StudentDB()
#     db.connect('data.json')
#
# def teardown_module(module):
#     print("tearing down ")
#     db.close()
import sys
@pytest.fixture(scope='module')
def db():
    print("Setting up...")
    # global db
    db = StudentDB()
    db.connect('data.json')
    yield db
    print("tearing down ")
#   db.close()

def test_madhu_data(db):
    madhur_data = db.get_data('Madhura')
    assert madhur_data['id'] == 1
    assert madhur_data['name'] == "Madhura"
    assert madhur_data['result'] == "pass"

# @pytest.mark.skipif(sys.version_info < (3,3),reason="do not run")
# def test_add(db):
#     assert db.add(7,5) == 11
#     assert db.add(7, 6) == 13
#     print(db.add(7,5), "______________________________________________s" )
#
# def test_add_float(db):
#     assert db.add(7.4,5.4) == 12.8
#     assert db.add(7.2, 6.6) == 13.8
# @pytest.mark.number
def test_prod(db):
    assert db.prod(5,4) == 20

# @pytest.mark.strings
# def test_add_string(db):
#     res = db.add("hello", "world")
#     assert res == "hello world"

@pytest.mark.parametrize('x, y,result',[(7, 3, 10), (7,6, 13), ('hello','world','hello world'), (7.4, 5.4, 12.8)])
def test_add_string(db, x, y, result):
    assert db.add(x, y) == result
    # assert db.add(7,5) == 11
    # assert db.add(7, 6) == 13
    # assert db.add(7.4,5.4) == 12.8
    # assert db.add(7.2, 6.6) == 13.8
    # res = db.add("hello", "world")
    # assert res == "hello world"