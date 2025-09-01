# from charm.toolbox.schemebase import SchemeBase
# from charm.toolbox.ABEnc import ABEnc
# from charm.toolbox.pairinggroup import ZR
from Cryptodome.PublicKey import RSA
key = RSA.generate(2048)
priv_key = key.export_key()
print(priv_key)
pub_key = key.publickey().export_key()
print(pub_key)