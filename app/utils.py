

import base64



# decode the image coming from the request
def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded

