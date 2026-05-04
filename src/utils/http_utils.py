from enum import Enum

from requests import Response


class HttpMethods(Enum):
    POST = "POST"
    PUT = "PUT"
    GET = "GET"


def supported_http_method(http_method: str):
    upper_http_method = http_method.upper()
    try:
        HttpMethods(upper_http_method)
        return upper_http_method
    except Exception as err:
        raise ValueError("Could not serilize") from err


def serialize_response(given_response: Response):
    status_code = given_response.status_code
    if status_code >= 400:
        raise ValueError("Http Request Failed")
    try:
        # Seriliaze to a given pydantic type
        return given_response.json()
    except Exception as err:
        raise ValueError("Could not serilize") from err
