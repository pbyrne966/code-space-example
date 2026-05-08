from enum import Enum
from typing import Any

from requests import Response


class HttpMethods(Enum):
    POST = "POST"
    PUT = "PUT"
    GET = "GET"


def supported_http_method(http_method: str) -> str:
    upper_http_method = http_method.upper()
    try:
        HttpMethods(upper_http_method)
        return upper_http_method
    except Exception as err:
        raise ValueError("Could not serilize") from err


def serialize_response(given_response: Response) -> dict[str, Any]:
    try:
        given_response.raise_for_status()
        response_payload = given_response.json()
        if not isinstance(response_payload, dict):
            raise ValueError("Expected JSON object response")
        return response_payload
    except Exception as err:
        raise ValueError("Could not serilize") from err
