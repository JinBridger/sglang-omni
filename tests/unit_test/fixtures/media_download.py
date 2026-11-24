# SPDX-License-Identifier: Apache-2.0
"""HTTP transport fixture for media downloads through the real connector."""

import asyncio
from collections.abc import Iterator

import httpx
import pytest

from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    ResourceHTTPConnection,
    resource_connector_scope,
)


@pytest.fixture
def audio_download() -> (
    Iterator[tuple[list[httpx.Request], list[httpx.Response | httpx.HTTPError]]]
):
    requests: list[httpx.Request] = []
    responses: list[httpx.Response | httpx.HTTPError] = [
        httpx.Response(200, content=b"encoded audio")
    ]

    def download(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, httpx.HTTPError):
            raise response
        else:
            return response

    connection = ResourceHTTPConnection()
    connection.async_client = httpx.AsyncClient(transport=httpx.MockTransport(download))
    connector = MultiModalResourceConnector(connection=connection)
    try:
        with resource_connector_scope(connector):
            yield requests, responses
    finally:
        asyncio.run(connection.close())
