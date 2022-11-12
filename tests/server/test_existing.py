import pytest
from httpx import AsyncClient

from main import app


@pytest.mark.asyncio
async def test_features():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        params = {"workspace": "blank"}
        response = await ac.get("/features/existing", params=params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_objects():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        params = {"workspace": "blank"}
        response = await ac.get("/objects/existing", params=params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_pipelines():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        params = {"workspace": "blank"}
        response = await ac.get("/pipelines/existing", params=params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_analyzer():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        params = {"workspace": "blank"}
        response = await ac.get("/analyzer/existing", params=params)
    assert response.status_code == 200
