import ssl

from neo4j import GraphDatabase
from pydantic import SecretStr

from app.config import settings
from app.services import neo4j_service


def test_secure_neo4j_uri_uses_verified_system_and_public_ca_context(monkeypatch):
    created = {}

    class _Driver:
        pass

    def create_driver(uri, *, auth, **options):
        created.update(uri=uri, auth=auth, options=options)
        return _Driver()

    monkeypatch.setattr(settings, "NEO4J_URI", "neo4j+s://db.example.test")
    monkeypatch.setattr(settings, "NEO4J_USERNAME", "neo4j")
    monkeypatch.setattr(settings, "NEO4J_PASSWORD", SecretStr("test-password"))
    monkeypatch.setattr(GraphDatabase, "driver", create_driver)
    monkeypatch.setattr(neo4j_service, "_driver", None)

    try:
        driver = neo4j_service._get_driver()
        context = created["options"]["ssl_context"]

        assert isinstance(driver, _Driver)
        assert created["uri"] == "neo4j://db.example.test"
        assert created["auth"] == ("neo4j", "test-password")
        assert created["options"]["encrypted"] is True
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert context.get_ca_certs()
    finally:
        monkeypatch.setattr(neo4j_service, "_driver", None)


def test_neo4j_health_reports_missing_credentials_as_unhealthy(monkeypatch):
    monkeypatch.setattr(settings, "NEO4J_URI", "neo4j+s://db.example.test")
    monkeypatch.setattr(settings, "NEO4J_USERNAME", "")
    monkeypatch.setattr(settings, "NEO4J_PASSWORD", SecretStr(""))
    monkeypatch.setattr(neo4j_service, "_driver", None)

    health = neo4j_service.neo4j_health()

    assert health == {"configured": True, "status": "unhealthy"}


def test_unconfigured_neo4j_remains_disabled(monkeypatch):
    monkeypatch.setattr(settings, "NEO4J_URI", "")
    monkeypatch.setattr(neo4j_service, "_driver", None)

    assert neo4j_service.neo4j_health() == {
        "configured": False,
        "status": "disabled",
    }
