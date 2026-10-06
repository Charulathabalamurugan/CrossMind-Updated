from app.main import app
from app.schemas import PublicResearchQueryRequest
from ingestion.public_research import (
    CrossrefProvider,
    EuropePMCProvider,
    MaterialsProjectProvider,
    QueryRouter,
    UnpaywallProvider,
    extract_keywords,
    normalize_crossref_result,
    normalize_europe_pmc_result,
    normalize_materials_project_result,
    normalize_unpaywall_result,
)


def test_biomedical_query_selects_primary_apis():
    router = QueryRouter()
    decision = router.choose("What role does alpha-synuclein play in Parkinson's?")

    assert decision["primary"] == "europe_pmc"
    assert "pubmed" in decision["secondary"]
    assert "openalex" in decision["secondary"]
    assert decision["keywords"] == ["alpha-synuclein", "Parkinson's"]


def test_router_classifies_quantum_and_computational_queries():
    router = QueryRouter()

    quantum = router.choose("Quantum materials and neural network simulation")
    assert quantum["primary"] == "arxiv"
    assert "openalex" in quantum["secondary"]


def test_europe_pmc_normalization_produces_crossmind_document():
    raw = {
        "id": "PMC123",
        "title": "Alpha-synuclein and Parkinson disease",
        "abstract": "A model of alpha-synuclein pathology.",
        "authorList": {"author": [{"name": "Ada Lovelace"}]},
        "date": "2026-01-01",
        "fullTextUrlList": {"fullTextUrl": [{"url": "https://example.test/full.xml"}]},
    }

    document = normalize_europe_pmc_result(raw, source="europe_pmc")

    assert document["title"] == "Alpha-synuclein and Parkinson disease"
    assert document["content"].startswith("A model of alpha-synuclein pathology")
    assert document["domain"] == "biomedical"
    assert document["authors"] == ["Ada Lovelace"]
    assert document["source"] == "europe_pmc"


def test_europe_pmc_normalization_uses_title_when_abstract_is_missing():
    document = normalize_europe_pmc_result(
        {
            "id": "PMC456",
            "title": "Neural network constraints",
            "authorList": {"author": []},
        },
        source="europe_pmc",
    )

    assert document["content"] == "Neural network constraints"
    assert document["authors"] == []


def test_europe_pmc_provider_search_uses_no_key_request():
    provider = EuropePMCProvider(timeout=1.0)
    request = provider.build_search_request("alpha-synuclein Parkinson's")

    assert request["url"].startswith("https://www.ebi.ac.uk/europepmc/webservices/rest/search")
    assert request["params"]["format"] == "json"
    assert request["params"]["resultType"] == "core"
    assert request["params"]["query"] == "alpha-synuclein AND Parkinson's"


def test_keyword_sequence_preserves_research_terms():
    keywords = extract_keywords("What role does alpha-synuclein play in Parkinson's?")

    assert keywords == ["alpha-synuclein", "Parkinson's"]


def test_public_research_route_is_registered_and_validated():
    route_paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert "/api/public-research/query" in route_paths
    request = PublicResearchQueryRequest(query="Alpha-synuclein and Parkinson's")
    assert request.max_results == 5
    assert request.user_role == "researcher"


def test_crossref_normalization_produces_crossmind_document():
    raw = {
        "DOI": "10.1038/test-crossref",
        "title": ["A crossref sample paper"],
        "abstract": "<p>Crossref abstracts use JATS XML.</p>",
        "author": [{"given": "Ada", "family": "Lovelace"}, {"family": "Byron"}],
        "published": {"date-parts": [[2023, 4, 1]]},
        "URL": "https://doi.org/10.1038/test-crossref",
        "type": "journal-article",
    }

    document = normalize_crossref_result(raw, source="crossref")

    assert document["id"] == "10.1038/test-crossref"
    assert document["title"] == "A crossref sample paper"
    assert document["content"].startswith("Crossref abstracts use JATS XML")
    assert document["year"] == 2023
    assert document["authors"] == ["Ada Lovelace", "Byron"]
    assert "crossref" in document["tags"]
    assert document["metadata"]["doi"] == "10.1038/test-crossref"


def test_crossref_provider_build_request_is_keyless():
    provider = CrossrefProvider(timeout=1.0)
    request = provider.build_search_request("battery storage")

    assert request["url"] == "https://api.crossref.org/works"
    assert request["params"]["query"] == "battery storage"
    assert request["params"]["select"] == "dois,title,abstract,author,published,URL,type"


def test_crossref_provider_search_returns_normalized_documents(monkeypatch):
    provider = CrossrefProvider(timeout=1.0)

    captured = {}

    def fake_get(self, url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        resp = {"message": {"items": [
            {"DOI": "10.1000/x1", "title": ["Title one"], "abstract": "abstract one"},
            {"DOI": "10.1000/x2", "title": ["Title two"], "abstract": ""},
        ]}}

        class R:
            def json(self):
                return resp

            def raise_for_status(self):
                return None

        return R()

    import httpx
    monkeypatch.setattr(httpx.Client, "get", fake_get)

    docs = provider.search("battery energy storage", max_results=5)

    assert captured["url"] == "https://api.crossref.org/works"
    assert captured["params"]["query"] == "battery energy storage"
    assert [d["id"] for d in docs] == ["10.1000/x1", "10.1000/x2"]
    assert docs[0]["title"] == "Title one"
    assert "crossref" in docs[0]["tags"]


def test_unpaywall_normalization_extracts_open_access_url():
    raw = {
        "doi": "10.1000/unpaywall-demo",
        "title": "Open access demo paper",
        "year": "2022",
        "oa_status": "gold",
        "oa_locations": [
            {"url": "https://repository.example/x.pdf", "license": "https://creativecommons.org/licenses/by/4.0/"},
            {"url_for_pdf": "https://mirror.example/y.pdf"},
        ],
    }

    document = normalize_unpaywall_result(raw, source="unpaywall")

    assert document["id"] == "10.1000/unpaywall-demo"
    assert document["title"] == "Open access demo paper"
    assert document["year"] == 2022
    assert document["metadata"]["oa_url"] == "https://repository.example/x.pdf"
    assert document["metadata"]["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert "open_access" in document["tags"]


def test_unpaywall_provider_build_lookup_request():
    provider = UnpaywallProvider(timeout=1.0, email="tester@example.org")
    request = provider.build_lookup_request("10.1000/xyz")

    assert request["url"] == "https://api.unpaywall.org/v2/10.1000/xyz"
    assert request["params"]["mailto"] == "tester@example.org"
    assert request["params"]["format"] == "json"


def test_unpaywall_provider_search_resolves_doi_to_open_access(monkeypatch):
    crossref = CrossrefProvider(timeout=1.0)
    monkeypatch.setattr(crossref, "search", lambda query, max_results=5: [
        {"metadata": {"doi": "10.1000/unpaywall-demo"}},
    ])

    provider = UnpaywallProvider(timeout=1.0, crossref_provider=crossref)

    def fake_get(self, url, params=None, **kwargs):
        assert url == "https://api.unpaywall.org/v2/10.1000/unpaywall-demo"
        class R:
            def json(self):
                return {
                    "doi": "10.1000/unpaywall-demo",
                    "title": "OA paper",
                    "year": "2021",
                    "oa_status": "gold",
                    "oa_locations": [{"url": "https://example.test/oa.pdf", "license": "https://example.org/by.nc/4.0"}],
                }

            def raise_for_status(self):
                return None

        return R()

    import httpx
    monkeypatch.setattr(httpx.Client, "get", fake_get)

    docs = provider.search("some query", max_results=5)

    assert len(docs) == 1
    assert docs[0]["metadata"]["oa_url"] == "https://example.test/oa.pdf"
    assert docs[0]["metadata"]["license"] == "https://example.org/by.nc/4.0"


def test_materials_project_normalization_builds_material_document():
    raw = {
        "material_id": "mp-149",
        "pretty_formula": "Au",
        "formation_energy_per_atom": -3.21,
    }

    document = normalize_materials_project_result(raw, source="materials_project")

    assert document["id"] == "mp-149"
    assert document["title"] == "Materials Project: Au (mp-149)"
    assert "Au" in document["content"]
    assert document["domain"] == "materials"
    assert document["metadata"]["formula"] == "Au"
    assert document["metadata"]["properties"]["formation_energy_per_atom"] == -3.21


def test_materials_project_provider_requires_api_key():
    provider = MaterialsProjectProvider(timeout=1.0)
    assert provider.api_key == ""
    import pytest
    with pytest.raises(RuntimeError):
        provider.search("silicon", max_results=1)


def test_materials_project_provider_build_request_includes_api_key():
    provider = MaterialsProjectProvider(timeout=1.0)
    provider.api_key = "test-key"
    request = provider.build_search_request("silicon carbide")

    assert request["url"] == "https://api.materialsproject.org/v1/summary/search"
    assert request["headers"]["X-API-KEY"] == "test-key"
    assert request["json"]["criteria"]["description"]["$regex"] == "silicon carbide"


def test_orchestrator_registers_crossref_unpaywall_and_materials_project():
    from ingestion.public_research import get_research_orchestrator

    provider_names = get_research_orchestrator().providers.keys()

    assert "crossref" in provider_names
    assert "unpaywall" in provider_names
    assert "materials_project" in provider_names


def test_router_includes_new_providers_in_secondary_selection():
    router = QueryRouter()

    biomedical = router.choose("What role does alpha-synuclein play in Parkinson's?")
    assert "crossref" in biomedical["secondary"]
    assert "unpaywall" in biomedical["secondary"]

    physics = router.choose("Quantum materials and neural network simulation")
    assert "materials_project" in physics["secondary"]


