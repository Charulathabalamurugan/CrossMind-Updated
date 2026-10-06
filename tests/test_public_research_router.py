from app.main import app
from app.schemas import PublicResearchQueryRequest
from ingestion.public_research import (
    EuropePMCProvider,
    QueryRouter,
    extract_keywords,
    normalize_europe_pmc_result,
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
