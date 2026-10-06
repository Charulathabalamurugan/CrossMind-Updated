"""Public scientific literature adapters and query-driven API orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import httpx

from config import settings
from ingestion.pipeline import IngestionPipeline
from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline

logger = logging.getLogger("crossmind.public_research")

KEYWORD_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9'_-]{0,127}", re.UNICODE
)


def extract_keywords(query: str) -> List[str]:
    """Extract meaningful, de-duplicated search terms from a query."""
    keywords = []
    stop_words = {
        "what", "role", "does", "play", "in", "the", "and", "or", "for", "of",
        "with", "from", "to", "across", "is", "are", "this", "that", "how",
        "research", "question", "query", "scientific", "find", "about", "on",
    }
    for match in KEYWORD_PATTERN.finditer(query):
        term = match.group(0).strip("'\"")
        normalized = term.lower()
        if len(term) < 2 or normalized in stop_words:
            continue
        if normalized.endswith("s") and normalized[:-1] in {"gene", "protein", "disease"}:
            pass
        if term not in keywords:
            keywords.append(term)
    return keywords


_BIOLOGICAL_TERMS = (
    "protein", "gene", "disease", "clinical", "biomedical", "medical", "parkinson",
    "cancer", "tumor", "neuroscience", "genomics", "proteomics", "pathway",
)


def _infer_domain(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in _BIOLOGICAL_TERMS):
        return "biomedical"
    if any(term in lowered for term in ("materials", "crystal", "compound", "chemistry", "polymer")):
        return "materials"
    return "general"


def _crossref_year(published: Any) -> int:
    if isinstance(published, dict):
        date_parts = published.get("date-parts") or []
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (ValueError, TypeError, IndexError):
                return 2024
    return 2024


class QueryRouter:
    """Select public research APIs from query semantics without API keys."""

    def choose(self, query: str) -> Dict[str, Any]:
        text = query.lower()
        keywords = extract_keywords(query)
        bio_terms = ("protein", "gene", "disease", "clinical", "biomedical", "medical", "parkinson", "alpha-synuclein")
        computational_terms = ("neural", "network", "algorithm", "model", "simulation", "computer")
        physics_terms = ("quantum", "physics", "materials", "chemistry")

        if any(term in text for term in bio_terms):
            primary = "europe_pmc"
            secondary = ["pubmed", "openalex", "crossref", "unpaywall"]
        elif any(term in text for term in physics_terms) or any(term in text for term in computational_terms):
            primary = "arxiv"
            secondary = ["openalex", "semantic_scholar", "materials_project"]
        else:
            primary = "openalex"
            secondary = ["europe_pmc", "semantic_scholar", "crossref", "unpaywall"]

        return {
            "primary": primary,
            "secondary": secondary,
            "keywords": keywords,
            "source": "query_router",
        }


class Provider:
    def __init__(self, name: str, base_url: str, timeout: float = 10.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class EuropePMCProvider(Provider):
    """Europe PMC search and metadata adapter; requires no API key."""

    def __init__(self, timeout: float = 10.0):
        super().__init__("europe_pmc", "https://www.ebi.ac.uk/europepmc/webservices/rest/search", timeout)

    def build_search_request(self, query: str) -> Dict[str, Any]:
        return {
            "method": "GET",
            "url": self.base_url,
            "params": {
                "query": _build_biomedical_query(query),
                "format": "json",
                "resultType": "core",
                "pageSize": 10,
            },
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        request = self.build_search_request(query)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(request["url"], params=request["params"])
            response.raise_for_status()
            payload = response.json()
        results = payload.get("resultList", {}).get("result", [])
        return [normalize_europe_pmc_result(item, source=self.name) for item in results[:max_results]]


class PubMedProvider(Provider):
    """PubMed E-utilities provider; requires no API key."""

    def __init__(self, timeout: float = 10.0):
        super().__init__("pubmed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", timeout)

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        search = urlencode({"db": "pubmed", "term": _build_biomedical_query(query), "retmode": "json", "retmax": max_results})
        with httpx.Client(timeout=self.timeout) as client:
            search_response = client.get(f"{self.base_url}/esearch.fcgi?{search}")
            search_response.raise_for_status()
            search_payload = search_response.json()
            pmids = search_payload.get("esearchresult", {}).get("idlist", [])[:max_results]
            if not pmids:
                return []
            fetch = urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
            fetch_response = client.get(f"{self.base_url}/efetch.fcgi?{fetch}")
            fetch_response.raise_for_status()
            return [_normalize_pubmed_xml(item, source=self.name) for item in _parse_pubmed_articles(fetch_response.text)[:max_results]]


class OpenAlexProvider(Provider):
    def __init__(self, timeout: float = 10.0):
        super().__init__("openalex", "https://api.openalex.org/works", timeout)

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.base_url, params={"search": query, "per_page": max_results})
            response.raise_for_status()
            payload = response.json()
        return [
            {
                "id": str(item.get("id", "")),
                "title": str(item.get("title", "Untitled work")),
                "content": str(item.get("abstract_inverted_index", "")),
                "domain": "general",
                "year": int(item.get("publication_year", 2024)),
                "authors": [str(author.get("name", "")) for author in item.get("authorships", []) if author.get("author", {}).get("name")],
                "tags": [str(tag) for tag in item.get("concepts", [])],
                "source": self.name,
                "metadata": {
                    "doi": item.get("doi"),
                    "url": item.get("primary_location", {}).get("landing_page_url"),
                    "cited_by_count": item.get("cited_by_count"),
                },
            }
            for item in payload.get("results", [])[:max_results]
        ]


class ArxivProvider(Provider):
    def __init__(self, timeout: float = 10.0):
        super().__init__("arxiv", "http://export.arxiv.org/api/query", timeout)

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            response = client.get(self.base_url, params={"search_query": f"all:{query}", "max_results": max_results})
            response.raise_for_status()
            payload = response.text
        return _parse_arxiv_entries(payload, source=self.name)


class SemanticScholarProvider(Provider):
    def __init__(self, timeout: float = 10.0):
        super().__init__("semantic_scholar", "https://api.semanticscholar.org/graph/v1/paper/search", timeout)

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.base_url, params={"query": query, "limit": max_results, "fields": "title,abstract,year,authors,url,externalIds"})
            response.raise_for_status()
            payload = response.json()
        return [
            {
                "id": str(item.get("id", "")),
                "title": str(item.get("title", "Untitled paper")),
                "content": str(item.get("abstract", "")),
                "domain": "general",
                "year": int(item.get("year", 2024)),
                "authors": [str(author.get("name", "")) for author in item.get("authors", []) if author.get("name")],
                "tags": ["semantic_scholar"],
                "source": self.name,
                "metadata": {"url": item.get("url"), "external_ids": item.get("externalIds", {})},
            }
            for item in payload.get("data", [])[:max_results]
        ]


def normalize_europe_pmc_result(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
    full_text_url = next((entry.get("url", "") for entry in full_text_urls if entry.get("url")), "")
    authors = []
    for author in item.get("authorList", {}).get("author", []):
        name = author.get("name") or author.get("displayName")
        if name:
            authors.append(str(name))
    abstract = str(item.get("abstract", "")).strip()
    title = str(item.get("title", "Untitled paper"))
    content = abstract or title
    if full_text_url:
        content += f"\nFull-text source: {full_text_url}"
    return {
        "id": str(item.get("id", "")) or str(item.get("pmcid", "")),
        "title": title,
        "content": content,
        "domain": "biomedical",
        "year": _parse_year(item.get("date")),
        "authors": authors,
        "tags": ["europe_pmc", "biomedical"],
        "allowed_roles": ["public", "researcher"],
        "source": source,
        "metadata": {"full_text_url": full_text_url, "pmcid": item.get("pmcid")},
    }


def _build_biomedical_query(query: str) -> str:
    return " AND ".join(extract_keywords(query)[:10])


def _parse_year(value: Any) -> int:
    if not value:
        return 2024
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else 2024


def _normalize_pubmed_xml(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "id": str(item.get("pmid", "")),
        "title": str(item.get("title", "Untitled PubMed article")),
        "content": str(item.get("abstract", "")),
        "domain": "biomedical",
        "year": _parse_year(item.get("date")),
        "authors": item.get("authors", []),
        "tags": ["pubmed", "biomedical"],
        "allowed_roles": ["public", "researcher"],
        "source": source,
        "metadata": {"pmid": item.get("pmid"), "url": item.get("url")},
    }


def _parse_pubmed_articles(xml: str) -> List[Dict[str, Any]]:
    articles = []
    for article in re.findall(r"<ArticleSet[^>]*>(.*?)</ArticleSet>", xml, re.DOTALL | re.IGNORECASE):
        pmid_match = re.search(r"<PubDate[^>]*>.*?</PubDate>", article, re.DOTALL)
        pmid = re.search(r"<PMID>(\d+)</PMID>", article)
        title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
        abstract = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
        if pmid:
            articles.append({
                "pmid": pmid.group(1),
                "title": _decode_xml(title.group(1)) if title else "Untitled PubMed article",
                "abstract": _decode_xml(abstract.group(1)) if abstract else "",
                "date": pmid_match.group(0) if pmid_match else "",
            })
    return articles


def _decode_xml(value: str) -> str:
    import html
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()


def _parse_arxiv_entries(xml: str, source: str) -> List[Dict[str, Any]]:
    entries = []
    for block in re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL | re.IGNORECASE):
        title = re.search(r"<title>(.*?)</title>", block, re.DOTALL | re.IGNORECASE)
        summary = re.search(r"<summary>(.*?)</summary>", block, re.DOTALL | re.IGNORECASE)
        link = re.search(r"<link[^>]+href=\"([^\"]+)\"", block, re.IGNORECASE)
        date = re.search(r"<published>(.*?)</published>", block, re.DOTALL | re.IGNORECASE)
        if title:
            entries.append({
                "id": str(link.group(1)) if link else str(hashlib.sha256(block.encode()).hexdigest()),
                "title": _decode_xml(title.group(1)),
                "content": _decode_xml(summary.group(1)) if summary else "",
                "domain": "computer_science",
                "year": _parse_year(date.group(1) if date else ""),
                "authors": [],
                "tags": [source, "preprint"],
                "allowed_roles": ["public", "researcher"],
                "source": source,
                "metadata": {"url": link.group(1) if link else ""},
            })
    return entries


class CrossrefProvider(Provider):
    """Crossref /works search; no API key required for unauthenticated queries."""

    def __init__(self, timeout: float = 10.0):
        super().__init__("crossref", "https://api.crossref.org/works", timeout)

    def build_search_request(self, query: str, rows: int = 10) -> Dict[str, Any]:
        return {
            "method": "GET",
            "url": self.base_url,
            "params": {
                "query": query,
                "rows": rows,
                "select": "dois,title,abstract,author,published,URL,type",
            },
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        request = self.build_search_request(query, rows=max_results)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(request["url"], params=request["params"])
            response.raise_for_status()
            payload = response.json()
        message = payload.get("message", {}) if payload else {}
        items = message.get("items", [])
        return [normalize_crossref_result(item, source=self.name) for item in items[:max_results]]


def normalize_crossref_result(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    titles = item.get("title")
    if isinstance(titles, list) and titles:
        title = str(titles[0])
    elif isinstance(titles, str):
        title = titles
    else:
        title = "Untitled work"
    authors = []
    for author in item.get("author", []):
        given = (author.get("given") or "").strip()
        family = (author.get("family") or "").strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    abstract = str(item.get("abstract", "") or "").strip()
    content = _decode_xml(abstract) if abstract else title
    year = _crossref_year(item.get("published"))
    domain = _infer_domain(f"{title} {abstract}")
    type_label = str(item.get("type", "") or "")
    return {
        "id": str(item.get("DOI", "")),
        "title": title,
        "content": content,
        "domain": domain,
        "year": year,
        "authors": authors,
        "tags": ["crossref", type_label] if type_label else ["crossref"],
        "allowed_roles": ["public", "researcher"],
        "source": source,
        "metadata": {
            "doi": item.get("DOI"),
            "url": item.get("URL") or item.get("url"),
            "type": type_label,
        },
    }


class UnpaywallProvider(Provider):
    """Unpaywall free-full-text resolver keyed by DOI; resolves DOIs via Crossref."""

    def __init__(
        self,
        timeout: float = 10.0,
        email: Optional[str] = None,
        crossref_provider: Optional[CrossrefProvider] = None,
    ):
        super().__init__("unpaywall", "https://api.unpaywall.org/v2", timeout)
        self.email = email or os.environ.get("UNPAYWALL_EMAIL", "crossmind@example.org")
        self._crossref = crossref_provider or CrossrefProvider(timeout=self.timeout)

    def build_lookup_request(self, doi: str) -> Dict[str, Any]:
        return {
            "method": "GET",
            "url": f"{self.base_url}/{doi}",
            "params": {"mailto": self.email, "format": "json"},
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        crossref_docs = self._crossref.search(query, max_results=max_results)
        seen_dois: set = set()
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for doc in crossref_docs:
                doi = (doc.get("metadata") or {}).get("doi")
                if not doi or doi in seen_dois:
                    continue
                seen_dois.add(doi)
                try:
                    request = self.build_lookup_request(doi)
                    response = client.get(request["url"], params=request["params"])
                    response.raise_for_status()
                    payload = response.json()
                except Exception as exc:
                    logger.warning("Unpaywall lookup failed for DOI %s: %s", doi, exc)
                    continue
                documents.append(normalize_unpaywall_result(payload, source=self.name))
                if len(documents) >= max_results:
                    break
        return documents


def normalize_unpaywall_result(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    doi = str(item.get("doi", "") or "")
    title = str(item.get("title") or "Untitled work")
    oa_url = ""
    license_url = ""
    for location in item.get("oa_locations", []) or []:
        if not oa_url:
            url = location.get("url") or location.get("url_for_pdf")
            if url:
                oa_url = str(url)
        if not license_url:
            license_url = str(location.get("license") or location.get("license_url") or "")
    oa_status = str(item.get("oa_status", "") or "")
    tags = ["unpaywall", "open_access"]
    if oa_status:
        tags.append(oa_status)
    description = title
    if oa_url:
        description += f"\nOpen-access source: {oa_url}"
    return {
        "id": doi or f"unpaywall:{str(item.get('paper_id', ''))}",
        "title": title,
        "content": description,
        "domain": "general",
        "year": _parse_year(item.get("year")),
        "authors": [],
        "tags": tags,
        "allowed_roles": ["public", "researcher"],
        "source": source,
        "metadata": {"doi": doi, "oa_url": oa_url, "license": license_url},
    }


class MaterialsProjectProvider(Provider):
    """Materials Project materials search; requires MATERIALS_PROJECT_API_KEY."""

    def __init__(self, timeout: float = 30.0):
        super().__init__("materials_project", "https://api.materialsproject.org/v1", timeout)
        self.api_key = os.environ.get("MATERIALS_PROJECT_API_KEY", "")

    def build_search_request(self, query: str) -> Dict[str, Any]:
        return {
            "method": "POST",
            "url": f"{self.base_url}/summary/search",
            "headers": {"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            "json": {
                "criteria": {"description": {"$regex": query}},
                "fields": ["material_id", "pretty_formula", "formation_energy_per_atom"],
            },
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("Materials Project API key is required (set MATERIALS_PROJECT_API_KEY)")
        request = self.build_search_request(query)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                request["url"], headers=request["headers"], json=request["json"]
            )
            response.raise_for_status()
            payload = response.json()
        raw_docs = payload.get("data", [])
        return [normalize_materials_project_result(item, source=self.name) for item in raw_docs[:max_results]]


def normalize_materials_project_result(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    material_id = str(item.get("material_id", ""))
    formula = (str(item.get("pretty_formula", "") or "").strip()) or "Unknown"
    energy = item.get("formation_energy_per_atom")
    content = f"Material {formula} ({material_id}) with formation energy {energy} eV/atom."
    return {
        "id": material_id,
        "title": f"Materials Project: {formula} ({material_id})",
        "content": content,
        "domain": "materials",
        "year": 2024,
        "authors": [],
        "tags": ["materials_project", "materials"],
        "allowed_roles": ["public", "researcher"],
        "source": source,
        "metadata": {
            "material_id": material_id,
            "formula": formula,
            "properties": {"formation_energy_per_atom": energy},
        },
    }


class ResearchOrchestrator:
    """Search public literature, ingest normalized records, then run CrossMind."""

    def __init__(self, router: Optional[QueryRouter] = None):
        self.router = router or QueryRouter()
        self.providers = {
            "europe_pmc": EuropePMCProvider(),
            "pubmed": PubMedProvider(),
            "openalex": OpenAlexProvider(),
            "arxiv": ArxivProvider(),
            "semantic_scholar": SemanticScholarProvider(),
            "crossref": CrossrefProvider(),
            "unpaywall": UnpaywallProvider(),
            "materials_project": MaterialsProjectProvider(),
        }

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        decision = self.router.choose(query)
        documents: List[Dict[str, Any]] = []
        provider_results: Dict[str, List[Dict[str, Any]]] = {}
        for provider_name in [decision["primary"], *decision["secondary"]]:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            try:
                normalized = provider.search(query, max_results=max_results)
                provider_results[provider_name] = normalized
                documents.extend(normalized)
            except Exception as exc:
                logger.warning("Public provider %s failed: %s", provider_name, exc)
        return {
            "decision": decision,
            "provider_results": provider_results,
            "documents": documents,
        }

    def orchestrate(
        self,
        query: str,
        max_results: int = 5,
        user_role: str = "researcher",
        session_id: str = "public-research",
    ) -> Dict[str, Any]:
        """Run the typed public-research sequence: route → search → ingest → reason."""
        search_result = self.search(query, max_results=max_results)
        documents = search_result["documents"]
        inserted_ids: List[str] = []
        if documents:
            inserted_ids = IngestionPipeline().ingest_documents(documents)
        reasoning = get_neuro_symbolic_pipeline().process_query(
            query=query,
            user_role=user_role,
            confidence_thresholds={"proceed": 0.75, "investigate": 0.5},
            session_id=session_id,
        )
        return {
            "query": query,
            "decision": search_result["decision"],
            "provider_results": search_result["provider_results"],
            "ingested_count": len(inserted_ids),
            "inserted_ids": inserted_ids,
            "reasoning": reasoning,
        }


def get_research_orchestrator() -> ResearchOrchestrator:
    return ResearchOrchestrator()


__all__ = [
    "ArxivProvider",
    "CrossrefProvider",
    "EuropePMCProvider",
    "MaterialsProjectProvider",
    "OpenAlexProvider",
    "PubMedProvider",
    "QueryRouter",
    "ResearchOrchestrator",
    "SemanticScholarProvider",
    "UnpaywallProvider",
    "extract_keywords",
    "get_research_orchestrator",
    "normalize_crossref_result",
    "normalize_europe_pmc_result",
    "normalize_materials_project_result",
    "normalize_unpaywall_result",
]
