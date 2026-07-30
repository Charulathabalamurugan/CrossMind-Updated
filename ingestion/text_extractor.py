import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger("crossmind.text_extractor")

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".eml", ".html", ".xml"}

class TextExtractor:
    def __init__(self):
        self._pdf_available = False
        self._docx_available = False
        self._tika_available = False
        self._tika_server_url = None
        try:
            import PyPDF2
            self._pdf_available = True
            self._pdf_lib = PyPDF2
        except ImportError:
            logger.info("PyPDF2 not installed. PDF extraction disabled.")
        try:
            import docx
            self._docx_available = True
        except ImportError:
            logger.info("python-docx not installed. DOCX extraction disabled.")
        try:
            import tika
            from tika import parser as tika_parser
            self._tika_available = True
            self._tika_parser = tika_parser
            self._tika_server_url = os.getenv("TIKA_SERVER_URL", "http://localhost:9998")
            logger.info("Apache Tika available for extraction fallback.")
        except ImportError:
            logger.info("tika-python not installed. Apache Tika fallback disabled.")

    def extract(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            return self._extract_txt(file_path)
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext == ".docx":
            return self._extract_docx(file_path)
        if ext == ".md":
            return self._extract_txt(file_path)
        if ext == ".csv":
            return self._extract_csv(file_path)
        if ext == ".json":
            return self._extract_json(file_path)
        if ext == ".eml":
            return self._extract_email(file_path)
        if ext in (".html", ".xml"):
            return self._extract_html(file_path)
        logger.warning(f"Unsupported file extension: {ext}. Attempting plain text read.")
        return self._extract_txt(file_path)

    def _extract_with_tika(self, file_path: str) -> str:
        if not self._tika_available:
            logger.warning("Apache Tika not available. Install tika-python package.")
            return ""
        try:
            parsed = self._tika_parser.from_file(file_path, serverEndpoint=self._tika_server_url)
            return parsed.get("content", "") or ""
        except Exception as exc:
            logger.error(f"Apache Tika extraction failed for {file_path}: {exc}")
            return ""

    def _extract_txt(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception as exc:
            logger.error(f"Error reading text file {file_path}: {exc}")
            return ""

    def _extract_pdf(self, file_path: str) -> str:
        if self._pdf_available:
            try:
                reader = self._pdf_lib.PdfReader(file_path)
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                if text_parts:
                    return "\n".join(text_parts)
            except Exception as exc:
                logger.error(f"PyPDF2 extraction failed for {file_path}: {exc}")
        return self._extract_with_tika(file_path)

    def _extract_docx(self, file_path: str) -> str:
        if self._docx_available:
            try:
                import docx
                doc = docx.Document(file_path)
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                if paragraphs:
                    return "\n".join(paragraphs)
            except Exception as exc:
                logger.error(f"python-docx extraction failed for {file_path}: {exc}")
        return self._extract_with_tika(file_path)

    def _extract_csv(self, file_path: str) -> str:
        try:
            import csv
            with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.reader(fh)
                rows = [" | ".join(row) for row in reader]
            return "\n".join(rows)
        except Exception as exc:
            logger.error(f"Error extracting CSV {file_path}: {exc}")
            return ""

    def _extract_json(self, file_path: str) -> str:
        try:
            import json
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Error extracting JSON {file_path}: {exc}")

    def _extract_email(self, file_path: str) -> str:
        return self._extract_with_tika(file_path) or self._extract_txt(file_path)

    def _extract_html(self, file_path: str) -> str:
        return self._extract_with_tika(file_path) or self._extract_txt(file_path)

_extractor_instance = None

def get_text_extractor() -> TextExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = TextExtractor()
    return _extractor_instance