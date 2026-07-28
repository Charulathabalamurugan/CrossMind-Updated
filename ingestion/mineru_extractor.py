import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("crossmind.mineru")

class MinerUExtractor:
    def __init__(self):
        self.mineru_available = False
        try:
            import mineru
            self.mineru_available = True
            logger.info("MinerU available for scientific PDF extraction.")
        except ImportError:
            logger.info("MinerU not installed. PDF extraction disabled.")

        self.tika_available = False
        try:
            import tika
            self.tika_available = True
            logger.info("Apache Tika available as fallback.")
        except ImportError:
            logger.info("Apache Tika not installed. Tika fallback disabled.")

    def extract(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext in (".txt", ".md"):
            return self._extract_text(file_path)
        if ext in (".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"):
            return self._extract_office(file_path)
        if ext == ".eml":
            return self._extract_email(file_path)
        logger.warning(f"Unsupported extension: {ext}. Attempting text read.")
        return self._extract_text(file_path)

    def _extract_pdf(self, file_path: str) -> str:
        if self.mineru_available:
            try:
                from mineru import MinerU
                extractor = MinerU()
                result = extractor.extract(file_path)
                if result and hasattr(result, 'text') and result.text:
                    return result.text
                logger.warning(f"MinerU returned empty for {file_path}, trying fallback.")
            except Exception as exc:
                logger.warning(f"MinerU extraction failed for {file_path}: {exc}. Trying Tika fallback.")
        return self._extract_pdf_tika(file_path)

    def _extract_pdf_tika(self, file_path: str) -> str:
        if not self.tika_available:
            logger.warning(f"Tika not available for PDF: {file_path}")
            return ""
        try:
            from tika import parser
            parsed = parser.from_file(file_path)
            return parsed.get("content", "") or ""
        except Exception as exc:
            logger.error(f"Tika extraction failed for {file_path}: {exc}")
            return ""

    def _extract_text(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception as exc:
            logger.error(f"Error reading text file {file_path}: {exc}")
            return ""

    def _extract_office(self, file_path: str) -> str:
        if self.tika_available:
            try:
                from tika import parser
                parsed = parser.from_file(file_path)
                return parsed.get("content", "") or ""
            except Exception as exc:
                logger.error(f"Tika office extraction failed for {file_path}: {exc}")
        return ""

    def _extract_email(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            import email as email_lib
            msg = email_lib.message_from_string(raw)
            body = msg.get_body(preferencelist=("plain", "html"))
            if body:
                return body.get_content() or ""
            return raw
        except Exception as exc:
            logger.error(f"Email extraction failed for {file_path}: {exc}")
            return ""

_extractor_instance = None

def get_mineru_extractor() -> MinerUExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MinerUExtractor()
    return _extractor_instance
