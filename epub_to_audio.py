#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPUB -> MP3 (sesli kitap) dönüştürücü.

Örnekler:
    python epub_to_audio.py --input kitap.epub
    python epub_to_audio.py --input "out/*.epub" --voice tr-TR-EmelNeural
    python epub_to_audio.py --input kitap.epub --start 101 --end 200
    python epub_to_audio.py --input kitap.epub --count-chapters
    python epub_to_audio.py --list-voices tr-TR
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import ebooklib
import edge_tts
from bs4 import BeautifulSoup, FeatureNotFound
from ebooklib import epub

log = logging.getLogger("epub_to_audio")

DEFAULT_VOICE = "tr-TR-AhmetNeural"
DEFAULT_OUTPUT_DIR = "audio_out"

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCK_TAGS = HEADING_TAGS + [
    "p", "div", "li", "blockquote", "pre", "tr", "td", "th",
    "dt", "dd", "figcaption", "section", "article",
]
JUNK_TAGS = ["script", "style", "head", "nav", "svg", "img", "math"]

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))
_TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


# --------------------------------------------------------------------------
# EPUB okuma ve metin temizleme
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Chapter:
    index: int
    title: str
    text: str


def clean_line(text: str) -> str:
    """Unicode normalize eder, görünmez karakterleri atar, boşlukları sadeleştirir."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(content: bytes) -> tuple[str, str]:
    """HTML/XHTML içeriğinden (başlık, temiz düz metin) döndürür."""
    try:
        soup = BeautifulSoup(content, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(content, "html.parser")

    # Başlığı, <head> silinmeden önce belirle
    title = ""
    heading = soup.find(HEADING_TAGS)
    if heading:
        title = clean_line(heading.get_text(" "))
    if not title and soup.title:
        title = clean_line(soup.title.get_text(" "))

    for tag in soup(JUNK_TAGS):
        tag.decompose()

    # Başlıkların sonuna nokta ekle ki seslendirmede doğal bir duraklama olsun
    for h in soup.find_all(HEADING_TAGS):
        t = h.get_text(strip=True)
        if t and t[-1] not in ".!?…:;":
            h.append(".")

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for el in soup.find_all(BLOCK_TAGS):
        el.insert_before("\n")
        el.insert_after("\n")

    lines = (clean_line(line) for line in soup.get_text().splitlines())
    # "* * *", "---" gibi sadece sembol içeren satırları at
    text = "\n".join(l for l in lines if l and re.search(r"\w", l))
    return title, text


def _is_document(item) -> bool:
    return (
        item is not None
        and item.get_type() == ebooklib.ITEM_DOCUMENT
        and not isinstance(item, epub.EpubNav)
    )


def _iter_documents(book: epub.EpubBook) -> list:
    """Okuma sırasına (spine) göre doküman öğelerini döndürür."""
    ordered, seen = [], set()
    for entry in book.spine:
        idref = entry[0] if isinstance(entry, (tuple, list)) else entry
        item = book.get_item_with_id(idref)
        if idref not in seen and _is_document(item):
            seen.add(idref)
            ordered.append(item)
    if not ordered:  # spine bozuksa tüm dokümanları sırayla al
        ordered = [i for i in book.get_items() if _is_document(i)]
    return ordered


def extract_chapters(epub_path: Path, min_chars: int) -> list[Chapter]:
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    chapters: list[Chapter] = []
    for item in _iter_documents(book):
        try:
            title, text = html_to_text(item.get_content())
        except Exception as exc:  # noqa: BLE001
            log.warning("Bölüm atlandı (%s): %s", item.get_name(), exc)
            continue
        if len(text) < min_chars:
            log.debug("Kısa bölüm atlandı: %s (%d karakter)", item.get_name(), len(text))
            continue
        chapters.append(Chapter(len(chapters) + 1, title, text))
    return chapters


# --------------------------------------------------------------------------
# Dosya adı ve metin parçalama
# --------------------------------------------------------------------------
def slugify(text: str, max_len: int = 50) -> str:
    text = text.translate(_TR_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:max_len].strip("_")


def build_filename(ch: Chapter, width: int) -> str:
    slug = slugify(ch.title) or f"Bolum_{ch.index}"
    return f"{ch.index:0{width}d}_{slug}.mp3"


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Çok uzun tek bir cümleyi kelime sınırlarından böler."""
    parts, cur = [], ""
    for word in sentence.split(" "):
        if len(word) > max_chars:  # patolojik durum: boşluksuz dev kelime
            if cur:
                parts.append(cur)
                cur = ""
            parts.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
        elif cur and len(cur) + 1 + len(word) > max_chars:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        parts.append(cur)
    return parts


def split_text(text: str, max_chars: int) -> list[str]:
    """Metni paragraf/cümle sınırlarını koruyarak en fazla max_chars'lık parçalara böler."""
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for paragraph in text.split("\n"):
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            pieces = [sentence] if len(sentence) <= max_chars else _hard_sp
