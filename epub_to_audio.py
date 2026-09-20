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
            pieces = [sentence] if len(sentence) <= max_chars else _hard_split(sentence, max_chars)
            for piece in pieces:
                if buf and len(buf) + 1 + len(piece) > max_chars:
                    flush()
                sep = "" if (not buf or buf.endswith("\n")) else " "
                buf = f"{buf}{sep}{piece}"
        if buf and not buf.endswith("\n"):
            buf += "\n"  # paragraf sonu
    flush()
    return chunks


# --------------------------------------------------------------------------
# Seslendirme (edge-tts)
# --------------------------------------------------------------------------
async def synth_chunk(text: str, voice: str, rate: str, retries: int = 5) -> bytes:
    """Tek bir metin parçasını MP3 baytlarına çevirir; ağ hatalarında yeniden dener."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            audio = bytearray()
            async for msg in edge_tts.Communicate(text, voice, rate=rate).stream():
                if msg["type"] == "audio":
                    audio.extend(msg["data"])
            if not audio:
                raise RuntimeError("servis ses döndürmedi")
            return bytes(audio)
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = min(2 ** attempt, 30)
            log.warning("Parça %d/%d başarısız (%s); %d sn sonra tekrar", attempt, retries, exc, wait)
            await asyncio.sleep(wait)
    raise RuntimeError(f"ses üretilemedi: {last}")


async def convert_chapter(ch: Chapter, out_path: Path, args, sem: asyncio.Semaphore) -> bool:
    async with sem:
        try:
            parts = [
                await synth_chunk(chunk, args.voice, args.rate)
                for chunk in split_text(ch.text, args.max_chars)
            ]
            tmp = out_path.with_suffix(".mp3.part")
            tmp.write_bytes(b"".join(parts))  # MP3 kareleri art arda eklenince sorunsuz çalar
            tmp.replace(out_path)
            log.info("Tamam: %s (%d KiB)", out_path.name, out_path.stat().st_size // 1024)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("Bölüm %d atlandı: %s", ch.index, exc)
            return False


async def convert_all(selected: list[Chapter], total: int, args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(3, len(str(total)))
    sem = asyncio.Semaphore(max(1, args.concurrency))
    jobs = []
    for ch in selected:
        out = out_dir / build_filename(ch, width)
        if out.exists() and out.stat().st_size > 0:
            log.info("Zaten var, atlandı: %s", out.name)
            continue
        jobs.append(convert_chapter(ch, out, args, sem))
    results = await asyncio.gather(*jobs)
    failed = results.count(False)
    log.info("Bitti: %d üretildi, %d hata, %d zaten vardı",
             results.count(True), failed, len(selected) - len(jobs))
    return 1 if failed else 0


async def print_voices(locale: str) -> int:
    for v in await edge_tts.list_voices():
        if not locale or v["Locale"].lower().startswith(locale.lower()) \
                or v["ShortName"].lower().startswith(locale.lower()):
            print(f"{v['ShortName']}\t{v['Gender']}")
    return 0


# --------------------------------------------------------------------------
# Komut satırı
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EPUB -> MP3 (sesli kitap) dönüştürücü")
    p.add_argument("--input", help="EPUB yolu (glob olabilir; tek dosyaya çözülmeli)")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--rate", default="+0%", help="örn. +0%%, +10%%, -15%%")
    p.add_argument("--start", type=int, default=1, help="ilk bölüm (1'den başlar)")
    p.add_argument("--end", type=int, default=None, help="son bölüm (dahil)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--min-chars", type=int, default=100, help="bundan kısa bölümler atlanır")
    p.add_argument("--max-chars", type=int, default=3000, help="tek seferde seslendirilen en uzun parça")
    p.add_argument("--concurrency", type=int, default=3, help="aynı anda işlenen bölüm sayısı")
    p.add_argument("--count-chapters", action="store_true", help="yalnızca bölüm sayısını yazdır")
    p.add_argument("--list-voices", nargs="?", const="", metavar="LOCALE")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _normalize_argv(argv: list[str]) -> list[str]:
    """'--rate -15%' argparse'ta seçenek sanılır; '--rate=-15%' biçimine çevir."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--rate" and i + 1 < len(argv):
            out.append(f"--rate={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def resolve_input(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"Hata: '{pattern}' ile eşleşen dosya yok.")
    if len(matches) > 1:
        raise SystemExit(f"Hata: '{pattern}' birden fazla dosyaya çözüldü: {matches}")
    return Path(matches[0])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(sys.argv[1:] if argv is None else argv))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    if args.list_voices is not None:
        return asyncio.run(print_voices(args.list_voices))
    if not args.input:
        parser.error("--input gerekli")

    chapters = extract_chapters(resolve_input(args.input), args.min_chars)
    if args.count_chapters:
        print(len(chapters))  # stdout'a yalnızca sayı: workflow bunu okuyor
        return 0
    if not chapters:
        log.error("EPUB'da seslendirilecek bölüm bulunamadı.")
        return 1

    start = max(1, args.start)
    end = min(args.end or len(chapters), len(chapters))
    selected = [c for c in chapters if start <= c.index <= end]
    if not selected:
        log.error("Bölüm aralığı boş: %d-%d (toplam %d bölüm)", start, end, len(chapters))
        return 1
    log.info("%d bölüm seslendirilecek (%d-%d / toplam %d), ses: %s, hız: %s",
             len(selected), start, end, len(chapters), args.voice, args.rate)
    return asyncio.run(convert_all(selected, len(chapters), args))


if __name__ == "__main__":
    sys.exit(main())
