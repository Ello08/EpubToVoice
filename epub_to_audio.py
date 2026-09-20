#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPUB -> MP3 (sesli kitap) dönüştürücü.

Örnekler:
    python epub_to_audio.py --input kitap.epub
    python epub_to_audio.py --input "out/*.epub" --voice tr-TR-EmelNeural
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
                buf = f"{buf} {piece}" if buf else piece
        if buf:  # paragraf sonu: satır sonu ekle (sığıyorsa)
            if len(buf) + 1 <= max_chars:
                buf += "\n"
            else:
                flush()
    flush()
    return [c for c in chunks if re.search(r"\w", c)]


# --------------------------------------------------------------------------
# Sentezleme
# --------------------------------------------------------------------------
async def _stream_audio(text: str, args: argparse.Namespace) -> bytes:
    comm = edge_tts.Communicate(
        text, args.voice, rate=args.rate, volume=args.volume, pitch=args.pitch
    )
    buf = bytearray()
    async for msg in comm.stream():
        if msg["type"] == "audio":
            buf.extend(msg["data"])
    if not buf:
        raise RuntimeError("Servisten ses verisi alınamadı")
    return bytes(buf)


async def synth_chunk(text: str, args: argparse.Namespace, label: str) -> bytes:
    """Tek bir parçayı zaman aşımı + üstel geri çekilmeli yeniden deneme ile sentezler."""
    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            return await asyncio.wait_for(_stream_audio(text, args), timeout=args.chunk_timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < args.retries:
                delay = min(2 ** attempt, 20)
                log.warning(
                    "%s: deneme %d/%d başarısız (%s: %s); %d sn sonra tekrar denenecek",
                    label, attempt, args.retries, type(exc).__name__, exc, delay,
                )
                await asyncio.sleep(delay)
    raise RuntimeError(f"{label}: {args.retries} denemede sentezlenemedi") from last_exc


async def convert_chapter(
    ch: Chapter, out_path: Path, total: int,
    args: argparse.Namespace, sem: asyncio.Semaphore,
) -> bool:
    if out_path.exists() and not args.overwrite:
        log.info("[%d/%d] Zaten var, atlandı: %s", ch.index, total, out_path.name)
        return True

    async with sem:
        chunks = split_text(ch.text, args.max_chars)
        log.info(
            "[%d/%d] %s  (%d karakter, %d parça)",
            ch.index, total, out_path.name, len(ch.text), len(chunks),
        )
        tmp = out_path.with_name(out_path.name + ".part")
        try:
            with tmp.open("wb") as fh:
                for i, chunk in enumerate(chunks, 1):
                    fh.write(await synth_chunk(chunk, args, f"{out_path.name} parça {i}/{len(chunks)}"))
            tmp.replace(out_path)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("[%d/%d] BAŞARISIZ: %s -> %s", ch.index, total, out_path.name, exc)
            tmp.unlink(missing_ok=True)
            return False


async def process_book(epub_path: Path, out_dir: Path, args: argparse.Namespace) -> tuple[int, int]:
    log.info("Kitap: %s", epub_path)
    try:
        chapters = extract_chapters(epub_path, args.min_chars)
    except Exception as exc:  # noqa: BLE001
        log.error("EPUB okunamadı (%s): %s", epub_path, exc)
        return 0, 1

    if args.limit:
        chapters = chapters[: args.limit]
    if not chapters:
        log.error("Seslendirilecek bölüm bulunamadı: %s", epub_path)
        return 0, 1

    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(chapters))))
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(
        convert_chapter(ch, out_dir / build_filename(ch, width), len(chapters), args, sem)
        for ch in chapters
    ))
    ok = sum(results)
    return ok, len(results) - ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def resolve_inputs(value: str) -> list[Path]:
    p = Path(value)
    if p.is_dir():
        found = sorted(p.glob("*.epub"))
    elif any(c in value for c in "*?["):
        found = sorted(Path(m) for m in glob.glob(value, recursive=True))
    else:
        found = [p] if p.exists() else []
    return [f for f in found if f.is_file() and f.suffix.lower() == ".epub"]


async def list_voices(locale: str) -> None:
    for v in sorted(await edge_tts.list_voices(), key=lambda v: v["ShortName"]):
        if v["Locale"].lower().startswith(locale.lower()):
            print(f'{v["ShortName"]:<28} {v["Gender"]}')


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    p = argparse.ArgumentParser(description="EPUB dosyalarını edge-tts ile MP3 sesli kitaba dönüştürür.")
    p.add_argument("--input", "-i", help="EPUB dosyası, klasör veya glob deseni (örn. 'out/*.epub')")
    p.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Çıktı klasörü (varsayılan: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--voice", "-v", default=DEFAULT_VOICE, help=f"Edge TTS sesi (varsayılan: {DEFAULT_VOICE})")
    p.add_argument("--rate", default="+0%", help="Konuşma hızı, örn. +10%% veya -15%%")
    p.add_argument("--volume", default="+0%", help="Ses seviyesi, örn. +10%%")
    p.add_argument("--pitch", default="+0Hz", help="Ses perdesi, örn. +5Hz")
    p.add_argument("--max-chars", type=int, default=3000, help="Tek seferde sentezlenecek en fazla karakter")
    p.add_argument("--min-chars", type=int, default=200, help="Bundan kısa bölümleri atla (kapak, telif vb.)")
    p.add_argument("--retries", type=int, default=4, help="Parça başına yeniden deneme sayısı")
    p.add_argument("--chunk-timeout", type=float, default=120.0, help="Parça başına zaman aşımı (sn)")
    p.add_argument("--concurrency", type=int, default=2, help="Aynı anda işlenecek bölüm sayısı")
    p.add_argument("--limit", type=int, default=0, help="Sadece ilk N bölümü işle (test için)")
    p.add_argument("--overwrite", action="store_true", help="Var olan MP3 dosyalarının üzerine yaz")
    p.add_argument("--list-voices", nargs="?", const="tr-TR", metavar="LOCALE",
                   help="Kullanılabilir sesleri listele (varsayılan: tr-TR) ve çık")
    p.add_argument("--verbose", action="store_true", help="Ayrıntılı log")
    args = p.parse_args(argv)

    if args.list_voices is None:
        if not args.input:
            p.error("--input gerekli")
        if args.max_chars < 200:
            p.error("--max-chars en az 200 olmalı")
        args.concurrency = max(1, args.concurrency)
        args.retries = max(1, args.retries)
        try:  # rate/pitch/volume/voice biçimini baştan doğrula
            edge_tts.Communicate("test", args.voice, rate=args.rate, volume=args.volume, pitch=args.pitch)
        except (ValueError, TypeError) as exc:
            p.error(f"Geçersiz ses ayarı: {exc}")
    return args, p


def main(argv: list[str] | None = None) -> int:
    args, _ = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_voices is not None:
        asyncio.run(list_voices(args.list_voices))
        return 0

    books = resolve_inputs(args.input)
    if not books:
        log.error("EPUB dosyası bulunamadı: %s", args.input)
        return 1

    base = Path(args.output_dir)
    total_ok = total_fail = 0
    for book in books:
        # Birden fazla kitap varsa her biri kendi alt klasörüne yazılır
        out_dir = base if len(books) == 1 else base / (slugify(book.stem, 80) or "kitap")
        ok, fail = asyncio.run(process_book(book, out_dir, args))
        total_ok += ok
        total_fail += fail

    log.info("Bitti: %d bölüm başarılı, %d başarısız. Çıktı: %s", total_ok, total_fail, base.resolve())
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
