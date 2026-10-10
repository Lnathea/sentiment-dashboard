import pytest

from ml.preprocess import preprocess, preprocess_many


def test_lowercase() -> None:
    assert preprocess("Makanan ENAK Sekali") == "makanan enak sekali"


@pytest.mark.parametrize(
    "raw",
    [
        "cek https://example.com/a?b=1 sekarang",
        "cek http://example.com sekarang",
        "cek www.example.com sekarang",
        "cek HTTPS://EXAMPLE.COM sekarang",
    ],
)
def test_removes_urls(raw: str) -> None:
    assert preprocess(raw) == "cek sekarang"


def test_removes_mentions() -> None:
    assert preprocess("@budi_99 terima kasih @Ani") == "terima kasih"


def test_email_is_not_treated_as_mention() -> None:
    assert preprocess("kirim ke a@b.com") == "kirim ke a@b.com"


def test_hashtag_symbol_removed_word_kept() -> None:
    assert preprocess("liburan #SeruBanget #2024") == "liburan serubanget 2024"


def test_lone_hash_kept() -> None:
    assert preprocess("nomor # satu") == "nomor # satu"


def test_repeated_characters_shortened_to_two() -> None:
    assert preprocess("bagusssss bangeeeet!!!!") == "baguss bangeet!!"


def test_double_letters_preserved() -> None:
    assert preprocess("maaf saat ini") == "maaf saat ini"


def test_whitespace_collapsed() -> None:
    assert preprocess("  halo \n\t dunia   ") == "halo dunia"


def test_unicode_normalized() -> None:
    # full-width letters -> ASCII via NFKC
    assert preprocess("ＢＡＧＵＳ") == "bagus"


def test_empty_and_only_noise() -> None:
    assert preprocess("") == ""
    assert preprocess("   @user https://x.y  ") == ""


def test_idempotent() -> None:
    raw = "Mantappp!!! @admin #Promo https://t.co/x   ok"
    once = preprocess(raw)
    assert preprocess(once) == once


def test_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        preprocess(None)  # type: ignore[arg-type]


def test_preprocess_many() -> None:
    assert preprocess_many(["A", "B  C"]) == ["a", "b c"]
