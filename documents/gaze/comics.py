"""Builds `data/comics/` for the gaze-head documents from `baulab/openai-comic-strips` (500 six-panel strips):
one stitched strip per comic, as the paper assembles it (panels resized to a common height of 256 pixels, each
dimension rounded to a multiple of 32, a 6-pixel white gap), a lettered copy with a random letter A-Z on each
panel (drawn without replacement, seeded by the comic id), and the rows that reference them:

- `comics.json`: one row per strip, with each panel's box in fractions of the strip (`panels`) and its letter;
- `queries.json`: six rows per strip, `chat` asking what happens in the k-th panel of `strip` (the paper's
  discovery prompt), `panel` the queried panel's box, `split` `k0` to `k5`;
- `letter_queries.json`: six rows per strip over the lettered copy, `chat` asking for the letter on the k-th
  panel and `ambiguous` for the letter on "this panel", `answer` the queried panel's letter, `target` the box
  and `target_answer` the letter of another panel (a seeded draw), the panel a redirection aims at.

Run from the repository root: `uv run python documents/gaze/comics.py`. It needs the dataset in the Hugging Face
cache (or the hub), Pillow and `datasets`; about 360 MB of images, which `.gitignore` leaves out.
"""
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth"]
QUESTION = ("Look carefully at this six-panel comic strip. What is happening in the {ordinal} panel from the "
            "left? Answer briefly.")
LETTER_QUESTION = ("Look carefully at this six-panel comic strip. Each panel has a letter written on it. What is "
                   "the letter on the {ordinal} panel from the left? Answer with the letter only.")
AMBIGUOUS = ("Look carefully at this comic strip. Each panel has a letter written on it. What is the letter on "
             "this panel? Answer with the letter only.")


def round32(x: float) -> int:
    return max(32, 32 * round(float(x) / 32.0))


def strip_of(panels: list, height: int = 256, gap: int = 6) -> tuple:
    """The panels resized to one height and pasted left to right, with a white gap at each boundary."""
    resized, widths = [], []
    for p in panels:
        w, h = p.size
        nw = round32(w * height / max(h, 1))
        resized.append(p.resize((nw, height), Image.LANCZOS))
        widths.append(nw)
    out = Image.new("RGB", (sum(widths), height), (255, 255, 255))
    x = 0
    for img in resized:
        out.paste(img, (x, 0))
        x += img.size[0]
    draw = ImageDraw.Draw(out)
    x = 0
    for w in widths[:-1]:
        x += w
        draw.rectangle([x - gap // 2, 0, x + (gap - gap // 2) - 1, height - 1], fill="white")
    return out, widths


def lettered(strip: Image.Image, widths: list, letters: list) -> Image.Image:
    """The strip with a letter in a white box at each panel's top left."""
    out = strip.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=64)
    x = 0
    for w, letter in zip(widths, letters):
        draw.rectangle([x + 8, 8, x + 72, 80], fill="white", outline="black", width=3)
        draw.text((x + 40, 44), letter, fill="black", font=font, anchor="mm")
        x += w
    return out


def chat(image: str, text: str) -> list:
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]


def main() -> None:
    import datasets

    out = os.path.join(ROOT, "data", "comics")
    for sub in ("strips", "lettered"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    rows = datasets.load_dataset("baulab/openai-comic-strips", split="train")
    comics, queries, letter_queries = [], [], []
    for row in rows:
        cid = int(row["comic_id"])
        strip, widths = strip_of([row[f"panel_{k}"].convert("RGB") for k in range(1, 7)])
        letters = random.Random(cid).sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 6)
        name = f"comic{cid:03d}.png"
        strip.save(os.path.join(out, "strips", name))
        lettered(strip, widths, letters).save(os.path.join(out, "lettered", name))
        total, boxes, x = sum(widths), [], 0
        for w in widths:
            boxes.append([round(x / total, 6), 0.0, round((x + w) / total, 6), 1.0])
            x += w
        comic = {"comic_id": cid, "strip": f"comics/strips/{name}", "lettered": f"comics/lettered/{name}",
                 "widths": widths, "panels": boxes, "letters": letters,
                 "captions": [row[f"caption_{k}"] for k in range(1, 7)]}
        comics.append(comic)
        for k in range(6):
            queries.append({**comic, "k": k, "split": f"k{k}", "panel": boxes[k],
                            "chat": chat(comic["strip"], QUESTION.format(ordinal=ORDINAL[k]))})
            j = random.Random(1000 * cid + k).choice([x for x in range(6) if x != k])
            letter_queries.append({**comic, "k": k, "split": f"k{k}", "panel": boxes[k], "answer": letters[k],
                                   "target_k": j, "target": boxes[j], "target_answer": letters[j],
                                   "chat": chat(comic["lettered"], LETTER_QUESTION.format(ordinal=ORDINAL[k])),
                                   "ambiguous": chat(comic["lettered"], AMBIGUOUS)})
    for name, found in (("comics", comics), ("queries", queries), ("letter_queries", letter_queries)):
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(found, f, indent=1)
    print(f"wrote {len(comics)} strips and {len(queries)} queries under {out}")


if __name__ == "__main__":
    main()
