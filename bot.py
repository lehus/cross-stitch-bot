# bot.py
# Telegram bot + FastAPI webhook for Render
# Start command on Render:
#   uvicorn bot:app --host 0.0.0.0 --port $PORT
#
# Env vars:
#TELEGRAM_BOT_TOKEN=7750970184:AAGJyUd5fAweywY23rpdShcTyX4QGI4g0ek
# Optional:
#DMC_CSV_PATH=dmc_colors_100.csv

import os
import re
import csv
import zipfile
import tempfile
import logging
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cross-stitch-bot")

# ---------- Bot settings ----------
SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{};:,.<>?/\\|")

DEFAULTS = {
    "width": 60,          # stitches
    "colors": 20,         # max colors (KMeans clusters)
    "preview_scale": 10,  # preview upscaling
}

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

DMC_CSV_PATH = os.getenv("DMC_CSV_PATH", "dmc_colors_100.csv").strip()

# ---------- Color math: sRGB -> LAB and DeltaE ----------
def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_xyz(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = np.moveaxis(_srgb_to_linear(rgb), -1, 0)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return np.stack([x, y, z], axis=-1)

def xyz_to_lab(xyz):
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz = xyz / ref

    eps = 216 / 24389
    k = 24389 / 27

    def f(t):
        return np.where(t > eps, np.cbrt(t), (k * t + 16) / 116)

    fx, fy, fz = np.moveaxis(f(xyz), -1, 0)
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))

def delta_e76(lab1, lab2):
    lab1 = np.asarray(lab1, dtype=np.float32)
    lab2 = np.asarray(lab2, dtype=np.float32)
    diff = lab1[:, None, :] - lab2[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

# ---------- IO: load DMC palette ----------
def load_dmc_csv(path: str):
    floss, rgb, names = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            floss.append(str(row["floss"]).strip())
            rgb.append([int(row["R"]), int(row["G"]), int(row["B"])])
            names.append(row.get("name", "").strip())

    rgb = np.array(rgb, dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    return floss, rgb, lab, names

# ---------- Core steps ----------
def resize_to_stitches(img: Image.Image, width_st: int = None, height_st: int = None) -> Image.Image:
    w, h = img.size
    if width_st is None and height_st is None:
        raise ValueError("Set width_st or height_st (or both).")
    if width_st is None:
        width_st = round(w * (height_st / h))
    if height_st is None:
        height_st = round(h * (width_st / w))
    return img.convert("RGB").resize((width_st, height_st), Image.Resampling.LANCZOS)

def kmeans_centers(img_small: Image.Image, n_colors: int = 20, seed: int = 42):
    arr = np.array(img_small, dtype=np.uint8)
    pixels = arr.reshape(-1, 3).astype(np.float32)

    km = KMeans(n_clusters=n_colors, n_init="auto", random_state=seed)
    km.fit(pixels)

    centers = np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)
    labels = km.predict(pixels).reshape(arr.shape[0], arr.shape[1])
    return labels, centers

def map_centers_to_dmc(centers_rgb, dmc_lab):
    centers_lab = rgb_to_lab(centers_rgb)
    dist = delta_e76(centers_lab.astype(np.float32), dmc_lab.astype(np.float32))
    nearest = np.argmin(dist, axis=1)
    return nearest

def build_pattern(labels_2d, center_to_dmc_idx, dmc_rgb):
    dmc_idx_2d = center_to_dmc_idx[labels_2d]
    out_rgb = dmc_rgb[dmc_idx_2d]
    return dmc_idx_2d, Image.fromarray(out_rgb.astype(np.uint8), mode="RGB")

def make_preview(img_1px_per_stitch: Image.Image, scale: int = 10):
    w, h = img_1px_per_stitch.size
    return img_1px_per_stitch.resize((w * scale, h * scale), Image.Resampling.NEAREST)

def make_symbol_chart(
    dmc_idx_2d,
    floss_codes,
    dmc_rgb,
    names,
    out_png,
    out_csv,
    max_colors=20,
    block=18,
    grid_step=10,
):
    h, w = dmc_idx_2d.shape

    used = np.unique(dmc_idx_2d).tolist()
    if len(used) > max_colors:
        raise ValueError(f"Used colors {len(used)} > {max_colors}. Reduce /set colors or expand palette.")

    if len(used) > len(SYMBOLS):
        raise ValueError(f"Not enough symbols for {len(used)} colors. Reduce colors or extend SYMBOLS.")

    used_sorted = sorted(used, key=lambda i: floss_codes[i])
    symbol_map = {dmc_i: SYMBOLS[k] for k, dmc_i in enumerate(used_sorted)}

    img_w, img_h = w * block, h * block
    chart = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(chart)

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=int(block * 0.6))
    except:
        font = ImageFont.load_default()

    for y in range(h):
        for x in range(w):
            dmc_i = int(dmc_idx_2d[y, x])
            rgb = tuple(int(v) for v in dmc_rgb[dmc_i])
            x0, y0 = x * block, y * block
            x1, y1 = x0 + block, y0 + block
            draw.rectangle([x0, y0, x1, y1], fill=rgb)

            r, g, b = rgb
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sym_color = (0, 0, 0) if lum > 140 else (255, 255, 255)
            sym = symbol_map[dmc_i]

            bb = draw.textbbox((0, 0), sym, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((x0 + (block - tw) / 2, y0 + (block - th) / 2 - 1), sym, fill=sym_color, font=font)

    # grid
    for gx in range(0, w + 1):
        xpix = gx * block
        lw = 3 if gx % grid_step == 0 else 1
        draw.line([(xpix, 0), (xpix, img_h)], fill=(50, 50, 50), width=lw)

    for gy in range(0, h + 1):
        ypix = gy * block
        lw = 3 if gy % grid_step == 0 else 1
        draw.line([(0, ypix), (img_w, ypix)], fill=(50, 50, 50), width=lw)

    chart.save(out_png)

    cnt = Counter(dmc_idx_2d.flatten().tolist())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "DMC", "R", "G", "B", "stitches", "name"])
        for dmc_i in used_sorted:
            r, g, b = (int(v) for v in dmc_rgb[dmc_i])
            writer.writerow([symbol_map[dmc_i], floss_codes[dmc_i], r, g, b, cnt[dmc_i], names[dmc_i]])

def image_to_cross_stitch_dmc(
    input_path: str,
    out_prefix: str,
    width_st: int,
    height_st: int | None,
    max_colors: int,
    preview_scale: int,
    dmc_csv_path: str,
    seed: int = 42,
):
    floss, dmc_rgb, dmc_lab, names = load_dmc_csv(dmc_csv_path)

    max_colors = max(2, int(max_colors))
    max_colors = min(max_colors, len(floss))  # cannot exceed palette size

    img = Image.open(input_path)
    img_small = resize_to_stitches(img, width_st=width_st, height_st=height_st)

    labels_2d, centers_rgb = kmeans_centers(img_small, n_colors=max_colors, seed=seed)
    center_to_dmc = map_centers_to_dmc(centers_rgb, dmc_lab)

    dmc_idx_2d, stitch_map = build_pattern(labels_2d, center_to_dmc, dmc_rgb)

    stitch_map.save(f"{out_prefix}_stitches.png")
    make_preview(stitch_map, scale=preview_scale).save(f"{out_prefix}_preview.png")

    make_symbol_chart(
        dmc_idx_2d,
        floss,
        dmc_rgb,
        names,
        out_png=f"{out_prefix}_chart.png",
        out_csv=f"{out_prefix}_legend.csv",
        max_colors=max_colors,
        block=18,
        grid_step=10,
    )

    used_colors = len(np.unique(dmc_idx_2d))
    return used_colors

# ---------- Telegram handlers ----------
def _get_settings(context: ContextTypes.DEFAULT_TYPE):
    s = context.user_data.get("settings")
    if not s:
        s = DEFAULTS.copy()
        context.user_data["settings"] = s
    return s

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = _get_settings(context)
    await update.message.reply_text(
        "Привет! Пришли картинку — я сделаю схему для вышивки крестиком (ZIP).\n\n"
        "Команды:\n"
        "/settings — показать настройки\n"
        "/set width 80 — ширина в крестиках (10..400)\n"
        "/set colors 20 — максимум цветов (2..120, но не больше палитры)\n"
        "/set preview_scale 10 — масштаб превью (2..30)\n\n"
        f"Текущие:\n• width={s['width']}\n• colors={s['colors']}\n• preview_scale={s['preview_scale']}\n"
        f"Палитра: {DMC_CSV_PATH}"
    )

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = _get_settings(context)
    await update.message.reply_text(
        "Текущие настройки:\n"
        f"• width: {s['width']}\n"
        f"• colors: {s['colors']}\n"
        f"• preview_scale: {s['preview_scale']}\n"
        f"• palette: {DMC_CSV_PATH}"
    )

async def set_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = _get_settings(context)
    text = (update.message.text or "").strip()

    m = re.match(r"^/set\s+(width|colors|preview_scale)\s+(\d+)$", text, flags=re.I)
    if not m:
        await update.message.reply_text("Формат: /set width 80 | /set colors 20 | /set preview_scale 10")
        return

    key = m.group(1).lower()
    val = int(m.group(2))

    if key == "width":
        val = max(10, min(400, val))
    elif key == "colors":
        val = max(2, min(120, val))
    elif key == "preview_scale":
        val = max(2, min(30, val))

    s[key] = val
    await update.message.reply_text(f"Ок: {key} = {val}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = _get_settings(context)
    msg = update.message

    tg_file = None
    file_name = "image.jpg"

    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
        file_name = "photo.jpg"
    elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
        tg_file = await msg.document.get_file()
        file_name = msg.document.file_name or "image"
    else:
        await msg.reply_text("Пришли фото или файл-картинку (JPG/PNG).")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / file_name
        await tg_file.download_to_drive(custom_path=str(in_path))

        # validate
        try:
            Image.open(in_path).verify()
        except Exception:
            await msg.reply_text("Похоже, это не картинка. Пришли JPG/PNG.")
            return

        out_prefix = str(td / "pattern")

        palette_path = DMC_CSV_PATH
        if not Path(palette_path).is_absolute():
            palette_path = str((Path.cwd() / palette_path).resolve())

        if not Path(palette_path).exists():
            await msg.reply_text(
                f"Не найдена палитра CSV: {DMC_CSV_PATH}\n"
                "Положи файл рядом с bot.py или установи DMC_CSV_PATH."
            )
            return

        try:
            used = image_to_cross_stitch_dmc(
                input_path=str(in_path),
                out_prefix=out_prefix,
                width_st=int(s["width"]),
                height_st=None,
                max_colors=int(s["colors"]),
                preview_scale=int(s["preview_scale"]),
                dmc_csv_path=palette_path,
            )
        except Exception as e:
            await msg.reply_text(f"Ошибка обработки: {e}")
            return

        zip_path = td / "pattern.zip"
        outputs = [
            f"{out_prefix}_stitches.png",
            f"{out_prefix}_preview.png",
            f"{out_prefix}_chart.png",
            f"{out_prefix}_legend.csv",
        ]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fp in outputs:
                p = Path(fp)
                if p.exists():
                    z.write(p, arcname=p.name)

        caption = (
            f"Готово ✅\n"
            f"Ширина: {s['width']} крестиков\n"
            f"Цветов (лимит): {s['colors']} | использовано: {used}\n"
            f"Палитра: {Path(palette_path).name}"
        )

        await msg.reply_document(document=zip_path.open("rb"), filename="pattern.zip", caption=caption)

# ---------- FastAPI + webhook glue for Render ----------
app = FastAPI()

telegram_app = Application.builder().token(TOKEN).build()
telegram_app.add_handler(CommandHandler("start", start_cmd))
telegram_app.add_handler(CommandHandler("settings", settings_cmd))
telegram_app.add_handler(CommandHandler("set", set_cmd))
telegram_app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))

@app.on_event("startup")
async def _startup():
    # init telegram Application (required when used without run_polling)
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Telegram application started")

@app.on_event("shutdown")
async def _shutdown():
    await telegram_app.stop()
    await telegram_app.shutdown()
    logger.info("Telegram application stopped")

@app.get("/")
async def root():
    return {"ok": True, "service": "cross-stitch-bot"}

@app.post("/bot/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}

