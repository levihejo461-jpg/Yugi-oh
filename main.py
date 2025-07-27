import random
import requests
from datetime import datetime
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    InlineQueryResultCachedPhoto,
    InlineQueryResultPhoto
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    InlineQueryHandler,
    ChosenInlineResultHandler,
    ContextTypes,
    filters
)
import pytz
import sqlite3
from pathlib import Path
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io
from io import BytesIO
import hashlib
import re
import os

# ─── CONFIG ───
BOT_TOKEN = os.environ.get('BOT_TOKEN', "")
VALID_DECK_SIZES = {30, 40, 50}
DB_FILE = "yugioh_bot.db"

# ─── IMAGE URLS ───
WELCOME_IMG = "https://files.catbox.moe/2qqj64.jpg"
BATTLE_IMG = "https://files.catbox.moe/waaybl.jpg"
JOIN_IMAGES = [
    "https://files.catbox.moe/e0yu0z.jpg",
    "https://files.catbox.moe/sdyyds.jpg"
]
DRAW_READY_GIFS = [
    "https://files.catbox.moe/5xmojc.mp4",
    "https://files.catbox.moe/ahma73.mp4",
    "https://files.catbox.moe/o0dmkm.mp4",
    "https://files.catbox.moe/8etssr.mp4"
]
BATTLE_BEGIN_IMGS = [
    "https://files.catbox.moe/f52el9.jpg",
    "https://files.catbox.moe/sdyyds.jpg"
]

# ─── DATABASE SETUP ───
def init_db():
    Path(DB_FILE).touch()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_cards (
            user_id TEXT,
            card_type TEXT,
            file_id TEXT,
            atk INTEGER,
            defense INTEGER,
            level INTEGER,
            PRIMARY KEY (user_id, file_id)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_decks (
            user_id TEXT,
            file_id TEXT,
            position INTEGER,
            PRIMARY KEY (user_id, file_id)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS drawn_cards (
            user_id TEXT,
            file_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS card_hashes (
            user_id TEXT,
            card_hash TEXT,
            file_id TEXT,
            PRIMARY KEY (user_id, card_hash)
        )""")

init_db()

# ─── IMAGE PROCESSING FUNCTIONS ───
async def create_battle_begin_image():
    try:
        right_response = requests.get(BATTLE_BEGIN_IMGS[0])
        left_response = requests.get(BATTLE_BEGIN_IMGS[1])
        right_img = Image.open(BytesIO(right_response.content))
        left_img = Image.open(BytesIO(left_response.content))

        target_height = min(right_img.height, left_img.height)
        right_width = int((right_img.width * target_height) / right_img.height)
        left_width = int((left_img.width * target_height) / left_img.height)

        right_img = right_img.resize((right_width, target_height), Image.Resampling.LANCZOS)
        left_img = left_img.resize((left_width, target_height), Image.Resampling.LANCZOS)

        total_width = left_width + right_width
        combined = Image.new('RGB', (total_width, target_height))
        combined.paste(left_img, (0, 0))
        combined.paste(right_img, (left_width, 0))

        output = BytesIO()
        combined.save(output, format='JPEG', quality=95)
        output.seek(0)
        return output
    except Exception as e:
        print(f"Error creating combined battle image: {e}")
        try:
            response = requests.get(BATTLE_BEGIN_IMGS[0])
            return BytesIO(response.content)
        except:
            return None

async def rotate_image(file_url, angle):
    try:
        response = requests.get(file_url)
        img = Image.open(BytesIO(response.content))
        rotated = img.rotate(angle, expand=True)
        out = BytesIO()
        rotated.save(out, format="JPEG")
        out.seek(0)
        return out
    except Exception as e:
        print(f"Rotation error: {e}")
        return None

def is_monster_card(text):
    if not text:
        return False
    keywords = ["monster", "dragon", "zombie", "beast", "cyber", "robot"]
    return any(kw.lower() in text.lower() for kw in keywords)

def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def is_duplicate_card(user_id, image_bytes):
    card_hash = get_image_hash(image_bytes)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT file_id FROM card_hashes WHERE user_id='public' AND card_hash=?",
            (card_hash,)
        )
        return cursor.fetchone() is not None

def save_card_hash(user_id, file_id, image_bytes):
    card_hash = get_image_hash(image_bytes)
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO card_hashes VALUES (?, ?, ?)",
            ('public', card_hash, file_id)
        )

def analyze_card_image(file_id):
    try:
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_id}"
        response = requests.get(file_url)
        image_bytes = response.content
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        full_text = ""
        text_type = None
        try:
            full_text = pytesseract.image_to_string(img, config='--psm 6')
            if full_text.strip():
                text_type = detect_card_type_by_text(full_text)
        except Exception as ocr_error:
            print(f"OCR not available: {ocr_error}")

        color_type = detect_card_type_by_color(img_cv)
        atk, defense = extract_atk_def_from_text(full_text) if full_text else (None, None)
        level = extract_level_from_text(full_text) if full_text else None

        if atk is not None and defense is not None:
            card_type = 'monster'
            confidence = 95
        else:
            detection_methods = []
            if text_type:
                detection_methods.append(text_type)
            if color_type:
                detection_methods.append(color_type)
            
            if not detection_methods:
                card_type = None
                confidence = 0
            elif len(detection_methods) == 1:
                card_type = detection_methods[0]
                confidence = 70
            elif text_type == color_type:
                card_type = text_type
                confidence = 90
            else:
                if text_type in ['spell', 'trap']:
                    card_type = text_type
                    confidence = 60
                elif color_type:
                    card_type = color_type
                    confidence = 50
                else:
                    card_type = text_type if text_type else color_type
                    confidence = 40

        if card_type:
            confidence = calculate_confidence_score(card_type, full_text, atk, defense)
            if text_type and color_type and text_type == color_type:
                confidence = min(confidence + 20, 100)

        if confidence < 40:
            card_type = None

        return {
            "type": card_type,
            "ATK": atk,
            "DEF": defense,
            "level": level,
            "confidence": confidence,
            "image_bytes": image_bytes
        }
    except Exception as e:
        print(f"Card Analysis Error: {e}")
        return {
            "type": None,
            "ATK": None,
            "DEF": None,
            "level": None,
            "confidence": 0,
            "image_bytes": None
        }

def detect_card_type_by_color(img_cv):
    try:
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        height, width = img_cv.shape[:2]

        top_border = hsv[int(height*0.01):int(height*0.20), int(width*0.02):int(width*0.98)]
        bottom_border = hsv[int(height*0.80):int(height*0.99), int(width*0.02):int(width*0.98)]
        left_border = hsv[int(height*0.10):int(height*0.90), 0:int(width*0.12)]
        right_border = hsv[int(height*0.10):int(height*0.90), int(width*0.88):width]
        center_top = hsv[int(height*0.05):int(height*0.25), int(width*0.20):int(width*0.80)]

        all_regions = [
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3),
            center_top.reshape(-1, 3)
        ]
        border_pixels = np.vstack(all_regions)

        hist_h = cv2.calcHist([border_pixels], [0], None, [180], [0, 180])
        monster_range = np.sum(hist_h[5:45])
        spell_range = np.sum(hist_h[50:80])
        trap_range = np.sum(hist_h[120:160])

        rgb_regions = np.vstack([
            rgb[int(height*0.01):int(height*0.20), int(width*0.02):int(width*0.98)].reshape(-1, 3),
            rgb[int(height*0.80):int(height*0.99), int(width*0.02):int(width*0.98)].reshape(-1, 3)
        ])
        avg_rgb = np.mean(rgb_regions, axis=0)
        r, g, b = avg_rgb

        rgb_monster_score = 0
        rgb_spell_score = 0
        rgb_trap_score = 0

        if r > g and r > b and (r - g) > 20:
            rgb_monster_score += 50
        if g > r and g > b and (g - r) > 15:
            rgb_spell_score += 50
        if (r > g and b > g) or (b > r and b > g):
            rgb_trap_score += 50

        monster_total = monster_range + rgb_monster_score
        spell_total = spell_range + rgb_spell_score
        trap_total = trap_range + rgb_trap_score
        total_pixels = np.sum(hist_h)
        min_threshold = total_pixels * 0.03

        scores = {'monster': monster_total, 'spell': spell_total, 'trap': trap_total}
        max_type = max(scores, key=scores.get)
        max_value = scores[max_type]

        if max_value < min_threshold:
            return None

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.2:
            return max_type
        elif sorted_scores[0] > min_threshold * 2:
            return max_type
        return None
    except Exception as e:
        print(f"Color detection error: {e}")
        return None

def detect_card_type_by_text(full_text):
    if not full_text or len(full_text.strip()) < 3:
        return None
    full_lower = ' '.join(full_text.lower().split())

    spell_patterns = [
        '[spell card]', 'spell card', '[magic card]', 'magic card',
        'spell/magic', 'spell-card', 'spellcard', '[spell]',
        'type: spell', 'type:spell', '(spell card)', 'spell-type',
        'magic/spell', 'card type: spell'
    ]
    trap_patterns = [
        '[trap card]', 'trap card', 'trap-card', 'trapcard',
        '[trap]', 'type: trap', 'type:trap', '(trap card)',
        'trap-type', 'card type: trap'
    ]
    monster_patterns = [
        '[monster card]', 'monster card', 'monster-card',
        'monstercard', '[monster]', 'type: monster',
        'type:monster', '(monster card)'
    ]

    for pattern in spell_patterns:
        if pattern in full_lower:
            return 'spell'
    for pattern in trap_patterns:
        if pattern in full_lower:
            return 'trap'
    for pattern in monster_patterns:
        if pattern in full_lower:
            return 'monster'

    atk_def_patterns = [
        r'atk[/\s]*\d+[/\s]*def[/\s]*\d+',
        r'atk:\s*\d+\s*def:\s*\d+',
        r'atk\s*\d+\s*def\s*\d+',
        r'\d+\s*atk\s*/\s*\d+\s*def',
        r'attack[/\s]*\d+[/\s]*defense[/\s]*\d+'
    ]
    for pattern in atk_def_patterns:
        if re.search(pattern, full_lower):
            return 'monster'

    spell_score = 0
    trap_score = 0
    monster_score = 0

    spell_keywords = [
        'spell', 'magic', 'equip', 'field', 'ritual',
        'quick-play spell', 'continuous spell', 'normal spell',
        'special summon', 'activate', 'draw card',
        'search your deck', 'add to your hand', 'banish',
        'send to graveyard', 'pay life points', 'increase atk',
        'destroy all monsters', 'target 1 monster', 'fusion summon'
    ]
    trap_keywords = [
        'trap', 'counter trap', 'continuous trap', 'normal trap',
        'when your opponent', 'negate the activation',
        'destroy that card', 'cannot be special summoned',
        'your opponent cannot', 'during your opponent',
        'when this card is activated', 'pay 1000 life points',
        'mirror force', 'torrential tribute'
    ]
    monster_keywords = [
        'warrior', 'dragon', 'spellcaster', 'fiend', 'beast',
        'machine', 'aqua', 'pyro', 'rock', 'winged beast',
        'plant', 'insect', 'thunder', 'zombie', 'reptile',
        'psychic', 'divine-beast', 'level', 'rank', 'link-',
        'xyz', 'synchro', 'fusion', 'pendulum', 'tuner',
        'flip', 'gemini', 'spirit', 'toon', 'union', 'effect',
        'dark', 'light', 'earth', 'water', 'fire', 'wind'
    ]

    for keyword in spell_keywords:
        if keyword in full_lower:
            if keyword in ['spell', 'magic']:
                spell_score += 5
            elif keyword in ['equip', 'field', 'ritual', 'quick-play spell', 'continuous spell']:
                spell_score += 4
            else:
                spell_score += 2

    for keyword in trap_keywords:
        if keyword in full_lower:
            if keyword in ['trap']:
                trap_score += 5
            elif keyword in ['counter trap', 'continuous trap', 'normal trap']:
                trap_score += 4
            else:
                trap_score += 2

    for keyword in monster_keywords:
        if keyword in full_lower:
            if keyword in ['level', 'rank', 'xyz', 'synchro', 'fusion']:
                monster_score += 4
            elif keyword in ['dark', 'light', 'earth', 'water', 'fire', 'wind']:
                monster_score += 3
            else:
                monster_score += 2

    max_score = max(spell_score, trap_score, monster_score)
    if max_score < 4:
        return None

    if spell_score == max_score and spell_score > max(trap_score, monster_score) * 1.3:
        return 'spell'
    elif trap_score == max_score and trap_score > max(spell_score, monster_score) * 1.3:
        return 'trap'
    elif monster_score == max_score and monster_score > max(spell_score, trap_score) * 1.3:
        return 'monster'
    return None

def extract_atk_def_from_text(full_text):
    if not full_text:
        return None, None
    atk, defense = None, None
    patterns = [
        r'atk[/\s]*(\d+)[/\s]*def[/\s]*(\d+)',
        r'attack[/\s]*(\d+)[/\s]*defense[/\s]*(\d+)',
        r'(\d+)[/\s]*atk[/\s]*(\d+)[/\s]*def',
        r'(\d+)[/\s]*/[/\s]*(\d+)',
        r'atk\s*(\d+)\s*def\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, full_text)
        if match:
            try:
                atk = int(match.group(1))
                defense = int(match.group(2))
                break
            except (ValueError, IndexError):
                continue
    return atk, defense

def extract_level_from_text(full_text):
    if not full_text:
        return None
    level_patterns = [
        r'level\s*(\d+)',
        r'rank\s*(\d+)',
        r'link-(\d+)',
        r'lv\s*(\d+)',
    ]
    for pattern in level_patterns:
        match = re.search(pattern, full_text)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    return None

def extract_attribute(full_text, type_text):
    attributes = ['dark', 'light', 'earth', 'water', 'fire', 'wind', 'divine']
    text_to_search = (full_text + " " + type_text).lower()
    for attr in attributes:
        if attr in text_to_search:
            return attr.upper()
    return None

def detect_special_type(full_text, type_text):
    special_types = ['xyz', 'synchro', 'fusion', 'link', 'pendulum', 'ritual', 'effect', 'normal']
    text_to_search = (full_text + " " + type_text).lower()
    detected_types = []
    for special in special_types:
        if special in text_to_search:
            detected_types.append(special.capitalize())
    return detected_types if detected_types else None

def calculate_confidence_score(card_type, full_text, atk, defense):
    confidence = 0
    if card_type:
        confidence += 50
    if full_text and len(full_text.strip()) > 10:
        confidence += 30
    if card_type == 'monster':
        if atk is not None:
            confidence += 10
        if defense is not None:
            confidence += 10
    return min(confidence, 100)

# ─── PAGINATION FUNCTIONS ───
def get_user_cards_paginated(user_id, card_type=None, page=1, per_page=50):
    offset = (page - 1) * per_page
    with sqlite3.connect(DB_FILE) as conn:
        if card_type:
            cursor = conn.execute(
                "SELECT file_id FROM user_cards WHERE user_id='public' AND card_type=? LIMIT ? OFFSET ?",
                (card_type, per_page, offset)
            )
            count_cursor = conn.execute(
                "SELECT COUNT(*) FROM user_cards WHERE user_id='public' AND card_type=?",
                (card_type,)
            )
        else:
            cursor = conn.execute(
                "SELECT file_id FROM user_cards WHERE user_id='public' LIMIT ? OFFSET ?",
                (per_page, offset)
            )
            count_cursor = conn.execute(
                "SELECT COUNT(*) FROM user_cards WHERE user_id='public'"
            )
        cards = [row[0] for row in cursor.fetchall()]
        total_count = count_cursor.fetchone()[0]
        total_pages = (total_count + per_page - 1) // per_page
    return cards, total_pages, total_count

def parse_page_number(query_parts):
    try:
        for part in query_parts:
            if part.isdigit():
                return int(part)
        return 1
    except:
        return 1

# ─── DATABASE FUNCTIONS ───
def validate_card_group(detected_type, chosen_group):
    if not detected_type:
        return False, "❌ Could not detect card type automatically. Card rejected for safety."
    if detected_type.lower() == chosen_group.lower():
        return True, f"✅ Correct! This is a {detected_type.capitalize()} card."
    else:
        return False, f"❌ This appears to be a {detected_type.capitalize()} card, but you're trying to save it as {chosen_group.capitalize()}. Please choose the correct group."

def save_card_to_db(user_id, file_id, card_type, atk=None, defense=None, level=None, attribute=None, special_type=None):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_cards VALUES (?, ?, ?, ?, ?, ?)",
            ('public', card_type, file_id, atk, defense, level)
        )

def get_user_cards(user_id, card_type=None):
    with sqlite3.connect(DB_FILE) as conn:
        if card_type:
            cursor = conn.execute(
                "SELECT file_id FROM user_cards WHERE user_id='public' AND card_type=?",
                (card_type,)
            )
        else:
            cursor = conn.execute(
                "SELECT file_id FROM user_cards WHERE user_id='public'"
            )
        return [row[0] for row in cursor.fetchall()]

def is_card_in_collection(user_id, file_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM user_cards WHERE user_id='public' AND file_id=?",
            (file_id,)
        )
        return cursor.fetchone() is not None

def save_deck_to_db(user_id, file_ids):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM user_decks WHERE user_id=?", (user_id,))
        for idx, file_id in enumerate(file_ids):
            conn.execute(
                "INSERT INTO user_decks VALUES (?, ?, ?)",
                (user_id, file_id, idx)
            )

def get_user_deck(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT file_id FROM user_decks WHERE user_id=? ORDER BY position",
            (user_id,)
        )
        return [row[0] for row in cursor.fetchall()]

def save_drawn_cards(user_id, file_ids):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM drawn_cards WHERE user_id=?", (user_id,))
        for file_id in file_ids:
            conn.execute(
                "INSERT INTO drawn_cards VALUES (?, ?, datetime('now'))",
                (user_id, file_id, datetime.now())
            )

def get_drawn_cards(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            "SELECT file_id FROM drawn_cards WHERE user_id=? ORDER BY timestamp DESC",
            (user_id,)
        )
        return [row[0] for row in cursor.fetchall()]

def remove_sent_card_from_drawn(user_id, file_id):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "DELETE FROM drawn_cards WHERE user_id=? AND file_id=? AND rowid IN (SELECT rowid FROM drawn_cards WHERE user_id=? AND file_id=? LIMIT 1)",
            (user_id, file_id, user_id, file_id)
        )

def remove_from_deck(user_id, file_ids):
    with sqlite3.connect(DB_FILE) as conn:
        for file_id in file_ids:
            conn.execute(
                "DELETE FROM user_decks WHERE user_id=? AND file_id=?",
                (user_id, file_id)
            )

def delete_card_from_collection(user_id, file_id):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "DELETE FROM user_cards WHERE user_id='public' AND file_id=?",
            (file_id,)
        )
        conn.execute(
            "DELETE FROM user_decks WHERE file_id=?",
            (file_id,)
        )
        conn.execute(
            "DELETE FROM drawn_cards WHERE file_id=?",
            (file_id,)
        )

def delete_card_from_deck_only(user_id, file_id):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "DELETE FROM user_decks WHERE user_id=? AND file_id=?",
            (user_id, file_id)
        )

def format_collection(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM user_cards WHERE user_id='public'")
        conn.execute("DELETE FROM user_decks")
        conn.execute("DELETE FROM drawn_cards")
        conn.execute("DELETE FROM card_hashes WHERE user_id='public'")

def format_deck(user_id):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM user_decks WHERE user_id=?", (user_id,))

# ─── HANDLERS ───
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (
        "🃏➾ <b>WELCOME TO YU GI OH GAME DUEL</b>\n\n"
        " ▶ <b>GAME COMMANDS</b>\n"
        "1. Browse PUBLIC card collection:\n"
        "◎ @Theexper_bot\n"
        " • @Theexper_bot collection [monster|spell|trap]\n"
        " All players share the same card collection!\n\n"
        "2. Build your personal 30/40/50-card deck:\n"
        " • Send exactly X photos from public collection → /mydeck\n"
        " ONLY IN DM.(Private CHAT with the bot).\n"
        " To see your deck type @Theexper_bot mydeck\n\n"
        "3. Draw cards during play:\n"
        "➾ it only work in duel\n"
        " • /draw N(number of cards)→ then send it then type on the keyboard\n"
        " ➤ @Theexper_bot deck\n\n"
        "4. Challenge a friend:\n"
        " • /playgame → Join Duel → guess 1–5 how guess the correct number Starr→ duel begins!\n\n"
        "5. Add cards to PUBLIC collection:\n"
        " • Send card photos → /save (password protected)"
    )
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=WELCOME_IMG,
        caption=caption,
        parse_mode="HTML"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    file_id = update.message.photo[-1].file_id
    message_id = update.message.message_id

    if 'temp_photos' not in context.user_data:
        context.user_data['temp_photos'] = []
    if 'temp_message_ids' not in context.user_data:
        context.user_data['temp_message_ids'] = {}

    context.user_data['temp_photos'].append(file_id)
    context.user_data['temp_message_ids'][file_id] = message_id

    if 'deck_temp_photos' not in context.user_data:
        context.user_data['deck_temp_photos'] = []
    context.user_data['deck_temp_photos'].append(file_id)

async def handle_card_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    file_id = update.message.photo[-1].file_id
    file = await context.bot.get_file(file_id)
    file_url = file.file_path

    analysis = analyze_card_image(file_id)
    card_type = analysis.get("type", "")
    card_text = card_type if card_type else "Monster"

    if is_monster_card(card_text):
        await update.message.delete()
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🛡️ Defense Mode", callback_data=f"defense|{file_id}")],
            [InlineKeyboardButton("⚔️ Attack Mode", callback_data=f"attack|{file_id}")]
        ])
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=file_id,
            caption="🃏 Monster Card Played!",
            reply_markup=kb
        )

async def handle_flip_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode, file_id = query.data.split("|", 1)
    await query.message.delete()

    try:
        file = await context.bot.get_file(file_id)
        file_url = file.file_path

        if mode == "defense":
            rotated_img = await rotate_image(file_url, 90)
            if rotated_img:
                await context.bot.send_photo(
                    chat_id=query.message.chat.id,
                    photo=rotated_img,
                    caption="🛡️ Card in Defense Mode"
                )
        elif mode == "attack":
            await context.bot.send_photo(
                chat_id=query.message.chat.id,
                photo=file_id,
                caption="⚔️ Card in Attack Mode"
            )
    except Exception as e:
        print(f"Flip mode error: {e}")
        await context.bot.send_message(
            chat_id=query.message.chat.id,
            text="⚠️ Error processing card rotation. Please try again."
        )

async def save_card(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)

    if 'authenticated' not in context.user_data or not context.user_data['authenticated']:
        context.user_data['waiting_for_password'] = True
        return await update.message.reply_text("🔒 Enter password to access card saving:")

    if 'temp_photos' not in context.user_data or not context.user_data['temp_photos']:
        return await update.message.reply_text("⚠️ Send card photos first, then /save.")

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🤖 Save Automatically", callback_data="save_auto")],
        [InlineKeyboardButton("👤 Save Manually", callback_data="save_manual")]
    ])
    await update.message.reply_text("Choose save method:", reply_markup=kb)

async def handle_group_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except Exception as e:
        print(f"Query answer error (continuing anyway): {e}")

    user_id = str(query.from_user.id)
    data = query.data

    if data == "save_auto":
        if 'temp_photos' not in context.user_data or not context.user_data['temp_photos']:
            try:
                await query.edit_message_text("⚠️ No photos to save. Send photos first.")
            except Exception as e:
                await context.bot.send_message(
                    chat_id=query.message.chat.id,
                    text="⚠️ No photos to save. Send photos first."
                )
            return

        photos = context.user_data['temp_photos']
        temp_message_ids = context.user_data.get('temp_message_ids', {})
        saved_count = 0
        duplicate_count = 0
        unrecognized_cards = []
        saved_card_messages = []
        duplicate_card_messages = []

        await query.edit_message_text("🔄 Processing cards automatically...")

        for i, file_id in enumerate(photos):
            analysis = analyze_card_image(file_id)
            detected_type = analysis.get('type')
            confidence = analysis.get('confidence', 0)
            image_bytes = analysis.get('image_bytes')

            is_duplicate = False
            if image_bytes and is_duplicate_card(user_id, image_bytes):
                is_duplicate = True
            public_collection = get_user_cards('public')
            if file_id in public_collection:
                is_duplicate = True

            if is_duplicate:
                duplicate_count += 1
                duplicate_card_messages.append(file_id)
                if file_id in temp_message_ids:
                    try:
                        await context.bot.send_message(
                            chat_id=query.message.chat.id,
                            text="🔄 This card is already in your collection (duplicate)",
                            reply_to_message_id=temp_message_ids[file_id]
                        )
                    except Exception as e:
                        print(f"Could not reply to duplicate card: {e}")
                continue

            if detected_type and confidence >= 40:
                save_card_to_db(
                    user_id=user_id,
                    file_id=file_id,
                    card_type=detected_type,
                    atk=analysis.get('ATK'),
                    defense=analysis.get('DEF'),
                    level=analysis.get('level')
                )
                if image_bytes:
                    save_card_hash(user_id, file_id, image_bytes)
                saved_count += 1
                saved_card_messages.append(file_id)
                print(f"Auto-saved {detected_type} card with {confidence}% confidence")
            else:
                unrecognized_cards.append(file_id)
                print(f"Card needs manual classification - detected: {detected_type}, confidence: {confidence}%")

        context.user_data['unrecognized_cards'] = unrecognized_cards
        deleted_count = 0
        for file_id in saved_card_messages:
            if file_id in temp_message_ids:
                message_id = temp_message_ids[file_id]
                try:
                    await context.bot.delete_message(
                        chat_id=query.message.chat.id,
                        message_id=message_id
                    )
                    deleted_count += 1
                except Exception as e:
                    print(f"Could not delete message {message_id}: {e}")

        context.user_data['temp_photos'] = []
        context.user_data['deck_temp_photos'] = []
        context.user_data['temp_message_ids'] = {}

        response_parts = []
        if saved_count > 0:
            response_parts.append(f"✅ <b>Successfully saved:</b> {saved_count} cards to their correct groups")
            response_parts.append(f"🗑️ <b>Deleted from chat:</b> {deleted_count} saved card photos")
        if duplicate_count > 0:
            response_parts.append(f"🔄 <b>Duplicates found:</b> {duplicate_count} cards (already in collection)")
            response_parts.append(" └ Check the replies above - duplicates were not deleted")
        if unrecognized_cards:
            response_parts.append(f"❓ <b>Unrecognized cards:</b> {len(unrecognized_cards)} cards remain in chat")
            response_parts.append(" └ Bot couldn't determine card type automatically")
            response_parts.append(" └ These cards were NOT deleted from chat")
            response_parts.append(" └ Use /groups to manually classify them")

        if not response_parts:
            response = "⚠️ No cards were processed."
        else:
            response = "\n".join(response_parts)
            if unrecognized_cards:
                response += f"\n\n💡 <b>Next step:</b> Type /groups to classify the {len(unrecognized_cards)} unrecognized cards."
            elif saved_count > 0:
                response += f"\n\n🎉 <b>All done!</b> {saved_count} cards successfully added to your collection."

        await query.edit_message_text(response, parse_mode="HTML")

    elif data == "save_manual":
        if 'temp_photos' not in context.user_data or not context.user_data['temp_photos']:
            try:
                await query.edit_message_text("⚠️ No photos to save. Send photos first.")
            except Exception as e:
                await context.bot.send_message(
                    chat_id=query.message.chat.id,
                    text="⚠️ No photos to save. Send photos first."
                )
            return

        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🐉 Monster", callback_data="group_monster")],
            [InlineKeyboardButton("✨ Spell", callback_data="group_spell")],
            [InlineKeyboardButton("🕳️ Trap", callback_data="group_trap")]
        ])
        await query.edit_message_text("Choose group to save these cards:", reply_markup=kb)

    elif data.startswith("group_"):
        group = data.split("_", 1)[1]
        if 'temp_photos' not in context.user_data or not context.user_data['temp_photos']:
            try:
                await query.edit_message_text("⚠️ No photos to save. Send photos first.")
            except Exception as e:
                await context.bot.send_message(
                    chat_id=query.message.chat.id,
                    text="⚠️ No photos to save. Send photos first."
                )
            return

        photos = context.user_data['temp_photos']
        saved_count = 0
        duplicate_count = 0

        for i, file_id in enumerate(photos, 1):
            analysis = analyze_card_image(file_id)
            image_bytes = analysis.get('image_bytes')

            if image_bytes and is_duplicate_card(user_id, image_bytes):
                duplicate_count += 1
                continue

            public_collection = get_user_cards('public')
            if file_id in public_collection:
                duplicate_count += 1
                continue

            save_card_to_db(
                user_id=user_id,
                file_id=file_id,
                card_type=group,
                atk=analysis.get('ATK'),
                defense=analysis.get('DEF'),
                level=analysis.get('level')
            )
            if image_bytes:
                save_card_hash(user_id, file_id, image_bytes)
            saved_count += 1

        context.user_data['temp_photos'] = []
        context.user_data['deck_temp_photos'] = []
        if 'temp_message_ids' in context.user_data:
            context.user_data['temp_message_ids'] = {}

        total_processed = len(photos)
        response = f"✅ <b>Cards Saved Successfully!</b>\n\n"
        response += f"📊 <b>Saved {saved_count} cards to {group.capitalize()} group</b>\n"
        if duplicate_count > 0:
            response += f"🔄 Skipped {duplicate_count} duplicates\n"
        response += f"📋 Total processed: {total_processed}"
        await query.edit_message_text(response, parse_mode="HTML")

    elif data.startswith("manual_"):
        group = data.split("_", 1)[1]
        if 'unrecognized_cards' not in context.user_data or not context.user_data['unrecognized_cards']:
            try:
                await query.edit_message_text("No cards to classify.")
            except:
                await context.bot.send_message(
                    chat_id=query.message.chat.id,
                    text="No cards to classify."
                )
            return

        current_card = context.user_data['unrecognized_cards'].pop(0)
        analysis = analyze_card_image(current_card)
        save_card_to_db(
            user_id=user_id,
            file_id=current_card,
            card_type=group,
            atk=analysis.get('ATK'),
            defense=analysis.get('DEF'),
            level=analysis.get('level')
        )
        image_bytes = analysis.get('image_bytes')
        if image_bytes:
            save_card_hash(user_id, current_card, image_bytes)

        try:
            await query.message.delete()
        except Exception as e:
            print(f"Could not delete message: {e}")

        if context.user_data['unrecognized_cards']:
            next_card = context.user_data['unrecognized_cards'][0]
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🐉 Monster", callback_data="manual_monster")],
                [InlineKeyboardButton("✨ Spell", callback_data="manual_spell")],
                [InlineKeyboardButton("🕳️ Trap", callback_data="manual_trap")]
            ])
            try:
                await context.bot.send_photo(
                    chat_id=query.message.chat.id,
                    photo=next_card,
                    caption=f"Cards remaining: {len(context.user_data['unrecognized_cards'])}\nChoose the correct group:",
                    reply_markup=kb
                )
            except Exception as e:
                print(f"Error sending next card: {e}")
                await context.bot.send_message(
                    chat_id=query.message.chat.id,
                    text=f"Error showing next card. Use /groups to continue."
                )
        else:
            await context.bot.send_message(
                chat_id=query.message.chat.id,
                text="✅ <b>Succeeded!</b>\nAll cards have been classified and saved.",
                parse_mode="HTML"
            )
            if 'unrecognized_cards' in context.user_data:
                del context.user_data['unrecognized_cards']

async def handle_delete_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)

    if query.data == "delete_select_card":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🐉 Monster", callback_data="delete_type_monster")],
            [InlineKeyboardButton("✨ Spell", callback_data="delete_type_spell")],
            [InlineKeyboardButton("🕳️ Trap", callback_data="delete_type_trap")]
        ])
        await query.edit_message_text(
            "🗑️ <b>Select Card Type to Delete</b>\n\n"
            "Choose a card type:",
            parse_mode="HTML",
            reply_markup=kb
        )
    elif query.data == "delete_format_collection":
        format_collection(user_id)
        await query.edit_message_text("✅ All cards have been deleted from your collection!")
    elif query.data.startswith("delete_type_"):
        card_type = query.data.split("_")[2]
        await query.edit_message_text(
            f"🗑️ Now type: @Theexper_bot {card_type}\n\n"
            f"Select a {card_type} card to delete permanently from your collection."
        )
        context.user_data['deletion_mode'] = 'collection'
        context.user_data['deletion_type'] = card_type
    elif query.data == "delete_deck_select":
        await query.edit_message_text(
            "🗑️ Now type: @Theexper_bot mydeck\n\n"
            "Select a card to delete from your deck."
        )
        context.user_data['deletion_mode'] = 'deck'
    elif query.data == "delete_deck_format":
        format_deck(user_id)
        await query.edit_message_text("✅ All cards have been deleted from your deck!")

async def mydeck(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != 'private':
        return await update.message.reply_text("⚠️ /mydeck command only works in private chat. Please message me directly.")

    user_id = str(update.effective_user.id)
    if 'deck_temp_photos' not in context.user_data:
        return await update.message.reply_text("⚠️ Send deck photos first.")

    photos = context.user_data['deck_temp_photos']
    count = len(photos)
    if count not in VALID_DECK_SIZES:
        return await update.message.reply_text(
            f"⚠️ Invalid count: {count}. "
            f"Your deck must be exactly one of {sorted(VALID_DECK_SIZES)} cards."
        )

    public_collection = get_user_cards('public')
    invalid_cards = []
    valid_cards = []
    for photo in photos:
        if photo in public_collection:
            valid_cards.append(photo)
        else:
            invalid_cards.append(photo)

    if invalid_cards:
        return await update.message.reply_text(
            f"⚠️ {len(invalid_cards)} card(s) are not in the public collection!\n"
            f"Only cards saved with /save can be added to your deck.\n"
            f"Valid cards: {len(valid_cards)}/{count}\n\n"
            f"Please only send cards from the public collection (type @Theexper_bot to see them)."
        )

    save_deck_to_db(user_id, valid_cards)
    context.user_data['deck_temp_photos'] = []
    await update.message.reply_text(f"✅ Your {count}-card deck is now saved!")

async def deletecollection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != 'private':
        return await update.message.reply_text("⚠️ /deletecollection command only works in private chat. Please message me directly.")

    if 'delete_authenticated' not in context.user_data or not context.user_data['delete_authenticated']:
        context.user_data['waiting_for_delete_password'] = True
        return await update.message.reply_text("🔒 Enter password to access deletion commands:")

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🃏 Select Card", callback_data="delete_select_card")],
        [InlineKeyboardButton("🗑️ Format", callback_data="delete_format_collection")]
    ])
    await update.message.reply_text(
        "🗑️ <b>Delete Collection</b>\n\n"
        "Choose an option:",
        parse_mode="HTML",
        reply_markup=kb
    )

async def deletedeck_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type != 'private':
        return await update.message.reply_text("⚠️ /deletedeck command only works in private chat. Please message me directly.")

    if 'delete_authenticated' not in context.user_data or not context.user_data['delete_authenticated']:
        context.user_data['waiting_for_delete_password'] = True
        return await update.message.reply_text("🔒 Enter password to access deletion commands:")

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🃏 Select Cards", callback_data="delete_deck_select")],
        [InlineKeyboardButton("🗑️ Format", callback_data="delete_deck_format")]
    ])
    await update.message.reply_text(
        "🗑️ <b>Delete Deck</b>\n\n"
        "Choose an option:",
        parse_mode="HTML",
        reply_markup=kb
    )

async def totalcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM user_cards WHERE user_id='public'")
        total = cursor.fetchone()[0]
        cursor = conn.execute("SELECT card_type, COUNT(*) FROM user_cards WHERE user_id='public' GROUP BY card_type")
        breakdown = cursor.fetchall()

    response = f"📊 <b>Total Cards in Collection: {total}</b>\n\n"
    if breakdown:
        response += "<b>Breakdown by Type:</b>\n"
        for card_type, count in breakdown:
            emoji = {"monster": "🐉", "spell": "✨", "trap": "🕳️"}.get(card_type, "🃏")
            response += f"{emoji} {card_type.capitalize()}: {count}\n"
    else:
        response += "No cards in collection yet."
    await update.message.reply_text(response, parse_mode="HTML")

async def groups_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    if 'unrecognized_cards' not in context.user_data or not context.user_data['unrecognized_cards']:
        return await update.message.reply_text("No cards waiting to be classified manually.")

    current_card = context.user_data['unrecognized_cards'][0]
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🐉 Monster", callback_data="manual_monster")],
        [InlineKeyboardButton("✨ Spell", callback_data="manual_spell")],
        [InlineKeyboardButton("🕳️ Trap", callback_data="manual_trap")]
    ])
    try:
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=current_card,
            caption=f"Cards remaining: {len(context.user_data['unrecognized_cards'])}\nChoose the correct group for this card:",
            reply_markup=kb
        )
    except Exception as e:
        print(f"Error in groups command: {e}")
        await update.message.reply_text(
            f"Error displaying card for classification. "
            f"Remaining cards: {len(context.user_data['unrecognized_cards'])}\n"
            f"Try again with /groups"
        )

async def draw_cards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id

    if 'sessions' not in context.bot_data:
        return await update.message.reply_text("⚠️ No active game. Use /playgame first.")

    active_session = None
    session_chat_id = None
    for cid, session in context.bot_data['sessions'].items():
        if user_id in session['players']:
            active_session = session
            session_chat_id = cid
            break

    if not active_session:
        return await update.message.reply_text("⚠️ You must join a game first with /playgame.")

    if chat_id != session_chat_id:
        return await update.message.reply_text("⚠️ You can only use /draw in the game chat where you joined the duel.")

    if not active_session.get('winner'):
        return await update.message.reply_text("⚠️ Game hasn't started yet. Wait for the number guessing phase to complete.")

    current_turn = active_session.get('current_turn')
    if current_turn is None:
        active_session['current_turn'] = active_session['winner']
        current_turn = active_session['current_turn']

    if user_id != current_turn:
        return await update.message.reply_text("⚠️ It's not your turn! You cannot draw cards.")

    deck = get_user_deck(user_id)
    if not deck:
        return await update.message.reply_text("⚠️ No deck found. Build one with /mydeck in private chat.")

    command_text = update.message.text.strip()
    number_match = re.search(r'/draw(?:@\w+)?\s*(\d+)', command_text)
    if not number_match:
        return await update.message.reply_text("⚠️ Usage: /draw <number>")
    try:
        n = int(number_match.group(1))
    except:
        return await update.message.reply_text("⚠️ Usage: /draw <number>")

    if n < 1 or n > len(deck):
        return await update.message.reply_text(f"⚠️ Draw between 1 and {len(deck)}.")

    drawn = random.sample(deck, n)
    save_drawn_cards(user_id, drawn)
    remove_from_deck(user_id, drawn)

    random_gif = random.choice(DRAW_READY_GIFS)
    await context.bot.send_animation(
        chat_id=update.effective_chat.id,
        animation=random_gif,
        caption=f"🎴 {n} cards drawn! Type: @Theexper_bot deck\nto view them.",
        reply_to_message_id=update.message.message_id
    )

async def inline_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.inline_query.query.strip().lower()
    uid = str(update.inline_query.from_user.id)
    results = []
    query_parts = q.split()

    if not query_parts or query_parts == [""]:
        cards, total_pages, total_count = get_user_cards_paginated(uid, page=1)
        for i, fid in enumerate(cards):
            result_id = f"all_p1_{i}_{uid}"
            caption = f"Card #{i+1} • Page 1/{total_pages}" if total_pages > 1 else f"Card #{i+1}"
            results.append(InlineQueryResultCachedPhoto(
                id=result_id,
                photo_file_id=fid,
                caption=caption
            ))
    elif len(query_parts) == 1 and query_parts[0].isdigit():
        page = int(query_parts[0])
        cards, total_pages, total_count = get_user_cards_paginated(uid, page=page)
        for i, fid in enumerate(cards):
            result_id = f"all_p{page}_{i}_{uid}"
            global_card_num = (page - 1) * 50 + i + 1
            caption = f"Card #{global_card_num} • Page {page}/{total_pages}" if total_pages > 1 else f"Card #{global_card_num}"
            results.append(InlineQueryResultCachedPhoto(
                id=result_id,
                photo_file_id=fid,
                caption=caption
            ))
    elif query_parts[0] == "collection":
        if len(query_parts) == 1:
            cards, total_pages, total_count = get_user_cards_paginated(uid, page=1)
            for i, fid in enumerate(cards):
                result_id = f"col_p1_{i}_{uid}"
                caption = f"Collection Card #{i+1} • Page 1/{total_pages}" if total_pages > 1 else f"Collection Card #{i+1}"
                results.append(InlineQueryResultCachedPhoto(
                    id=result_id,
                    photo_file_id=fid,
                    caption=caption
                ))
        elif len(query_parts) == 2 and query_parts[1].isdigit():
            page = int(query_parts[1])
            cards, total_pages, total_count = get_user_cards_paginated(uid, page=page)
            for i, fid in enumerate(cards):
                result_id = f"col_p{page}_{i}_{uid}"
                global_card_num = (page - 1) * 50 + i + 1
                caption = f"Collection Card #{global_card_num} • Page {page}/{total_pages}" if total_pages > 1 else f"Collection Card #{global_card_num}"
                results.append(InlineQueryResultCachedPhoto(
                    id=result_id,
                    photo_file_id=fid,
                    caption=caption
                ))
    elif query_parts[0] in ["monster", "spell", "trap"]:
        card_type = query_parts[0]
        if len(query_parts) == 1:
            cards, total_pages, total_count = get_user_cards_paginated(uid, card_type=card_type, page=1)
            for i, fid in enumerate(cards):
                result_id = f"{card_type}_p1_{i}_{uid}"
                caption = f"{card_type.capitalize()} #{i+1} • Page 1/{total_pages}" if total_pages > 1 else f"{card_type.capitalize()} #{i+1}"
                results.append(InlineQueryResultCachedPhoto(
                    id=result_id,
                    photo_file_id=fid,
                    caption=caption
                ))
        elif len(query_parts) == 2 and query_parts[1].isdigit():
            page = int(query_parts[1])
            cards, total_pages, total_count = get_user_cards_paginated(uid, card_type=card_type, page=page)
            for i, fid in enumerate(cards):
                result_id = f"{card_type}_p{page}_{i}_{uid}"
                global_card_num = (page - 1) * 50 + i + 1
                caption = f"{card_type.capitalize()} #{global_card_num} • Page {page}/{total_pages}" if total_pages > 1 else f"{card_type.capitalize()} #{global_card_num}"
                results.append(InlineQueryResultCachedPhoto(
                    id=result_id,
                    photo_file_id=fid,
                    caption=caption
                ))
    elif q in ("mydeck", "my deck"):
        deck = get_user_deck(uid)
        for i, fid in enumerate(deck):
            result_id = f"md{i}{uid}"
            results.append(InlineQueryResultCachedPhoto(
                id=result_id,
                photo_file_id=fid,
                caption=f"Deck Card #{i+1}"
            ))
    elif q in ("deck", "decks"):
        drawn = get_drawn_cards(uid)
        for i, fid in enumerate(drawn):
            result_id = f"drawn_{i}_{uid}_{fid[:10]}"
            results.append(InlineQueryResultCachedPhoto(
                id=result_id,
                photo_file_id=fid,
                caption=f"Drawn Card #{i+1}"
            ))

    await update.inline_query.answer(results, cache_time=0)

async def handle_chosen_inline_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = update.chosen_inline_result
    user_id = str(result.from_user.id)
    result_id = result.result_id

    if 'deletion_mode' in context.user_data:
        context.user_data['pending_deletion'] = True
        return

    if result_id.startswith("drawn_"):
        try:
            parts = result_id.split("_")
            if len(parts) >= 4:
                card_index = int(parts[1])
                drawn_cards = get_drawn_cards(user_id)
                if 0 <= card_index < len(drawn_cards):
                    file_id = drawn_cards[card_index]
                    remove_sent_card_from_drawn(user_id, file_id)
                    print(f"Removed drawn card {file_id} at index {card_index} for user {user_id}")
                    if 'pending_removals' not in context.bot_data:
                        context.bot_data['pending_removals'] = {}
                    context.bot_data['pending_removals'][user_id] = {
                        'original_file_id': file_id,
                        'timestamp': datetime.now().timestamp()
                    }
        except (ValueError, IndexError) as e:
            print(f"Could not parse result_id: {result_id}, error: {e}")

async def handle_sent_card(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return

    file_id = update.message.photo[-1].file_id
    user_id = str(update.message.from_user.id)
    chat_id = update.effective_chat.id

    if 'sessions' in context.bot_data:
        for cid, session in context.bot_data['sessions'].items():
            if user_id in session['players'] and cid == chat_id:
                current_turn = session.get('current_turn')
                if current_turn and current_turn != user_id:
                    try:
                        await update.message.delete()
                    except:
                        pass
                    await update.message.reply_text("⚠️ It's not your turn! You cannot send cards.")
                    return

    if 'deletion_mode' in context.user_data:
        deletion_mode = context.user_data['deletion_mode']
        if deletion_mode == 'collection':
            delete_card_from_collection(user_id, file_id)
            await update.message.reply_text("✅ Card deleted permanently from your collection and deck!")
        elif deletion_mode == 'deck':
            delete_card_from_deck_only(user_id, file_id)
            await update.message.reply_text("✅ Card deleted from your deck!")
        del context.user_data['deletion_mode']
        if 'deletion_type' in context.user_data:
            del context.user_data['deletion_type']
        if 'pending_deletion' in context.user_data:
            del context.user_data['pending_deletion']
        return

    if 'pending_removals' in context.bot_data and user_id in context.bot_data['pending_removals']:
        del context.bot_data['pending_removals'][user_id]
        print(f"Card removal already handled via chosen inline result for user {user_id}")
        return

    drawn_cards = get_drawn_cards(user_id)
    if file_id in drawn_cards:
        remove_sent_card_from_drawn(user_id, file_id)
        print(f"Removed card {file_id} from drawn cards for user {user_id}")
    else:
        if drawn_cards:
            first_card = drawn_cards[0]
            remove_sent_card_from_drawn(user_id, first_card)
            print(f"Removed first drawn card {first_card} for user {user_id} (file_id mismatch: {file_id})")
        else:
            print(f"No drawn cards found for user {user_id}")

async def playgame(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if update.effective_chat.type == 'private':
        return await update.message.reply_text("⚠️ /Playgame command only works in group chats.")

    if 'sessions' not in context.bot_data:
        context.bot_data['sessions'] = {}
    context.bot_data['sessions'][cid] = {
        'players': [],
        'message_id': None,
        'secret': None,
        'guess_idx': 0,
        'winner': None,
        'current_turn': None,
        'pre_battle_messages': []
    }

    caption = (
        "<b>YOU JUST STARTED THE YU-GI-OH BATTLE FIELD</b>\n"
        "1v1 Duel – each player must already have a valid deck via /mydeck.\n\n"
        "Tap 'Join Duel' when ready."
    )
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("Join Duel", callback_data="join_game")]])
    msg = await context.bot.send_photo(
        chat_id=cid,
        photo=BATTLE_IMG,
        caption=caption,
        parse_mode="HTML",
        reply_markup=kb
    )
    context.bot_data['sessions'][cid]['message_id'] = msg.message_id

async def handle_join_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    cid = query.message.chat.id
    uid = str(query.from_user.id)
    uname = query.from_user.username or query.from_user.first_name

    if 'sessions' not in context.bot_data or cid not in context.bot_data['sessions']:
        return await query.message.reply_text("No duel active. Use /playgame.")

    sess = context.bot_data['sessions'][cid]
    if uid in sess['players']:
        return await query.message.reply_text(f"⚠️ @{uname}, you already joined.")
    if len(sess['players']) >= 2:
        return await query.message.reply_text("⚠️ Duel is full.")

    sess['players'].append(uid)
    sess['pre_battle_messages'] = []
    img = random.choice(JOIN_IMAGES)
    join_msg = await context.bot.send_photo(
        chat_id=cid,
        photo=img,
        caption=f"<b>@{uname} HAS JOINED THE BATTLE!</b>\n⏱️ Joined at {datetime.now().strftime('%H:%M:%S')}",
        parse_mode="HTML"
    )
    sess['pre_battle_messages'].append(join_msg.message_id)
    if sess['message_id'] and sess['message_id'] not in sess['pre_battle_messages']:
        sess['pre_battle_messages'].append(sess['message_id'])

    if len(sess['players']) == 2:
        sess['secret'] = random.randint(1, 5)
        sess['guess_idx'] = 0
        try:
            first_player = await context.bot.get_chat_member(cid, int(sess['players'][0]))
            first_username = first_player.user.username or first_player.user.first_name
            msg = await context.bot.send_message(
                chat_id=cid,
                text=(
                    "<b>Guess a number between 1 and 5 to decide who goes first!</b>\n"
                    f"@{first_username}, you're up!"
                ),
                parse_mode="HTML"
            )
            sess['pre_battle_messages'].append(msg.message_id)
        except Exception as e:
            msg = await context.bot.send_message(
                chat_id=cid,
                text="<b>Guess a number between 1 and 5 to decide who goes first!</b>",
                parse_mode="HTML"
            )
            sess['pre_battle_messages'].append(msg.message_id)

async def handle_password_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if 'waiting_for_password' in context.user_data and context.user_data['waiting_for_password']:
        if text == "xooox":
            context.user_data['authenticated'] = True
            context.user_data['waiting_for_password'] = False
            await update.message.reply_text("✅ Access granted! You can now use /save to save cards.")
        else:
            context.user_data['waiting_for_password'] = False
            await update.message.reply_text("❌ Incorrect password. Access denied.")
        try:
            await update.message.delete()
        except:
            pass
        return True

    if 'waiting_for_delete_password' in context.user_data and context.user_data['waiting_for_delete_password']:
        if text == "xooox":
            context.user_data['delete_authenticated'] = True
            context.user_data['waiting_for_delete_password'] = False
            await update.message.reply_text("✅ Access granted! You can now use deletion commands.")
        else:
            context.user_data['waiting_for_delete_password'] = False
            await update.message.reply_text("❌ Incorrect password. Access denied.")
        try:
            await update.message.delete()
        except:
            pass
        return True
    return False

async def end_turn_only(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id

    if 'sessions' not in context.bot_data:
        return await update.message.reply_text("⚠️ No active game. Use /playgame first.")

    active_session = None
    session_chat_id = None
    for cid, session in context.bot_data['sessions'].items():
        if user_id in session['players']:
            active_session = session
            session_chat_id = cid
            break

    if not active_session:
        return await update.message.reply_text("⚠️ You must join a game first with /playgame.")

    if chat_id != session_chat_id:
        return await update.message.reply_text("⚠️ You can only use /endturn in the game chat where you joined the duel.")

    if not active_session.get('winner'):
        return await update.message.reply_text("⚠️ Game hasn't started yet.")

    current_turn = active_session.get('current_turn')
    if current_turn is None:
        active_session['current_turn'] = active_session['winner']
        current_turn = active_session['current_turn']

    if user_id != current_turn:
        return await update.message.reply_text("⚠️ It's not your turn!")

    other_player = None
    for player in active_session['players']:
        if player != user_id:
            other_player = player
            break

    if other_player:
        active_session['current_turn'] = other_player
        try:
            other_member = await context.bot.get_chat_member(chat_id, int(other_player))
            other_name = other_member.user.username or other_member.user.first_name
            random_gif = random.choice(DRAW_READY_GIFS)
            await context.bot.send_animation(
                chat_id=chat_id,
                animation=random_gif,
                caption=f"<b>@{other_name}</b>, it's your turn!\n\nType: <code>/draw 5</code>\n(tap to copy)",
                parse_mode="HTML"
            )
        except:
            random_gif = random.choice(DRAW_READY_GIFS)
            await context.bot.send_animation(
                chat_id=chat_id,
                animation=random_gif,
                caption="Next player's turn!\n\nType: <code>/draw 5</code>\n(tap to copy)",
                parse_mode="HTML"
            )

async def end_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    chat_id = update.effective_chat.id

    if 'sessions' not in context.bot_data:
        return await update.message.reply_text("⚠️ No active game to end.")

    active_session = None
    session_chat_id = None
    for cid, session in context.bot_data['sessions'].items():
        if user_id in session['players']:
            active_session = session
            session_chat_id = cid
            break

    if not active_session:
        return await update.message.reply_text("⚠️ You are not in an active game.")

    if chat_id != session_chat_id:
        return await update.message.reply_text("⚠️ You can only use /end in the game chat where you joined the duel.")

    del context.bot_data['sessions'][session_chat_id]

    try:
        p1, p2 = active_session['players']
        member1 = await context.bot.get_chat_member(chat_id, int(p1))
        member2 = await context.bot.get_chat_member(chat_id, int(p2))
        name1 = member1.user.username or member1.user.first_name
        name2 = member2.user.username or member2.user.first_name
        ender_name = update.effective_user.username or update.effective_user.first_name
        await update.message.reply_text(
            f"🏁 <b>GAME ENDED</b>\n\n"
            f"Game between @{name1} and @{name2} has been ended by @{ender_name}.\n\n"
            f"Use /playgame to start a new duel!",
            parse_mode="HTML"
        )
    except:
        await update.message.reply_text(
            f"🏁 <b>GAME ENDED</b>\n\n"
            f"The duel has been ended.\n\n"
            f"Use /playgame to start a new duel!",
            parse_mode="HTML"
        )

async def handle_guess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await handle_password_check(update, context):
        return

    cid = update.effective_chat.id
    text = update.message.text.strip()

    if 'sessions' not in context.bot_data or cid not in context.bot_data['sessions']:
        return
    sess = context.bot_data['sessions'][cid]

    if sess.get('winner') is not None:
        return
    if not text.isdigit() or not (1 <= int(text) <= 5):
        return

    uid = str(update.effective_user.id)
    idx = sess['guess_idx']
    if uid != sess['players'][idx]:
        return

    guess = int(text)
    sess['pre_battle_messages'].append(update.message.message_id)

    if guess == sess['secret']:
        sess['winner'] = uid
        uname = update.effective_user.username or update.effective_user.first_name
        msg = await update.message.reply_html(
            f"<b>@{uname} guessed correctly and will start the game!</b>"
        )
        sess['pre_battle_messages'].append(msg.message_id)
    else:
        sess['guess_idx'] += 1
        if sess['guess_idx'] < 2:
            other = sess['players'][1]
            try:
                other_member = await context.bot.get_chat_member(cid, int(other))
                name = other_member.user.username or other_member.user.first_name
                msg = await update.message.reply_text(f"Wrong! Now @{name}, your turn to guess.")
                sess['pre_battle_messages'].append(msg.message_id)
            except:
                msg = await update.message.reply_text("Wrong! Second player's turn to guess.")
                sess['pre_battle_messages'].append(msg.message_id)
        else:
            winner = random.choice(sess['players'])
            sess['winner'] = winner
            try:
                winner_member = await context.bot.get_chat_member(cid, int(winner))
                un = winner_member.user.username or winner_member.user.first_name
                msg = await update.message.reply_html(
                    f"No correct guesses—randomly selecting<br/><b>@{un} goes first!</b>"
                )
                sess['pre_battle_messages'].append(msg.message_id)
            except:
                msg = await update.message.reply_html(
                    "No correct guesses—randomly selecting winner!"
                )
                sess['pre_battle_messages'].append(msg.message_id)

    if sess.get('winner'):
        for msg_id in sess['pre_battle_messages']:
            try:
                await context.bot.delete_message(cid, msg_id)
            except:
                pass

        combined_img = await create_battle_begin_image()
        if combined_img:
            await context.bot.send_photo(
                chat_id=cid,
                photo=combined_img,
                caption="<b>[□¤THE BATTLE☆ BEGIN¤□¤]</b>",
                parse_mode="HTML"
            )
        else:
            await context.bot.send_photo(
                chat_id=cid,
                photo=BATTLE_BEGIN_IMGS[0],
                caption="<b>[□¤THE BATTLE☆ BEGIN¤□¤]</b>",
                parse_mode="HTML"
            )

        try:
            winner_member = await context.bot.get_chat_member(cid, int(sess['winner']))
            winner_name = winner_member.user.username or winner_member.user.first_name
            random_gif = random.choice(DRAW_READY_GIFS)
            await context.bot.send_animation(
                chat_id=cid,
                animation=random_gif,
                caption=f"<b>@{winner_name}</b> guessed correctly!\nYou start the game by drawing cards:\n\nType: <code>/draw 5</code>\n(tap to copy)",
                parse_mode="HTML"
            )
        except:
            random_gif = random.choice(DRAW_READY_GIFS)
            await context.bot.send_animation(
                chat_id=cid,
                animation=random_gif,
                caption="Winner starts the game by drawing cards:\n\nType: <code>/draw 5</code>\n(tap to copy)",
                parse_mode="HTML"
            )

        p1, p2 = sess['players']
        try:
            member1 = await context.bot.get_chat_member(cid, int(p1))
            member2 = await context.bot.get_chat_member(cid, int(p2))
            un1 = member1.user.username or member1.user.first_name
            un2 = member2.user.username or member2.user.first_name
            await context.bot.send_message(
                chat_id=cid,
                text=f"<b>@{un1} — L.P 4000</b>\n<b>@{un2} — L.P 4000</b>",
                parse_mode="HTML"
            )
        except:
            await context.bot.send_message(
                chat_id=cid,
                text="<b>Player 1 — L.P 4000</b>\n<b>Player 2 — L.P 4000</b>",
                parse_mode="HTML"
            )

# ─── MAIN APPLICATION ───
def main():
    print("🚀 Starting Yu-Gi-Oh Bot...")
    if not BOT_TOKEN:
        print("❌ ERROR: Bot token not set! Please set BOT_TOKEN environment variable.")
        exit(1)

    application = Application.builder().token(BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & filters.VIA_BOT, handle_sent_card))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.PHOTO, handle_card_photo))
    application.add_handler(CommandHandler("save", save_card))
    application.add_handler(CallbackQueryHandler(handle_group_choice, pattern="^(save_auto|save_manual|group_|manual_)"))
    application.add_handler(CommandHandler("mydeck", mydeck))
    application.add_handler(CommandHandler("draw", draw_cards))
    application.add_handler(CommandHandler("deletecollection", deletecollection))
    application.add_handler(CommandHandler("deletedeck", deletedeck_command))
    application.add_handler(CallbackQueryHandler(handle_delete_callbacks, pattern="^delete_"))
    application.add_handler(InlineQueryHandler(inline_handler))
    application.add_handler(ChosenInlineResultHandler(handle_chosen_inline_result))
    application.add_handler(CommandHandler("playgame", playgame))
    application.add_handler(CallbackQueryHandler(handle_join_game, pattern="^join_game$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_guess))
    application.add_handler(CallbackQueryHandler(handle_flip_mode, pattern=r"^(defense|attack)\|"))
    application.add_handler(CommandHandler("end", end_game))
    application.add_handler(CommandHandler("endturn", end_turn_only))
    application.add_handler(CommandHandler("totalcards", totalcards))
    application.add_handler(CommandHandler("groups", groups_command))

    print("✅ Bot is live—full duel features enabled.")
    print("🎯 Bot is now polling for messages...")
    
    # Start Flask keep-alive server in a thread
    from flask import Flask
    server = Flask(__name__)
    
    @server.route('/')
    def home():
        return "Yu-Gi-Oh Bot is running!"
        
    def run_server():
        server.run(host="0.0.0.0", port=8080)
        
    import threading
    threading.Thread(target=run_server, daemon=True).start()
    
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
