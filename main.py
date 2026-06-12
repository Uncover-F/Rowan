import asyncio
import os
import re
from datetime import datetime, timedelta

import aiosqlite
import discord
from discord.ext import commands
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()


# ----------------- TEXT COHERENCE ----------------- #
async def check_coherence(client, message_text):
    def run():
        return client.chat.completions.create(model="llama-3.3-70b-versatile",
                                              messages=[{"role": "system", "content": prompt},
                                                        {"role": "user", "content": message_text}], temperature=0,
                                              max_completion_tokens=256, top_p=1)

    result = await asyncio.to_thread(run)
    text = result.choices[0].message.content

    match = re.search(r"\{.*?\"coherence\"\s*:\s*(\d+).*?\}", text)

    print(text)
    return int(match.group(1)) if match else 11


# ----------------- IMAGE COHERENCE (GROQ VISION) ----------------- #
async def check_image_coherence(client, image_url, text=None):
    def run():
        content = []

        if text:
            content.append({"type": "text", "text": text})
        else:
            content.append({"type": "text", "text": prompt})

        content.append({"type": "image_url", "image_url": {"url": image_url}})

        return client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                                              messages=[{"role": "user", "content": content}], temperature=0,
                                              max_completion_tokens=256, top_p=1)

    result = await asyncio.to_thread(run)
    text_out = result.choices[0].message.content

    match = re.search(r"\{.*?\"coherence\"\s*:\s*(\d+).*?\}", text_out)

    print(text_out)
    return int(match.group(1)) if match else 11


# ----------------- DATABASE ----------------- #
class Database:
    def __init__(self):
        self.conn = None

    async def connect(self):
        self.conn = await aiosqlite.connect("store.db")

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY,
                account_created TEXT
            )
        """)

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT,
                timestamp TEXT,
                data TEXT
            )
        """)

        await self.conn.commit()

    async def createProfile(self, user_id, account_created):
        cursor = await self.conn.execute("SELECT user_id FROM profiles WHERE user_id = ?", (user_id,))
        if await cursor.fetchone():
            return

        await self.conn.execute("INSERT INTO profiles VALUES (?, ?)", (user_id, account_created))

        await self.conn.commit()

    async def addEvent(self, user_id, event_type, data=None):
        now = datetime.utcnow().isoformat()

        await self.conn.execute("""
            INSERT INTO events (user_id, event_type, timestamp, data)
            VALUES (?, ?, ?, ?)
        """, (user_id, event_type, now, str(data)))

        await self.conn.commit()

    async def getEvents(self, user_id, event_type=None):
        if event_type:
            cursor = await self.conn.execute("SELECT * FROM events WHERE user_id = ? AND event_type = ?",
                                             (user_id, event_type))
        else:
            cursor = await self.conn.execute("SELECT * FROM events WHERE user_id = ?", (user_id,))

        return await cursor.fetchall()


# ----------------- BOT ----------------- #
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

db = Database()


@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user}")
    await db.connect()
    print("Database connected")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    member = message.author

    # ----------------- RISK CALC ----------------- #
    rows = await db.getEvents(member.id, "update_risk")

    total_risk = 0
    for r in rows:
        try:
            total_risk += int(eval(r[4])["risk"])
        except:
            pass

    print(member.id, "->", total_risk)

    # ----------------- TEXT CHECK ----------------- #
    if message.content:
        if total_risk > 0:
            score = await check_coherence(client, message.content)

            if score <= 3:
                await message.delete()
                warn_msg = await message.channel.send(
                    "⚠️ Message removed.  If you think this was a mistake, please contact support.")
                await db.addEvent(member.id, "update_risk", {"risk": 1})
                await asyncio.sleep(10)
                await warn_msg.delete()

            elif 3 < score <= 10:
                await db.addEvent(member.id, "update_risk", {"risk": -2})

            elif score == 11:
                print("TEXT_INTERNAL_FAILURE")

    # ----------------- IMAGE CHECK ----------------- #
    if message.attachments and total_risk > 0:
        for att in message.attachments:
            if att.content_type and "image" in att.content_type:

                score = await check_image_coherence(client, att.url, message.content if message.content else None)

                if score <= 3:
                    await message.delete()
                    warn_msg = await message.channel.send(
                        "⚠️ Image removed. If you think this was a mistake, please contact support.")
                    await db.addEvent(member.id, "update_risk", {"risk": 2})
                    await asyncio.sleep(10)
                    await warn_msg.delete()


                elif score == 11:
                    print("IMAGE_INTERNAL_FAILURE")

    await bot.process_commands(message)


# ----------------- COMMANDS ----------------- #
@bot.command()
async def ping(ctx):
    await ctx.send("Pong! 🏓")


@bot.command()
async def whitelist(ctx, user: discord.User):
    """Set a user's risk level to 0"""
    await db.addEvent(user.id, "update_risk", {"risk": -999})
    await ctx.send(f"✅ Whitelisted {user.mention}")


# DEVTESTING - REMOVE
@bot.command()
async def getrisk(ctx, user: discord.User):
    """Get a user's current risk level"""
    rows = await db.getEvents(user.id, "update_risk")

    total_risk = 0
    for r in rows:
        try:
            total_risk += int(eval(r[4])["risk"])
        except:
            pass

    await ctx.send(f"📊 {user.mention}'s risk level: **{total_risk}**")


@bot.command()
async def addrisk(ctx, user: discord.User, points: int):
    """Add custom risk points to a user"""
    await db.addEvent(user.id, "update_risk", {"risk": points})
    await ctx.send(f"⚠️ Added **{points}** risk points to {user.mention}")


bot.run(os.getenv("DISCORD_TOKEN"))