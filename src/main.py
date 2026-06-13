import asyncio
import os
import re
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import aiosqlite
import discord
from discord.ext import commands
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ============== LOGGING ============== #

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log", encoding="utf-8")])
log = logging.getLogger("modbot")


# ============== CONSTANTS ============== #

class CoherenceThreshold(Enum):
    REMOVE = 3
    INTERNAL_ERROR = 11


RISK_CONFIG = {"new_member_base": 10, "new_account_penalty": 15, "new_account_threshold_days": 14, "max_risk": 100,
               "image_check_max_risk": 50, "text_coherence_penalty": 3, "text_coherence_reward": -2,
               "image_coherence_penalty": 5, }

DEV_MODE = True  # Set to False to disable dev commands
DEV_IDS = {848031845454839810}  # Add your Discord user IDs here


def is_dev(user_id: int) -> bool:
    return DEV_MODE and user_id in DEV_IDS


# ============== DATA MODELS ============== #

@dataclass
class UserProfile:
    user_id: int
    account_created: datetime
    join_timestamp: Optional[datetime] = None
    current_risk: int = 0

    @property
    def account_age_days(self) -> int:
        return (datetime.utcnow() - self.account_created).days

    @property
    def is_new_account(self) -> bool:
        return self.account_age_days <= RISK_CONFIG["new_account_threshold_days"]


# ============== GROQ CLIENT ============== #

class GroqCoherence:
    def __init__(self, api_key: str, prompt_path: str = "prompt.txt"):
        self.client = Groq(api_key=api_key)
        self.prompt = self._load_prompt(prompt_path)

    @staticmethod
    def _load_prompt(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            log.warning(f"Prompt file '{path}' not found. Using default.")
            return "Analyze the coherence of the following text and respond with JSON: {\"coherence\": <score 0-10>}"

    async def check_text(self, text: str) -> int:
        def run():
            return self.client.chat.completions.create(model="llama-3.3-70b-versatile",
                                                       messages=[{"role": "system", "content": self.prompt},
                                                                 {"role": "user", "content": text}], temperature=0,
                                                       max_completion_tokens=256, top_p=1)

        try:
            result = await asyncio.to_thread(run)
            text_response = result.choices[0].message.content
            log.info(f"[TEXT_CHECK] Response: {text_response}")

            match = re.search(r"\{.*?\"coherence\"\s*:\s*(\d+).*?\}", text_response)
            return int(match.group(1)) if match else CoherenceThreshold.INTERNAL_ERROR.value
        except Exception as e:
            log.error(f"Text coherence check failed: {e}")
            return CoherenceThreshold.INTERNAL_ERROR.value

    async def check_image(self, image_url: str, caption: Optional[str] = None) -> int:
        def run():
            content = [{"type": "image_url", "image_url": {"url": image_url}}]

            if caption:
                content.insert(0, {"type": "text", "text": caption})
            else:
                content.insert(0, {"type": "text", "text": self.prompt})

            return self.client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                                                       messages=[{"role": "user", "content": content}], temperature=0,
                                                       max_completion_tokens=256, top_p=1)

        try:
            result = await asyncio.to_thread(run)
            text_response = result.choices[0].message.content
            log.info(f"[IMAGE_CHECK] Response: {text_response}")

            match = re.search(r"\{.*?\"coherence\"\s*:\s*(\d+).*?\}", text_response)
            return int(match.group(1)) if match else CoherenceThreshold.INTERNAL_ERROR.value
        except Exception as e:
            log.error(f"Image coherence check failed: {e}")
            return CoherenceThreshold.INTERNAL_ERROR.value


# ============== DATABASE ============== #

class Database:
    def __init__(self):
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self.conn = await aiosqlite.connect("sqlite.db")
        await self._create_tables()
        log.info("Database connected and tables ensured")

    async def _create_tables(self):
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY,
                account_created TEXT NOT NULL,
                join_timestamp TEXT,
                current_risk INTEGER DEFAULT 0
            )
        """)

        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT
            )
        """)

        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_user_events ON events(user_id, event_type)")
        await self.conn.commit()

    async def get_or_create_profile(self, user_id: int, account_created: datetime,
                                    join_timestamp: Optional[datetime] = None) -> UserProfile:
        """Get profile, or create with calculated risk (used on member join)"""
        cursor = await self.conn.execute(
            "SELECT account_created, join_timestamp, current_risk FROM profiles WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()

        if row:
            return UserProfile(user_id=user_id, account_created=datetime.fromisoformat(row[0]),
                               join_timestamp=datetime.fromisoformat(row[1]) if row[1] else None, current_risk=row[2])

        join_ts = join_timestamp or datetime.utcnow()
        initial_risk = self._calculate_initial_risk(account_created, join_ts)

        await self.conn.execute(
            "INSERT INTO profiles (user_id, account_created, join_timestamp, current_risk) VALUES (?, ?, ?, ?)",
            (user_id, account_created.isoformat(), join_ts.isoformat(), initial_risk))
        await self.conn.commit()

        return UserProfile(user_id=user_id, account_created=account_created, join_timestamp=join_ts,
                           current_risk=initial_risk)

    @staticmethod
    def _calculate_initial_risk(account_created: datetime, join_timestamp: datetime) -> int:
        """New members start at 10. If account <30 days old at join, add 20 (total 30)."""
        base_risk = RISK_CONFIG["new_member_base"]
        account_age = (join_timestamp - account_created).days

        if account_age < 0:
            return RISK_CONFIG["max_risk"]

        if account_age <= RISK_CONFIG["new_account_threshold_days"]:
            return min(base_risk + RISK_CONFIG["new_account_penalty"], RISK_CONFIG["max_risk"])

        return base_risk

    async def update_risk(self, user_id: int, delta: int) -> int:
        cursor = await self.conn.execute("SELECT current_risk FROM profiles WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()

        if not row:
            return 0

        new_risk = max(0, min(RISK_CONFIG["max_risk"], row[0] + delta))
        await self.conn.execute("UPDATE profiles SET current_risk = ? WHERE user_id = ?", (new_risk, user_id))
        await self.conn.commit()

        return new_risk

    async def add_event(self, user_id: int, event_type: str, data: Optional[str] = None):
        now = datetime.utcnow().isoformat()
        await self.conn.execute("INSERT INTO events (user_id, event_type, timestamp, data) VALUES (?, ?, ?, ?)",
                                (user_id, event_type, now, data))
        await self.conn.commit()

    async def get_risk(self, user_id: int) -> Optional[int]:
        """Returns current risk, or None if no profile exists"""
        cursor = await self.conn.execute("SELECT current_risk FROM profiles WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def ensure_profile_zero_risk(self, user_id: int, account_created: datetime):
        """Create a profile with risk 0 if missing (fallback for users without a join event)"""
        cursor = await self.conn.execute("SELECT user_id FROM profiles WHERE user_id = ?", (user_id,))
        if await cursor.fetchone():
            return

        await self.conn.execute(
            "INSERT INTO profiles (user_id, account_created, join_timestamp, current_risk) VALUES (?, ?, ?, ?)",
            (user_id, account_created.isoformat(), datetime.utcnow().isoformat(), 0))
        await self.conn.commit()
        log.info(f"Created fallback zero-risk profile for user {user_id}")

    async def set_risk(self, user_id: int, value: int) -> int:
        clamped = max(0, min(value, RISK_CONFIG["max_risk"]))
        await self.conn.execute("UPDATE profiles SET current_risk = ? WHERE user_id = ?", (clamped, user_id))
        await self.conn.commit()
        return clamped

    async def reset_risk(self, user_id: int):
        await self.set_risk(user_id, 0)


# ============== BOT ============== #

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=".", intents=intents, help_command=None)

db = Database()
groq: Optional[GroqCoherence] = None


@bot.event
async def on_ready():
    global groq
    await db.connect()
    groq = GroqCoherence(os.getenv("GROQ_API_KEY"))
    log.info(f"Bot logged in as {bot.user} (id: {bot.user.id})")


@bot.event
async def on_member_join(member: discord.Member):
    if member.bot:
        return

    profile = await db.get_or_create_profile(member.id, account_created=member.created_at,
                                             join_timestamp=datetime.utcnow())

    reason = "new_account_penalty" if profile.is_new_account else "new_member_base"
    await db.add_event(member.id, "member_join", f"Risk: {profile.current_risk} ({reason})")
    log.info(f"Member joined: {member} (id: {member.id}) -> risk {profile.current_risk} [{reason}]")


# NEW MESSAGE ____________
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Don't run moderation on commands
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    user_risk = await db.get_risk(message.author.id)

    if user_risk is None:
        await db.ensure_profile_zero_risk(message.author.id, message.author.created_at)
        user_risk = 0
        log.info(f"No profile found for {message.author} (id: {message.author.id}) -> set risk to 0")

    if user_risk <= 0:
        await bot.process_commands(message)
        return

    log.info(f"Checking message from {message.author} (id: {message.author.id}), current risk: {user_risk}")

    # CRITICAL RISK - AUTO DELETE, skip other checks
    if user_risk >= RISK_CONFIG["image_check_max_risk"]:
        await _handle_critical_risk(message)
        await bot.process_commands(message)
        return

    if message.content:
        await _handle_text_check(message)

    if message.attachments:
        await _handle_image_check(message)

    if message.embeds:
        await _handle_generic_message_removal(message)

    await bot.process_commands(message)


# EDITED MESSAGE ____________
@bot.event
async def on_message_edit(before, after):
    if after.author.bot:
        return

    # Don't run moderation on commands
    if after.content.startswith(bot.command_prefix):
        await bot.process_commands(after)
        return

    user_risk = await db.get_risk(after.author.id)

    if user_risk is None:
        await db.ensure_profile_zero_risk(after.author.id, after.author.created_at)
        user_risk = 0
        log.info(f"No profile found for {after.author} (id: {after.author.id}) -> set risk to 0")

    if user_risk <= 0:
        await bot.process_commands(after)
        return

    log.info(f"Checking message from {after.author} (id: {after.author.id}), current risk: {user_risk}")

    # CRITICAL RISK - AUTO DELETE, skip other checks
    if user_risk >= RISK_CONFIG["image_check_max_risk"]:
        await _handle_critical_risk(after)
        await bot.process_commands(after)
        return

    if after.content:
        await _handle_text_check(after)

    if after.attachments:
        await _handle_image_check(after)

    if after.embeds:
        await _handle_generic_message_removal(after)


# FUNCTIONS ____________
async def _handle_text_check(message: discord.Message):
    score = await groq.check_text(message.content)

    if score == CoherenceThreshold.INTERNAL_ERROR.value:
        log.warning(f"TEXT_INTERNAL_ERROR for {message.author} (id: {message.author.id})")
        return

    if score <= CoherenceThreshold.REMOVE.value:
        await message.delete()
        new_risk = await db.update_risk(message.author.id, RISK_CONFIG["text_coherence_penalty"])
        await db.add_event(message.author.id, "text_removed", f"Coherence: {score}, New Risk: {new_risk}")
        log.info(f"Removed text from {message.author} (coherence {score}) -> risk {new_risk}")

        warn_msg = await message.channel.send(
            f"{message.author.mention} - You have awakened the spam filter. You will not enjoy this.")
        await asyncio.sleep(8)
        await warn_msg.delete()
    else:
        new_risk = await db.update_risk(message.author.id, RISK_CONFIG["text_coherence_reward"])
        await db.add_event(message.author.id, "text_approved", f"Coherence: {score}, New Risk: {new_risk}")
        log.info(f"Approved text from {message.author} (coherence {score}) -> risk {new_risk}")


async def _handle_image_check(message: discord.Message):
    for attachment in message.attachments:
        if not (attachment.content_type and "image" in attachment.content_type):
            continue

        score = await groq.check_image(attachment.url, message.content or None)

        if score == CoherenceThreshold.INTERNAL_ERROR.value:
            log.warning(f"IMAGE_INTERNAL_ERROR for {message.author} (id: {message.author.id})")
            continue

        if score <= CoherenceThreshold.REMOVE.value:
            await message.delete()
            new_risk = await db.update_risk(message.author.id, RISK_CONFIG["image_coherence_penalty"])
            await db.add_event(message.author.id, "image_removed", f"Coherence: {score}, New Risk: {new_risk}")
            log.info(f"Removed image from {message.author} (coherence {score}) -> risk {new_risk}")

            warn_msg = await message.channel.send(
                f"{message.author.mention} - Thank you for your spam! Your wait time is approximately *forever*.")
            await asyncio.sleep(8)
            await warn_msg.delete()
            break


async def _handle_critical_risk(message: discord.Message):
    await message.delete()
    warn_msg = await message.channel.send(f"⚠️ {message.author.mention} - Message removed. "
                                          f"Please contact support and request to be whitelisted.")
    await db.add_event(message.author.id, "critical_risk_deletion", "Auto-deleted due to critical risk")
    log.info(f"Critical risk deletion for {message.author} (id: {message.author.id})")
    await asyncio.sleep(8)
    await warn_msg.delete()


async def _handle_generic_message_removal(message: discord.Message):
    await message.delete()
    warn_msg = await message.channel.send(
        f"⚠️ {message.author.mention} Embed Removed - We don't know you that well, try talking around for a while!")
    await db.add_event(message.author.id, "critical_risk_deletion", "Auto-deleted due to critical risk")
    log.info(f"Embed deletion for {message.author} (id: {message.author.id})")
    await asyncio.sleep(8)
    await warn_msg.delete()


# ============== COMMANDS ============== #

@bot.command(name="help")
async def help_cmd(ctx):
    embed = discord.Embed(title="🤖 Bot Commands", color=discord.Color.blurple())
    embed.add_field(name=".help", value="Show this message", inline=False)
    embed.add_field(name=".ping", value="Check bot latency", inline=False)
    embed.add_field(name=".whitelist @user", value="Mark a user as trusted, stops moderation checks (mod only)",
                    inline=False)

    if is_dev(ctx.author.id):
        embed.add_field(name="── Dev Commands ──", value="\u200b", inline=False)
        embed.add_field(name=".getrisk [@user]", value="View a user's current risk score", inline=False)
        embed.add_field(name=".setrisk @user <amount>", value="Manually set a user's risk", inline=False)
        embed.add_field(name=".addrisk @user <points>", value="Add/subtract risk points (can be negative)",
                        inline=False)

    await ctx.send(embed=embed)


@bot.command(name="ping")
async def ping(ctx):
    await ctx.send(f"🏓 Pong! Latency: {bot.latency * 1000:.0f}ms")


@bot.command(name="whitelist")
@commands.has_permissions(manage_messages=True)
async def whitelist_cmd(ctx, user: discord.User):
    """Mark a user as trusted - resets their risk to 0 so moderation stops checking them"""
    await db.reset_risk(user.id)
    await db.add_event(user.id, "whitelisted", f"Whitelisted by {ctx.author.name}")
    log.info(f"{ctx.author} whitelisted {user} (id: {user.id})")
    await ctx.send(f"✅ **{user.name}** is now trusted")


# ============== DEV-ONLY COMMANDS ============== #

def dev_only():
    async def predicate(ctx):
        if not is_dev(ctx.author.id):
            return False
        return True

    return commands.check(predicate)


@bot.command(name="getrisk")
@dev_only()
async def get_risk_cmd(ctx, user: Optional[discord.User] = None):
    target = user or ctx.author
    risk = await db.get_risk(target.id)
    max_risk = RISK_CONFIG["max_risk"]

    if risk is None:
        await db.ensure_profile_zero_risk(target.id, target.created_at)
        risk = 0

    bar_length = 10
    filled = int((risk / max_risk) * bar_length)
    risk_bar = "█" * filled + "░" * (bar_length - filled)

    await ctx.send(f"📊 **{target.name}** Risk Level\n{risk_bar} `{risk}/{max_risk}`")


@bot.command(name="setrisk")
@dev_only()
async def set_risk_cmd(ctx, user: discord.User, amount: int):
    risk = await db.get_risk(user.id)
    if risk is None:
        await db.ensure_profile_zero_risk(user.id, user.created_at)

    new_risk = await db.set_risk(user.id, amount)
    await db.add_event(user.id, "risk_adjusted", f"Set to {new_risk} by {ctx.author.name}")
    log.info(f"{ctx.author} set risk for {user} (id: {user.id}) to {new_risk}")
    await ctx.send(f"⚠️ **{user.name}** risk set to **{new_risk}/{RISK_CONFIG['max_risk']}**")


@bot.command(name="addrisk")
@dev_only()
async def add_risk_cmd(ctx, user: discord.User, points: int):
    risk = await db.get_risk(user.id)
    if risk is None:
        await db.ensure_profile_zero_risk(user.id, user.created_at)

    new_risk = await db.update_risk(user.id, points)
    await db.add_event(user.id, "risk_adjusted", f"Added {points} by {ctx.author.name}")
    log.info(f"{ctx.author} added {points} risk to {user} (id: {user.id}) -> {new_risk}")
    await ctx.send(f"⚠️ Added **{points}** risk to **{user.name}**. New risk: **{new_risk}/{RISK_CONFIG['max_risk']}**")


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CheckFailure):
        # Silently ignore - don't reveal dev-only commands exist
        return
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You need Manage Messages permission")
    elif isinstance(error, commands.MemberNotFound) or isinstance(error, commands.UserNotFound):
        await ctx.send("❌ Couldn't find that user")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing argument: `{error.param.name}`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("❌ Invalid argument provided")
    else:
        log.error(f"Unhandled command error: {error}", exc_info=error)
        await ctx.send("❌ An error occurred running that command")


bot.run(os.getenv("DISCORD_TOKEN"))
