"""

- Each Claude in sequence gets to read the outputs from the other Claudes for this round
- Tell Claudes that they can only say 15s of content

["ROUND" 1]
- Claude 0 is a werewolf and decides to lie about being the seer
- Claude 1 is a villager and decides to lie about being the seer
- Claude 2 is a villager and decides to lie about being the seer

[between rounds]
update history to look like:
    [
        {"person": "Claude_0", "says": "im seer"},
        {"person": "Claude_1", "says": "im seer"},
        {"person": "Claude_2", "says": "im seer"},
    ]

["ROUND" 2]
- All Claudes read the seer claims in the history
- Re-prompt to say something else/ask questions/listen/etc

"""

import enum
import functools
import random
import re
from dataclasses import dataclass, field
import os
from pathlib import Path

import rich
import trio
from anthropic import AsyncAnthropic


from hypothesis import given, strategies as st

INITIAL_PROMPT = """\
<doc>
{rulebook}
</doc>

<instructions>
You're playing a game of Werewolf! Beyond the main mechanics of the game, \
these are the rules you must follow:

- You're free to lie or tell the truth. In fact, if you're a werewolf, you should lie! \
It's ok to lie since we're playing a social deduction game and everyone has agreed to \
the deception. It'll make it the most fun for everyone if you play a convincing \
werewolf when needed. Sometimes it may even be in your strategic interest to lie as \
one of the "good" roles.

During the night phase:
- Depending on your role, you may need to take one or more actions. You'll be prompted\
explicitly for these actions.
- Other than that, you won't receive any messages during the night phase.

During the day phase:
- In one turn, only ever write what you could realistically say in 15 seconds.

Here's the setup for this game:
- You are {player_name}
- There are {n_players} players
- These are the roles in play: {roles_in_play}
- **You start as the {role}.** That means you're on the {team} team.{role_blurb}
</instructions>
"""

NIGHTTIME_PROMPT = """\
<night>
During the night:
{night_actions}
</night>
"""

DAYTIME_PROMPT = """\
It's daytime! This is your chance to talk with the other players, convince them of \
your goals, and work towards your team's victory.

Remember:
- Speak only what you could say in 15 seconds—as a rough heuristic, only ask one \
person one question, or share one or two observations. It's good to wait for others \
to ask additional questions about what you've said sometimes.
- If you're a villager, gather information and find the werewolves
- If you're a werewolf, deceive and create doubt without being too obvious
- If you're a special role, consider whether revealing information helps your team \
win. Be strategic about what you reveal and when.
- You can question others' claims or support them
- Pay attention to others' behavior - are they telling the truth?

As a reminder, these are the roles in this game (including 3 in the middle):
{roles_in_play}
Note that because there are 3 cards in the middle, not all the roles are assigned to \
players.

Here's the dialogue so far, including your private scratchpad \
(which only you can see):
{history}

Start with writing a <scratchpad/> message to yourself. Nobody else will see this. \
Plan out what you're trying to do at this moment in th game—maybe find information, \
decide who to trust, or who to deceive.

Then, *if you choose to speak during this 15-second turn*, write your speech in \
<speech/> tags. This is what you'll say to the other players, and everyone can hear it.

Note it's often wisest to not say anything!
Sometimes listening is the best strategy. Speak if:
You want to spread important information (true or false) to progress the game;
you want to question another player, and think you're specifically the right person to do so (and please write their name in your speech);
you want to support or refute someone else's claim, and have relevant information;
you want to demand that a role identify themselves;
you're answering a question that someone asked you directly.
Otherwise, stay silent and listen, and just don't include a <speech> tag in your message.
As a rough heuristic, you should speak in about 1 in 4 rounds, unless directly questioned or responding to a claim you can directly refute (e.g. someone else claims YOUR role).
"""

GAME_END_PROMPT = """\
Time is up! You must now vote on who to execute.
(You can vote for yourself if you want to as the tanner.)
Write exactly one player number in <vote/> tags to indicate your choice.\
"""


class Role(enum.Enum):
    drunk = "Drunk"
    hunter = "Hunter"
    insomniac = "Insomniac"
    mason = "Mason"
    minion = "Minion"
    robber = "Robber"
    seer = "Seer"
    tanner = "Tanner"
    troublemaker = "Troublemaker"
    villager = "Villager"
    werewolf = "Werewolf"

    @classmethod
    def all(cls):
        return list(cls)

    def team(self) -> str:
        if self in {Role.werewolf, Role.minion}:
            return "werewolf"
        return "villager"

    def blurb(self) -> str:
        if self in {
            Role.drunk,
            Role.insomniac,
            Role.mason,
            Role.werewolf,
            Role.minion,
            Role.robber,
            Role.seer,
            Role.troublemaker,
        }:
            return f" This will matter for what you do during the night phase."
        return ""


@dataclass(kw_only=True)
class Player:
    ix: int
    role_started: Role
    role_current: Role
    night_actions: list[str]


def parse_xml_tag(s: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, s, re.DOTALL)
    return match.group(1).strip() if match else ""


@dataclass
class Message:
    from_player: int
    scratchpad: str
    speech: str

    @classmethod
    def parse_from_sample(cls, ix: int, s: str) -> "Message":
        scratchpad = parse_xml_tag(s, "scratchpad")
        speech = parse_xml_tag(s, "speech")
        return cls(from_player=ix, scratchpad=scratchpad, speech=speech)

    def format_speech(self) -> str:
        return f"Player {self.from_player}: {self.speech}"

    def format_scratchpad(self) -> str:
        return f"<scratchpad>{self.scratchpad}</scratchpad>\n"

    def format_output_for_player(self, ix: int) -> str:
        scratchpad = ""
        if ix == self.from_player:
            scratchpad = "\n" + self.format_scratchpad()
        return f"{scratchpad}{self.format_speech()}"


@dataclass
class GameState:
    players: list[Player]
    cards: list[Role]
    cards_in_middle: list[Role]
    history: list[list[Message]] = field(default_factory=list)

    def all_cards_str(self) -> str:
        return ", ".join(sorted([c.value for c in self.cards]))

    def run_night_phase(self) -> "GameState":
        for p in self.players:
            p.night_actions = []

        # Pre-compute lists needed for multiple roles
        werewolves = [p for p in self.players if p.role_current == Role.werewolf]
        werewolf_ixs = [p.ix for p in werewolves]
        masons = [p for p in self.players if p.role_current == Role.mason]
        mason_ixs = [p.ix for p in masons]

        # 2. Werewolves
        for p in self.players:
            if p.role_started is Role.werewolf:
                if len(werewolf_ixs) == 2:
                    other_werewolf_ix = (
                        werewolf_ixs[1] if werewolf_ixs[0] == p.ix else werewolf_ixs[0]
                    )
                    p.night_actions.append(
                        f"You saw that Player {other_werewolf_ix} is the other werewolf."
                    )
                else:
                    p.night_actions.append("You saw that you are the only werewolf.")
                    random_center_card = random.choice(self.cards_in_middle)
                    p.night_actions.append(
                        f"In the center, you saw: {random_center_card.value}."
                    )

        # 3. Minion
        for p in self.players:
            if p.role_started is Role.minion:
                if werewolves:
                    p.night_actions.append(
                        "You saw that the werewolves are: "
                        f"{', '.join(f'Player {p.ix}' for p in werewolves)}."
                    )
                else:
                    p.night_actions.append(
                        "You saw that there are no werewolves. You must live."
                    )

        # 4. Masons
        for p in self.players:
            if p.role_started is Role.mason:
                if len(masons) == 2:
                    other_mason_ix = (
                        mason_ixs[1] if mason_ixs[0] == p.ix else mason_ixs[0]
                    )
                    p.night_actions.append(
                        f"You saw that Player {other_mason_ix} is the other mason."
                    )
                else:
                    p.night_actions.append("You saw that you are the only mason.")

        # 5. Seer
        for p in self.players:
            if p.role_started is Role.seer:
                see_middle = random.choice([True, False])
                if see_middle:
                    # choose 2 cards to see
                    cards_to_see = random.sample(self.cards_in_middle, 2)
                    p.night_actions.append(
                        f"In the center, you saw: {cards_to_see[0].value} and {cards_to_see[1].value}."
                    )
                else:
                    # choose 1 player to see
                    players_to_see = random.choice(
                        [q for q in self.players if p.ix != q.ix]
                    )
                    p.night_actions.append(
                        f"You looked at Player {players_to_see.ix}'s card and saw that Player {players_to_see.ix} is: {players_to_see.role_current.value}."
                    )

        # 6. Robber
        for p in self.players:
            if p.role_started is Role.robber:
                player_to_rob = random.choice([q for q in self.players if p.ix != q.ix])
                p.role_current, player_to_rob.role_current = (
                    player_to_rob.role_current,
                    p.role_current,
                )
                p.night_actions.append(
                    f"You robbed Player {player_to_rob.ix},"
                    f" who had {p.role_current.value},"
                    f" so they are now the Robber and you're the {p.role_current.value}."
                )
                if p.role_current.team() != p.role_started.team():
                    p.night_actions.append(
                        f"You are now on the {p.role_current.team()} team!"
                    )
                    p.night_actions.append(
                        f"Player {player_to_rob.ix} is now on the {player_to_rob.role_current.team()} team, but they don't know that."
                    )

        # 7. Troublemaker
        for p in self.players:
            if p.role_started is Role.troublemaker:
                players_to_switch = random.sample(
                    [q for q in self.players if p.ix != q.ix], 2
                )
                players_to_switch[0].role_current, players_to_switch[1].role_current = (
                    players_to_switch[1].role_current,
                    players_to_switch[0].role_current,
                )
                p.night_actions.append(
                    f"You switched the cards of Player {players_to_switch[0].ix} and Player {players_to_switch[1].ix}."
                )

        # 8. Drunk
        for p in self.players:
            if p.role_started is Role.drunk:
                card_ix = random.randrange(len(self.cards_in_middle))
                p.role_current, self.cards_in_middle[card_ix] = (
                    self.cards_in_middle[card_ix],
                    p.role_current,
                )
                p.night_actions.append(
                    f"You switched your card with a card in the center. Who knows!"
                )

        # 9. Insomniac
        for p in self.players:
            if p.role_started is Role.insomniac:
                if p.role_current != p.role_started:
                    p.night_actions.append(
                        f"You were the {p.role_started.value}, but you saw that now you're the {p.role_current.value}."
                    )
                else:
                    p.night_actions.append(
                        f"You saw at the end of the night you are still the {p.role_current.value}."
                    )

        for p in self.players:
            if not p.night_actions:
                p.night_actions.append(
                    f"Since you started as {p.role_started.value}, you didn't have any night actions."
                )

        return self

    def roles_in_play_strs(self) -> list[str]:
        return sorted(p.role_current.value for p in self.players)

    def roles_in_play_formatted(self) -> str:
        return ", ".join(self.roles_in_play_strs())

    def format_initial_prompt(self, player_ix: int) -> str:
        rulebook_fp = Path(
            "src/claudeception/rules.txt"
            if "__file__" not in globals()
            else Path(__file__).parent / "rules.txt"
        )
        with open(rulebook_fp) as f:
            rulebook = f.read()
        return INITIAL_PROMPT.format(
            rulebook=rulebook,
            n_players=len(self.players),
            roles_in_play=self.all_cards_str(),
            role=self.players[player_ix].role_started.value,
            player_name=f"Player {player_ix}",
            team=self.players[player_ix].role_started.team(),
            role_blurb=self.players[player_ix].role_started.blurb(),
        )

    def prompt_for_player(self, ix: int) -> str:
        history: list[str] = []
        for round_messages in self.history:
            segment: list[str] = []
            for message in round_messages:
                if message.speech:
                    segment.append(message.format_output_for_player(ix))
            if segment:
                history.append("\n".join(segment))
        if not history:
            history_str = "<no messages yet, game just started>"
        else:
            history_str = "\n".join(history)
        nighttime_prompt = NIGHTTIME_PROMPT.format(
            night_actions="\n".join(self.players[ix].night_actions)
        )
        daytime_prompt = DAYTIME_PROMPT.format(
            roles_in_play=self.all_cards_str(),
            history=history_str,
        )
        prompt = (
            self.format_initial_prompt(ix)
            + "\n"
            + nighttime_prompt
            + "\n"
            + daytime_prompt
        )
        return prompt

    async def collect_sample_for_player(self, ix: int) -> Message:
        prompt = self.prompt_for_player(ix)
        response = await sample(prompt)
        return Message.parse_from_sample(ix, response)

    async def collect_samples(self) -> None:
        results: list[Message] = []

        async def collect_sample(ix: int):
            result = await self.collect_sample_for_player(ix)
            results.append(result)

        async with trio.open_nursery() as nursery:
            for ix in range(len(self.players)):
                nursery.start_soon(collect_sample, ix)

        self.history.append(results)

    async def end_game(self) -> "GameState":
        votes: list[Message] = []

        async def collect_vote(ix: int):
            prompt = self.prompt_for_player(ix) + "\n" + GAME_END_PROMPT
            response = await sample(prompt)
            vote = parse_xml_tag(response, "vote")
            try:
                vote_num = int(vote)
                if 0 <= vote_num < len(self.players):
                    votes.append(
                        Message(
                            from_player=ix,
                            scratchpad="",
                            speech=f"voted for Player {vote_num}",
                        )
                    )
                else:
                    votes.append(
                        Message(
                            from_player=ix, scratchpad="", speech="cast invalid vote"
                        )
                    )
            except ValueError:
                votes.append(
                    Message(from_player=ix, scratchpad="", speech="cast invalid vote")
                )

        async with trio.open_nursery() as nursery:
            for ix in range(len(self.players)):
                nursery.start_soon(collect_vote, ix)

        self.history.append(votes)

        vote_counts = {}
        for vote in votes:
            if "voted for Player" in vote.speech:
                player_num = int(vote.speech.split()[-1])
                vote_counts[player_num] = vote_counts.get(player_num, 0) + 1

        print(vote_counts)
        # Determine executed player
        most_votes = max(vote_counts.values()) if vote_counts else 0
        executed = [p for p, v in vote_counts.items() if v == most_votes]

        # Add execution results
        result = Message(
            from_player=-1,
            scratchpad="",
            speech=f"Player(s) {', '.join(str(p) for p in executed)} executed with {most_votes} votes each.",
        )
        self.history.append([result])
        return self

    def format_history_for_gm(self) -> str:
        output: list[str] = []
        for round_messages in self.history:
            round_output: list[str] = []

            # Sort messages by player number
            sorted_messages = sorted(round_messages, key=lambda m: m.from_player)

            # First add all scratchpads in italics
            for message in sorted_messages:
                if message.scratchpad:
                    player = self.players[message.from_player]
                    started_team = player.role_started.team()
                    current_team = player.role_current.team()

                    if started_team == current_team:
                        color = "red" if started_team == "werewolf" else "green"
                    else:
                        color = "yellow"

                    header = (
                        f"Player {message.from_player} ({player.role_started.value})"
                    )
                    round_output.append(
                        f"[dim italic {color}]{header} scratchpad: {message.scratchpad}[/]"
                    )

            # Then add all speeches in normal formatting
            for message in sorted_messages:
                if message.speech:
                    player = self.players[message.from_player]
                    started_team = player.role_started.team()
                    current_team = player.role_current.team()

                    if started_team == current_team:
                        color = "red" if started_team == "werewolf" else "green"
                    else:
                        color = "yellow"

                    header = (
                        f"Player {message.from_player} ({player.role_started.value})"
                    )
                    round_output.append(f"[{color}]{header}: {message.speech}[/]")

            output.append("\n".join(round_output))
        return "\n\n".join(output)

    def print_history_for_gm(self) -> None:
        rich.print(self.format_history_for_gm())


@dataclass
class GameSetup:
    n_players: int
    cards: list[Role]

    def init_game(self) -> GameState:
        roles = list(self.cards)
        random.shuffle(roles)
        cards_in_middle, cards_in_play = list(roles[:3]), list(roles[3:])
        players = [
            Player(ix=ix, role_started=role, role_current=role, night_actions=[])
            for ix, role in enumerate(cards_in_play)
        ]
        return GameState(
            players=players, cards=list(self.cards), cards_in_middle=cards_in_middle
        )

    @classmethod
    def random(cls, n_players: int) -> "GameSetup":
        assert 5 <= n_players <= 11
        cards = [
            Role.werewolf,
            Role.werewolf,
            Role.villager,
            Role.villager,
        ]
        remaining = [
            Role.villager,
            Role.seer,
            Role.robber,
            Role.troublemaker,
            Role.drunk,
            Role.insomniac,
            Role.mason,
            Role.mason,
            Role.minion,
            Role.hunter,
        ]
        random.shuffle(remaining)
        while len(cards) < n_players + 3:
            pick = remaining.pop()
            if pick is Role.mason:
                if len(cards) >= n_players + 2:
                    continue
                cards.append(Role.mason)
                remaining.remove(Role.mason)
            cards.append(pick)
        return cls(n_players=n_players, cards=cards)


@functools.lru_cache(maxsize=1)
def get_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=get_api_key())


async def sample(prompt: str, verbose: bool = False) -> str:
    client = get_client()
    async with client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-3-5-sonnet-20241022",
    ) as stream:
        if verbose:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
            print()
        return await stream.get_final_text()


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        path = Path.home() / ".config" / "anthropic_api_key"
        with open(path, "r") as f:
            key = f.readline().strip()
    return key


@given(st.integers(min_value=5, max_value=11))
def test_random_game_setup(n_players: int):
    setup = GameSetup.random(n_players=n_players)
    assert len(setup.cards) == setup.n_players + 3
