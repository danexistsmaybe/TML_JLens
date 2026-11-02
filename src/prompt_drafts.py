"""Drafted prompts for pass@k and swap success. Edit here, then run src/check_prompts.py.

Written by the agent on 2026-11-01 and NOT yet reviewed by the user. "source" says where each
came from: "subagent" prompts were invented by low-context subagents and chosen by the agent
(all candidates and verdicts are in data/prompts/subagent_sweep.json); "agent" prompts were
written afterward by the agent, using what probing showed about this model.

Difficulty is the agent's guess, used only to keep a spread (easy, medium, hard).
"""


def _p(domain, kind, difficulty, prompt, intermediate, answer=None, source="agent"):
    return {"domain": domain, "type": kind, "difficulty": difficulty, "prompt": prompt,
            "intermediate": intermediate, "answer": answer, "source": source}


def _s(domain, category, template, args, source="agent"):
    """args: {argument: answer}, with answers written as they follow the prompt."""
    return {"domain": domain, "category": category, "template": template, "args": args, "source": source}


# ---------------------------------------------------------------------------------------
# Pass@k: wiki (multihop). The intermediate is an unstated bridge; the read position is the
# last token, immediately before the answer.
# ---------------------------------------------------------------------------------------
_CAPITAL_LANGUAGE = [("Paris", "France", "French"), ("Tokyo", "Japan", "Japanese"), ("Berlin", "Germany", "German"),
                     ("Moscow", "Russia", "Russian"), ("Madrid", "Spain", "Spanish"), ("Rome", "Italy", "Italian"),
                     ("Lisbon", "Portugal", "Portuguese"), ("Athens", "Greece", "Greek"), ("Beijing", "China", "Chinese"),
                     ("Seoul", "Korea", "Korean"), ("Ankara", "Turkey", "Turkish"), ("Warsaw", "Poland", "Polish"),
                     ("Stockholm", "Sweden", "Swedish"), ("Helsinki", "Finland", "Finnish"), ("Copenhagen", "Denmark", "Danish"),
                     ("Budapest", "Hungary", "Hungarian"), ("Hanoi", "Vietnam", "Vietnamese"), ("Bangkok", "Thailand", "Thai"),
                     ("Cairo", "Egypt", "Arabic"), ("Oslo", "Norway", "Norwegian"), ("Amsterdam", "Netherlands", "Dutch"),
                     ("Tehran", "Iran", "Persian"), ("Kiev", "Ukraine", "Ukrainian")]
_CURRENCY = [("Tokyo", "Japan", "yen"), ("Moscow", "Russia", "rub"), ("Beijing", "China", "yuan"), ("Berlin", "Germany", "euro"),
             ("London", "Britain", "pound"), ("Washington", "America", "dollar"), ("Seoul", "Korea", "won"),
             ("New Delhi", "India", "rupee"), ("Bern", "Switzerland", "franc")]

PASS_AT_K = [_p("wiki", "multihop", "easy", f"The people of the country whose capital is {c} speak", country, language)
             for c, country, language in _CAPITAL_LANGUAGE]
PASS_AT_K += [_p("wiki", "multihop", "medium", f"Fact: People in the city of {c} mostly speak", country, language)
              for c, country, language in _CAPITAL_LANGUAGE[:12]]
PASS_AT_K += [_p("wiki", "multihop", "medium", f"The currency of the country whose capital is {c} is the", country, money)
              for c, country, money in _CURRENCY]
PASS_AT_K += [
    _p("wiki", "multihop", "medium", "The currency used in the country whose capital is Madrid is the", "Spain", "euro", "subagent"),
    _p("wiki", "multihop", "medium", "The currency used in the country whose capital is Stockholm is the", "Sweden", "kron", "subagent"),
    _p("wiki", "multihop", "hard", "The author of the play Hamlet, who was born in Stratford-upon-Avon, was a native speaker of", "Shakespeare", "English", "subagent"),
]
PASS_AT_K += [_p("wiki", "multihop", "hard", p, i, a) for p, i, a in [
    ("Visitors to the Kremlin hear people speaking", "Russia", "Russian"),
    ("Visitors to Mount Fuji hear people speaking", "Japan", "Japanese"),
    ("Visitors to the Eiffel Tower hear people speaking", "France", "French"),
    ("The language spoken at the Eiffel Tower is", "Paris", "French"),
    ("The currency used near the Kremlin is the", "Moscow", "rub"),
    ("The currency used near the Taj Mahal is the", "India", "ru"),
    ("The currency used near Mount Fuji is the", "Japan", "yen"),
    ("Fact: The capital of the country where sushi comes from is", "Japan", "Tokyo"),
    ("Fact: The capital of the country that built the Great Wall is", "China", "Beijing"),
    ("Fact: The capital of the country where the Colosseum stands is", "Italy", "Rome"),
    ("Fact: The capital of the country of the pyramids of Giza is", "Egypt", "Cairo"),
    ("Fact: The capital of the country where pizza was invented is", "Italy", "Rome"),
    ("Fact: The chemical symbol of the metal in wedding rings is", "gold", "Au"),
    ("Fact: The chemical symbol of the gas we breathe to live is", "oxygen", "O"),
    ("Fact: The chemical symbol of the metal that rusts is", "iron", "Fe"),
    ("Fact: The chemical symbol of the metal in old coins and mirrors is", "silver", "Ag"),
    ("Fact: The chemical symbol of the lightest gas is", "hydrogen", "H"),
    ("Fact: The color of the planet fourth from the Sun is", "Mars", "red"),
    ("Fact: The color of the fruit that monkeys love is", "banana", "yellow"),
    ("Fact: The color of the vegetable that rabbits love is", "carrot", "orange"),
    ("Fact: The number of legs on the animal that spins webs is", "spider", "eight"),
    ("Fact: The number of legs on the insect that makes honey is", "bee", "six"),
    ("Fact: The sound made by the animal that chases mice is", "cat", "me"),
    ("Fact: The month after the month of Christmas is", "December", "January"),
    ("Fact: The month after the month of Halloween is", "October", "November"),
    ("Fact: The month before the month of Halloween is", "October", "September"),
    ("Fact: The season after the season when leaves fall is", "autumn", "winter"),
    ("Fact: The day after the day after Monday is", "Tuesday", "Wednesday"),
    ("Fact: The day before the day before Sunday is", "Saturday", "Friday"),
    ("Fact: The number of sides on a triangle plus one is", "three", "four"),
    ("Fact: Two times the number of wheels on a bicycle is", "two", "four"),
    ("Fact: The number of days in a week plus one is", "seven", "eight"),
    ("Fact: The first name of the author of Hamlet is", "Shakespeare", "William"),
    ("Fact: The first name of the scientist who developed relativity is", "Einstein", "Albert"),
    ("Fact: The first name of the first president of the United States is", "Washington", "George"),
    ("Fact: The language in which Goethe wrote was", "Germany", "German"),
    ("Fact: The language in which Cervantes wrote was", "Spain", "Spanish"),
    ("Fact: The language in which Tolstoy wrote was", "Russia", "Russian"),
]]

# ---------------------------------------------------------------------------------------
# Pass@k: stories (association). A short passage implies a concept; see the read position note below.
# ---------------------------------------------------------------------------------------
PASS_AT_K += [
    _p("stories", "association", "medium", "Tom put on his warm coat and walked out to the barn. He checked the hay, and the cows were eating, and the hens were sleeping. Then he sat on a log and watched the green rows of corn grow under the sun.", "farm", source="subagent"),
    _p("stories", "association", "easy", "Tom put on his warm coat and his red hat. He sat by the pond and held his little rod, waiting for a fish to pull the line.", "fishing", source="subagent"),
    _p("stories", "association", "medium", "Tom put on his warm coat and walked out the door. He saw the little ship sit in the dock, and he rang the bell on top of the tall tower, where the big light turned round and round to guide the boats at night.", "lighthouse", source="subagent"),
]
PASS_AT_K += [_p("stories", "association", d, p, i) for d, p, i in [
    ("easy", "Mia blew out the candles while everyone sang, and then she opened her presents.", "birthday"),
    ("easy", "Ben made a big ball, then a small ball, and put a carrot on top for a nose. It was cold, and his hands were wet.", "winter"),
    ("easy", "Sara built a castle with her bucket and spade. The waves came in and washed it away.", "beach"),
    ("easy", "The teacher wrote on the board, and all the children sat at their desks and opened their books.", "school"),
    ("easy", "Grey clouds filled the sky. Drops fell on the window, and Tim put on his boots to jump in the puddles.", "rain"),
    ("easy", "The white flakes fell all night. In the morning the whole yard was soft and white, and the kids got out their sleds.", "snow"),
    ("easy", "The man in the white coat looked in Lily's mouth and said, \"Say ah.\" Then he gave her some medicine.", "doctor"),
    ("easy", "Lily sat in the big chair and opened wide. The man counted her teeth and told her to brush every day.", "dentist"),
    ("easy", "They saw a lion, two tall giraffes and a sleeping bear in cages. Then they bought ice cream and fed the goats.", "zoo"),
    ("easy", "The clown fell down and the lions jumped through a ring of fire. The crowd in the big tent clapped.", "circus"),
    ("easy", "Mom put the dough in the oven. Soon the whole house smelled sweet, and she put icing and a cherry on top.", "cake"),
    ("easy", "He turned off the light and climbed under the blanket. Mom kissed him goodnight and he closed his eyes.", "sleep"),
    ("easy", "It had four legs, a wagging tail and a wet nose. It barked when Tom came home and licked his face.", "dog"),
    ("easy", "It was small and soft. It said meow, drank milk from a bowl and sat in the sun all day.", "cat"),
    ("easy", "It had big wings and a long tail, and it breathed fire. The knight was afraid of it.", "dragon"),
    ("medium", "They put up the tent by the lake and made a fire. They cooked sausages on sticks and looked at the stars.", "camping"),
    ("medium", "She waved a wand and said a strange word. The frog turned into a prince, and the pumpkin turned into a coach.", "magic"),
    ("medium", "The men had a big ship and a map with an X on it. They wore eye patches and dug for a chest of gold.", "pirate"),
    ("medium", "He had a long white beard and a red suit. He came down the chimney and left toys under the tree.", "Santa"),
    ("medium", "The trees were tall and dark all around them. Birds sang up high, and the path was covered with old leaves.", "forest"),
    ("medium", "The bees flew from flower to flower. Mom watered the roses and Tim pulled the weeds and dug a hole for the seeds.", "garden"),
    ("medium", "They got a cart and went up and down the rows. They bought eggs, bread and milk and paid at the front.", "shop"),
    ("medium", "The train went fast and the bags were packed. They were going far away for a week to see new places and rest.", "holiday"),
    ("medium", "Dad cut the turkey and everyone sat at the long table. Grandma passed the potatoes and the room was full of family.", "dinner"),
    ("medium", "The bright lights went up in the sky, boom and bang. Red and gold sparks fell, and everybody said ooh.", "fireworks"),
    ("medium", "A big truck came with a loud siren. The men held a long hose and put water on the burning house.", "fire"),
    ("medium", "She tied her shoes and ran as fast as she could. The others were right behind her, but she crossed the line first and got a ribbon.", "race"),
    ("medium", "He went up the ladder and sat high in the branches. He could see all the way to the river from up there.", "tree"),
    ("medium", "The boy was lost and began to cry. He had no idea where his mom had gone, and all the people around him were strangers.", "scared"),
    ("medium", "She shared her toys and held the door for the old man. She always helped and never said a mean word.", "kind"),
    ("medium", "Ben took the cookie when no one was looking. When Mom asked, he said he had not touched it, and his face went red.", "lie"),
    ("medium", "They laughed and played all day and told each other secrets. They had known each other since they were little and did everything together.", "friends"),
    ("hard", "After the storm the sun came out. High above the wet hills there was a curve of red, green and blue.", "rainbow"),
    ("hard", "Her tummy hurt and she had a hot head. Mom told her to stay home from school and rest.", "sick"),
    ("hard", "She looked at the empty bed where Max used to sleep. She kept his old collar in a box and cried when she thought of him.", "dead"),
    ("hard", "He wanted the red bike that Sam had. Every time he saw it his face got hot and he felt bad inside.", "jealous"),
    ("hard", "The leaves turned orange and fell from the trees. The air was cool, and the children kicked the piles on their way to school.", "autumn"),
    ("hard", "The flowers came up and the baby birds sang. The snow was gone, and it got warmer every day.", "spring"),
    ("hard", "The sun was hot and the days were long. They ate ice cream, swam in the pool and had no school at all.", "summer"),
    ("hard", "The rooster crowed and the sun came up. Tim rubbed his eyes and got up to eat his toast and eggs.", "morning"),
    ("hard", "The moon was high and the owl hooted. All the houses were dark and everyone was asleep.", "night"),
    ("hard", "Dad worked hard all year and had saved up. They counted the coins in the jar and there was finally enough.", "money"),
    ("hard", "The two boys would not share. They pushed each other and yelled, and soon both of them were crying on the floor.", "fight"),
    ("hard", "She said she would come back and she did. She had told him she would help and she kept her word.", "promise"),
    ("hard", "The knight bowed to the woman with the crown, and she sat on a tall gold chair in the middle of the castle.", "queen"),
    ("hard", "He opened the old box and his mouth fell open. Nobody had told him they were all hiding, and they jumped out and yelled.", "surprise"),
    ("hard", "The little seed was in the ground for a long time. Each day it got a bit of water, and one day a green shoot came up.", "grow"),
    ("hard", "The bird had a hurt wing. Lily put it in a box, fed it worms and was gentle, until one day it could fly away.", "care"),
    ("hard", "Everyone said he could not climb it. He fell two times, but he got up and tried again until he reached the top.", "brave"),
    ("hard", "They put the old toys in a bag and took them to children who had none. It made everyone feel warm inside.", "gift"),
    ("hard", "The little boat went up and down on the big water. Sam held on tight and his tummy did not feel good.", "sea"),
    ("hard", "The big gray animal lifted water with its long nose and sprayed it over its back.", "elephant"),
    ("hard", "It hopped around the garden with long ears and a fluffy tail, and it nibbled the carrots.", "rabbit"),
    ("hard", "It lived in a web in the corner of the room, and it had eight legs.", "spider"),
    ("hard", "The girl in the blue dress lost her glass shoe at midnight, and the prince looked for her all over the land.", "princess"),
    ("hard", "They tied a long string to it and ran across the field until the wind lifted it up into the sky.", "kite"),
    ("hard", "She put her foot on the pedal and held the bars. Dad let go, and she rolled down the road by herself.", "bike"),
    ("hard", "Round and round it went, with horses going up and down and music playing at the fair.", "carousel"),
    ("easy", "The big yellow car stopped at the corner each morning, and all the children climbed on with their lunch boxes.", "bus"),
    ("easy", "They got popcorn and sat in the dark. Everyone looked up at the big screen when the lights went down.", "movie"),
    ("medium", "She put on her soft shoes with long ribbons and twirled across the stage to the music.", "dance"),
    ("medium", "Mom opened the book at bedtime and read about a bear who lost his hat. Tim asked her to read it again.", "story"),
    ("medium", "They kicked it across the field and shouted with joy when it went between the two posts.", "soccer"),
    ("medium", "Everyone wore their best clothes. The bride smiled, the man in the suit gave her a ring, and they cut a tall white cake.", "wedding"),
]]

# Read position (decided 2026-11-02, on validation prompts). The paper reads association prompts
# at the closing period, but on this small model the readout there is sentence starters and
# formatting. So we append one token, chosen by asking: would this ending ask the model to NAME
# the target? We want targets that differ from what is obviously being prompted for.
#   * The story describes a thing, and " The" would ask the model to name it (the small animal
#     that says meow, then " The"). These end in a pronoun, so a verb comes next and the target
#     can't simply be the next word.
#   * The target is a setting, event or abstract concept. After " The", the obvious next word is
#     something in the scene (a wedding story gives "bride"), so the target is still latent.
#     These keep " The", even where the target also ranks high as a next token. That is a
#     judgment about each prompt, not a rank cutoff.
_NAMED_BY_DESCRIPTION = {
    "dog": " It", "cat": " It", "dragon": " It", "spider": " It", "elephant": " It", "rabbit": " It", "kite": " It",
    "carousel": " It", "rainbow": " It", "bike": " She", "princess": " She", "queen": " She", "doctor": " She",
    "dentist": " She", "cake": " She", "Santa": " He", "tree": " He", "rain": " He", "story": " He",
    "lighthouse": " He", "sea": " He", "bus": " They", "pirate": " They", "fire": " They", "snow": " They",
    "movie": " They", "fireworks": " They",
}
for _item in PASS_AT_K:
    if _item["domain"] == "stories":
        _item["prompt"] += _NAMED_BY_DESCRIPTION.get(_item["intermediate"], " The")

# ---------------------------------------------------------------------------------------
# Pass@k: python. Two kinds: evaluating an expression, and code that implies a concept.
# ---------------------------------------------------------------------------------------
_EVAL = [  # (difficulty, code, intermediate, answer). The read position is the trailing space.
    ("easy", 'x = (1 + 2) + 1  # x is ', "3", "4"),
    ("easy", 'x = (2 + 2) + 1  # x is ', "4", "5"),
    ("easy", 'x = (3 + 3) + 1  # x is ', "6", "7"),
    ("easy", 'x = (4 + 4) + 1  # x is ', "8", "9"),
    ("easy", 'print(2 * 3 + 1)  # prints ', "6", "7"),
    ("easy", 'print(4 * 2 + 1)  # prints ', "8", "9"),
    ("easy", 'print(2 * 2 + 3)  # prints ', "4", "7"),
    ("easy", 'print(max(1, 2) + 1)  # prints ', "2", "3"),
    ("medium", 'print(abs(-3) + 1)  # prints ', "3", "4"),
    ("medium", 'print(abs(-6) + 2)  # prints ', "6", "8"),
    ("medium", 'print(sum([2, 2]) + 1)  # prints ', "4", "5"),
    ("medium", 'print(min(5, 7) + 1)  # prints ', "5", "6"),
    ("medium", 'print(max(3, 8) - 2)  # prints ', "8", "6"),
    ("medium", 'x = len("abc") + 1  # x is ', "3", "4"),
    ("medium", 'x = len("hello") + 1  # x is ', "5", "6"),
    ("medium", 'print(len("abcd") + 1)  # prints ', "4", "5"),
    ("medium", 'print(len("ab") + 5)  # prints ', "2", "7"),
    ("medium", 'print(len([0, 0, 0]) + 1)  # prints ', "3", "4"),
    ("hard", 'x = 2 + 2\ny = x + 1\nprint(y + 1)  # prints ', "5", "6"),
    ("hard", 'x = 3\ny = x + 3\nprint(y + 1)  # prints ', "6", "7"),
    ("hard", 'a = "abc"\nn = len(a)\nprint(n + 1)  # prints ', "3", "4"),
    ("hard", 'def f():\n    return 2 + 3\n\nprint(f() + 1)  # prints ', "5", "6"),
    ("hard", 'items = ["a", "b", "c", "d"]\nprint(len(items) + 1)  # prints ', "4", "5"),
    ("hard", 'x = (3 + 4) * 1\nprint(x + 1)  # prints ', "7", "8"),
    ("medium", 'print(len("abcdef") + 2)  # prints ', "6", "8"),
    ("medium", 'print(len("abc") * 2)  # prints ', "3", "6"),
    ("medium", 'x = len("ab") + 3  # x is ', "2", "5"),
    ("easy", 'x = (5 + 2) + 1  # x is ', "7", "8"),
    ("easy", 'x = (6 + 3) - 1  # x is ', "9", "8"),
    ("easy", 'print(3 * 3 - 1)  # prints ', "9", "8"),
    ("easy", 'print(2 * 4 - 1)  # prints ', "8", "7"),
]
PASS_AT_K += [_p("python", "evaluate", d, c, i, a) for d, c, i, a in _EVAL]
PASS_AT_K += [
    _p("python", "implied", "medium", "def f(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n", "duplicate", source="subagent"),
    _p("python", "implied", "medium", "def f(items):\n    out = []\n    for x in items:\n        out.insert(0, x)\n    return out\n", "reverse", source="subagent"),
    _p("python", "implied", "hard", "def f(items):\n    line = []\n    for x in items:\n        line.append(x)\n    return line.pop(0)\n", "queue", source="subagent"),
]
PASS_AT_K += [_p("python", "implied", d, p, i) for d, p, i in [
    ("easy", "def f(r):\n    return 3.14 * r * r\n", "circle"),
    ("easy", "def f(w, h):\n    return w * h\n", "area"),
    ("easy", "def f(n):\n    if n < 2:\n        return n\n    return f(n - 1) + f(n - 2)\n", "fibonacci"),
    ("easy", "def f(n):\n    if n == 0:\n        return 1\n    return n * f(n - 1)\n", "factorial"),
    ("easy", "def f(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n", "prime"),
    ("easy", "def f(a):\n    return sum(a) / len(a)\n", "average"),
    ("easy", "with open(path) as f:\n    data = f.read()\n", "file"),
    ("easy", 'cursor.execute("SELECT * FROM users")\nrows = cursor.fetchall()\n', "database"),
    ("easy", "a, b = b, a\n", "swap"),
    ("easy", "x = a.split(',')\n", "string"),
    ("medium", "for i in range(n):\n    for j in range(n - 1):\n        if a[j] > a[j + 1]:\n            a[j], a[j + 1] = a[j + 1], a[j]\n", "sort"),
    ("medium", "def f(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return lo\n", "binary"),
    ("medium", "r = requests.get(url)\nprint(r.status_code)\n", "http"),
    ("medium", "s = socket.socket()\ns.connect((host, port))\n", "network"),
    ("medium", "t = threading.Thread(target=work)\nt.start()\nt.join()\n", "parallel"),
    ("medium", "lock.acquire()\ncount += 1\nlock.release()\n", "thread"),
    ("medium", "class S:\n    def __init__(self):\n        self.a = []\n    def push(self, x):\n        self.a.append(x)\n    def pop(self):\n        return self.a.pop()\n", "stack"),
    ("medium", "class Dog(Animal):\n    def speak(self):\n        return super().speak()\n", "inheritance"),
    ("medium", "def f(fn):\n    def g(*args):\n        print(args)\n        return fn(*args)\n    return g\n", "decorator"),
    ("medium", "def test_add():\n    assert add(1, 2) == 3\n", "unit"),
    ("medium", "for i in range(3):\n    for j in range(3):\n        c[i][j] = a[i][j] * b[i][j]\n", "matrix"),
    ("medium", 'm = re.match(r"\\d+", s)\n', "regex"),
    ("medium", 'd = datetime.now()\nprint(d.year, d.month, d.day)\n', "date"),
    ("medium", "h = hashlib.sha256(s.encode()).hexdigest()\n", "hash"),
    ("medium", 'if "@" in s and "." in s:\n    send(s)\n', "email"),
    ("medium", "x = random.randint(1, 6)\n", "dice"),
    ("hard", "if user.pw == given:\n    session.start()\n", "password"),
    ("hard", "total = price * qty\ntotal += total * 0.08\n", "tax"),
    ("hard", "a = [x * 2 for x in a]\n", "list"),
    ("hard", "d = {}\nfor w in words:\n    d[w] = d.get(w, 0) + 1\n", "count"),
    ("hard", "try:\n    x = int(s)\nexcept ValueError:\n    x = 0\n", "exception"),
    ("hard", "model.train()\nloss = loss_fn(model(x), y)\nloss.backward()\noptimizer.step()\n", "gradient"),
    ("hard", "app = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return 'hi'\n", "web"),
    ("hard", "def f(s):\n    return s == s[::-1]\n", "palindrome"),
    ("hard", "def f(c):\n    return c * 9 / 5 + 32\n", "temperature"),
    ("hard", "def f(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n", "divisor"),
    ("hard", "def f(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n", "leap"),
    ("hard", "def f(g, s):\n    q = [s]\n    while q:\n        v = q.pop(0)\n        q.extend(g[v])\n", "graph"),
    ("hard", "x, y = 0, 0\nx += dx\ny += dy\nscreen.blit(img, (x, y))\n", "game"),
    ("hard", "def f(p, q):\n    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5\n", "distance"),
    ("easy", 'print(f"{name} is {age}")\n', "format"),
    ("easy", "os.makedirs(p)\n", "directory"),
    ("medium", "for line in sys.stdin:\n    print(line.upper())\n", "input"),
    ("medium", 'x = b"abc".decode("utf-8")\n', "encoding"),
    ("medium", "s = s.strip().lower()\n", "clean"),
    ("medium", "def f(a):\n    return max(a) - min(a)\n", "range"),
    ("medium", "time.sleep(5)\n", "wait"),
    ("hard", 'print("bad value")\nsys.exit(1)\n', "error"),
]]

# "Implied" python prompts are read at the end of the code, with nothing appended. An earlier
# draft appended "def", but that asks for a function name, which is exactly what most targets
# are. Reading at the end of the code matched next-token guesses on none of the validation
# prompts, and separated the J-lens from the logit lens more clearly (2026-11-02).

# Words pinned to validation, so that prompts already scored with a lens stay out of test when
# drafts change. Everything else that passes the checks is eligible for test.
VAL_WORDS = {
    'wiki': ['America', 'India', 'Italy', 'Korea', 'Moscow', 'Netherlands'],
    'stories': ['birthday', 'bus', 'camping', 'gift', 'lie', 'queen', 'rain', 'soccer', 'surprise', 'wedding'],
    'python': ['8', 'area', 'database', 'graph', 'leap', 'parallel', 'prime', 'regex'],
}

# ---------------------------------------------------------------------------------------
# Swap templates. Arguments are listed in order of preference; the first 4 that pass are used.
# ---------------------------------------------------------------------------------------
_COUNTRIES = ["France", "Germany", "Japan", "Spain", "Italy", "Russia", "China"]
_CAPITALS = dict(France=" Paris", Germany=" Berlin", Japan=" Tokyo", Spain=" Madrid", Italy=" Rome", Russia=" Moscow", China=" Beijing")
_LANGUAGES = dict(France=" French", Germany=" German", Japan=" Japanese", Spain=" Spanish", Italy=" Italian", Russia=" Russian", China=" Chinese")
_CITIES = dict(Paris=" France", Berlin=" Germany", Tokyo=" Japan", Madrid=" Spain", Rome=" Italy", Moscow=" Russia")
_ELEMENTS = dict(gold=" Au", silver=" Ag", iron=" Fe", sodium=" Na", oxygen=" O")
_OPPOSITES = dict(hot=" cold", big=" small", up=" down", good=" bad", left=" right")
_MONTHS = dict(March=" April", January=" February", April=" May", June=" July")
_NUMBERS = dict(two=" three", three=" four", four=" five", five=" six", six=" seven")

SWAPS = [
    _s("wiki", "countries", "Fact: The capital of {arg} is", {c: _CAPITALS[c] for c in _COUNTRIES}),
    _s("wiki", "countries", "He traveled to the capital of {arg}, the city of", {c: _CAPITALS[c] for c in _COUNTRIES}),
    _s("wiki", "countries", "Fact: People in {arg} speak", {c: _LANGUAGES[c] for c in _COUNTRIES}),
    _s("wiki", "countries", "Fact: The currency of {arg} is the", {"Japan": " yen", "Germany": " Euro", "Mexico": " peso", "Britain": " pound"}),
    _s("wiki", "countries", "Fact: {arg} is a country in", {"France": " Europe", "Japan": " Asia", "Egypt": " Africa", "Brazil": " South"}),
    _s("wiki", "cities", "Fact: The city of {arg} is in", _CITIES),
    _s("wiki", "cities", "Fact: People in the city of {arg} mostly speak", {"Paris": " French", "Tokyo": " Japanese", "Berlin": " German", "Madrid": " Spanish"}),
    _s("wiki", "elements", "Fact: The chemical symbol for {arg} is", _ELEMENTS, "subagent"),
    _s("wiki", "words", "Fact: The opposite of {arg} is", _OPPOSITES),
    _s("wiki", "colors", "Fact: The color of {arg} is", {"grass": " green", "blood": " red", "snow": " white", "milk": " white", "coal": " black"}),
    _s("wiki", "time", "Fact: The month after {arg} is", _MONTHS),
    _s("wiki", "time", "Fact: The day after {arg} is", {"Monday": " Tuesday", "Saturday": " Sunday", "Friday": " Saturday", "Wednesday": " Thursday"}),
    _s("wiki", "numbers", "Fact: The number after {arg} is", _NUMBERS),
    _s("wiki", "animals", "Fact: A baby {arg} is called a", {"dog": " puppy", "cat": " kitten", "sheep": " lamb", "cow": " calf"}),

    _s("stories", "objects", "Lily had a {arg}. She liked to", {"kite": " fly", "book": " read", "bike": " ride", "song": " sing", "ball": " play"}),
    _s("stories", "objects", "She got a {arg}, so she started to", {"book": " read", "ball": " play", "bike": " ride", "kite": " fly"}),
    _s("stories", "objects", "Tom needed a {arg} so he could", {"pencil": " draw", "spoon": " eat", "ball": " play", "bed": " sleep"}),
    _s("stories", "objects", "Harry went to the hill with a {arg}. He wanted to", {"sled": " slide", "kite": " fly", "shovel": " dig", "camera": " take"}, "subagent"),
    _s("stories", "objects", "Wendy looked at the {arg}. She wanted to", {"window": " open", "worm": " eat", "wagon": " pull", "whistle": " blow"}, "subagent"),
    _s("stories", "objects", "In the garden, Gus saw a {arg}. He wanted to", {"ball": " play", "flower": " pick", "swing": " swing", "puddle": " jump"}, "subagent"),
    _s("stories", "feelings", "Tom was very {arg}. He began to", {"sad": " cry", "hungry": " eat", "thirsty": " drink", "tired": " sleep"}),
    _s("stories", "feelings", "Lily felt very {arg}. She began to", {"sad": " cry", "hungry": " eat", "thirsty": " drink", "tired": " sleep"}),
    _s("stories", "feelings", "Tom was very {arg}, so he wanted to", {"sad": " cry", "hungry": " eat", "thirsty": " drink", "tired": " sleep"}),
    _s("stories", "objects", "Ben had a {arg}. He liked to", {"kite": " fly", "book": " read", "bike": " ride", "song": " sing", "ball": " play"}),
    _s("stories", "objects", "Anna found a {arg}, and she started to", {"book": " read", "ball": " play", "bike": " ride", "kite": " fly"}),
    _s("stories", "feelings", "Mia was very {arg}. She began to", {"sad": " cry", "hungry": " eat", "thirsty": " drink", "tired": " sleep"}),
    _s("stories", "feelings", "The boy was very {arg}. He began to", {"sad": " cry", "hungry": " eat", "thirsty": " drink", "tired": " sleep"}),
    _s("stories", "things", "Sam touched the {arg}. It felt very", {"ice": " cold", "fire": " hot", "pillow": " soft", "rock": " hard"}),
    _s("stories", "animals", "The little {arg} can", {"dog": " run", "bird": " fly", "frog": " jump", "fish": " swim"}),
    _s("stories", "animals", "The {arg} can", {"dog": " run", "bird": " fly", "frog": " jump", "fish": " swim"}),
    _s("stories", "animals", "The {arg} liked to eat", {"monkey": " bananas", "rabbit": " carrots", "cat": " fish", "dog": " bones"}),
    _s("stories", "animals", "The {arg} likes to", {"frog": " jump", "cow": " eat", "bird": " fly", "fish": " swim"}),
    _s("stories", "things", "Lily touched the {arg}. It was very", {"ice": " cold", "fire": " hot", "pillow": " soft", "rock": " hard"}),
    _s("stories", "things", "The {arg} was very", {"sun": " hot", "ice": " cold", "snow": " deep", "fire": " hot"}),
    _s("stories", "things", "Tom ate the {arg} with a", {"soup": " spoon", "meat": " knife", "cake": " fork", "rice": " spoon"}),

    _s("python", "operators", "def {arg}(a, b):\n    return a", {"add": " +", "sub": " -", "mul": " *", "div": " /"}),
    _s("python", "operators", "def {arg}(x, y):\n    return x", {"add": " +", "sub": " -", "mul": " *", "div": " /"}),
    _s("python", "operators", "class Calc:\n    def {arg}(self, a, b):\n        return a", {"add": " +", "sub": " -", "mul": " *", "div": " /"}),
    _s("python", "operators", "{arg} = lambda a, b: a", {"add": " +", "mul": " *", "div": " /", "sub": " -"}),
    _s("python", "operators", "def {arg}(a, b):\n    return a", {"shl": " <<", "shr": " >>", "sub": " -", "add": " +"}, "subagent"),
    _s("python", "aliases", "import {arg} as", {"numpy": " np", "pandas": " pd", "tensorflow": " tf", "scipy": " sp", "seaborn": " sns"}),
    _s("python", "aliases", "try:\n    import {arg} as", {"numpy": " np", "pandas": " pd", "tensorflow": " tf", "seaborn": " sns"}),
    _s("python", "aliases", "import os\nimport {arg} as", {"numpy": " np", "pandas": " pd", "tensorflow": " tf", "scipy": " sp"}),
    _s("python", "aliases", "# {arg} is usually imported as", {"numpy": " np", "pandas": " pd", "tensorflow": " tf", "seaborn": " sns"}),
    _s("python", "modules", "from {arg} import", {"math": " sqrt", "json": " loads", "time": " sleep", "random": " randint", "functools": " reduce", "sys": " argv"}),
    _s("python", "modules", "import os\nfrom {arg} import", {"math": " sqrt", "json": " loads", "time": " sleep", "random": " randint", "functools": " reduce"}),
    _s("python", "types", "value = {arg}()\nprint(type(value))  # <class '", {"list": "list", "dict": "dict", "set": "set", "str": "str"}),
    _s("python", "types", "x = {arg}()\nprint(type(x))  # <class '", {"list": "list", "dict": "dict", "set": "set", "str": "str"}),
    _s("python", "logging", 'import logging\n\nlogger = logging.getLogger(__name__)\nlogger.{arg}("disk almost full")\nlogger.setLevel(logging.', {"debug": "DEBUG", "info": "INFO", "warning": "WARNING", "error": "ERROR"}, "subagent"),
]
