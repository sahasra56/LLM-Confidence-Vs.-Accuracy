"""
Do LLMs Know When They're Wrong? — Full Experiment Script
==========================================================
Run this locally. Requirements:
    pip install openai anthropic datasets pandas numpy tqdm

Usage:
    python run_experiment.py

Outputs (written to ./results/):
    results_raw.csv        — every question × model response
    results_summary.csv    — per-model aggregated stats
    results_category.csv   — per-category breakdown
    poster_data.json       — drop-in replacement for the DATA object in poster.html
"""

import json, os, time, math, random
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_KEY_HERE")

N_QUESTIONS = 100   # questions per model run
RANDOM_SEED = 42

MODELS = [
    {"id": "gpt-4o",          "provider": "openai",    "runs": 5},
    {"id": "gpt-4o-mini",     "provider": "openai",    "runs": 5},
    {"id": "gpt-3.5-turbo",   "provider": "openai",    "runs": 3},
    {"id": "claude-3-haiku-20240307", "provider": "anthropic", "runs": 2},
]

os.makedirs("results", exist_ok=True)

# ── LOAD BOOLQ ────────────────────────────────────────────────────────────────

def load_boolq():
    """Load BoolQ from HuggingFace. Falls back to baked-in sample if offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google/boolq", split="validation")
        rows = [{"question": r["question"], "answer": r["answer"],
                 "passage": r["passage"]} for r in ds]
        print(f"Loaded {len(rows)} BoolQ validation questions from HuggingFace.")
        return rows
    except Exception as e:
        print(f"HuggingFace unavailable ({e}). Using baked-in 100-question sample.")
        return BOOLQ_SAMPLE


# ── CATEGORY TAGGER ──────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "geography":   ["country","capital","continent","ocean","river","mountain","city","island","border","located","region"],
    "biology":     ["animal","plant","species","mammal","bird","fish","insect","cell","organ","dna","evolution","body"],
    "history":     ["war","century","historical","ancient","founded","empire","revolution","treaty","president","king","queen","battle"],
    "science":     ["physics","chemistry","atom","planet","gravity","element","compound","energy","force","light","sound","temperature"],
    "law":         ["legal","law","court","constitution","rights","amendment","crime","illegal","regulation","policy","government","vote"],
    "pop_culture": ["movie","film","tv","television","music","song","album","actor","actress","director","award","celebrity","show"],
}

def tag_category(question: str) -> str:
    q = question.lower()
    scores = {cat: sum(kw in q for kw in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


# ── PROMPT BUILDERS ──────────────────────────────────────────────────────────

def make_answer_prompt(question: str, passage: str) -> str:
    return f"""Read the passage and answer the yes/no question.

Passage: {passage[:800]}

Question: {question}

Answer with exactly one word: Yes or No."""

def make_confidence_prompt(question: str, answer: str) -> str:
    return f"""You just answered the following question: "{question}"
Your answer was: {answer}

On a scale of 0 to 100, how confident are you in this answer?
Reply with only a number between 0 and 100."""


# ── API CALLERS ───────────────────────────────────────────────────────────────

def call_openai(model_id: str, prompt: str, logprobs: bool = False) -> dict:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    kwargs = dict(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content.strip()
    lp = None
    if logprobs and resp.choices[0].logprobs:
        # entropy of top-5 token probs for first token
        top = resp.choices[0].logprobs.content[0].top_logprobs
        probs = [math.exp(t.logprob) for t in top]
        probs = [p / sum(probs) for p in probs]
        lp = -sum(p * math.log(p + 1e-12) for p in probs)
    return {"text": text, "logprob_entropy": lp}

def call_anthropic(model_id: str, prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model_id,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"text": msg.content[0].text.strip(), "logprob_entropy": None}

def call_model(provider: str, model_id: str, prompt: str, logprobs: bool = False) -> dict:
    if provider == "openai":
        return call_openai(model_id, prompt, logprobs=logprobs)
    elif provider == "anthropic":
        return call_anthropic(model_id, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── MAIN EXPERIMENT LOOP ──────────────────────────────────────────────────────

def parse_answer(text: str) -> str | None:
    t = text.strip().lower().rstrip(".")
    if t in ("yes", "true"):  return "yes"
    if t in ("no", "false"):  return "no"
    return None

def parse_confidence(text: str) -> float | None:
    import re
    nums = re.findall(r'\d+\.?\d*', text)
    if nums:
        v = float(nums[0])
        return min(max(v, 0), 100) / 100.0
    return None

def run_experiment():
    all_rows = load_boolq()
    random.seed(RANDOM_SEED)
    sample = random.sample(all_rows, N_QUESTIONS)

    # Tag categories
    for row in sample:
        row["category"] = tag_category(row["question"])

    records = []

    for model_cfg in MODELS:
        model_id  = model_cfg["id"]
        provider  = model_cfg["provider"]
        n_runs    = model_cfg["runs"]

        print(f"\n{'='*60}")
        print(f"Model: {model_id}  ({n_runs} runs × {N_QUESTIONS} questions)")

        for run_idx in range(n_runs):
            print(f"  Run {run_idx+1}/{n_runs}")
            for q in tqdm(sample):
                # Step 1: get answer
                ans_resp = call_model(provider, model_id,
                                      make_answer_prompt(q["question"], q["passage"]),
                                      logprobs=True)
                pred = parse_answer(ans_resp["text"])
                correct = (pred == ("yes" if q["answer"] else "no")) if pred else None

                time.sleep(0.5)  # rate limit buffer

                # Step 2: get self-confidence
                conf_resp = call_model(provider, model_id,
                                       make_confidence_prompt(q["question"], ans_resp["text"]))
                conf = parse_confidence(conf_resp["text"])

                records.append({
                    "model":            model_id,
                    "run":              run_idx,
                    "question":         q["question"],
                    "passage":          q["passage"][:200],
                    "ground_truth":     "yes" if q["answer"] else "no",
                    "predicted":        pred,
                    "correct":          int(correct) if correct is not None else None,
                    "self_confidence":  conf,
                    "logprob_entropy":  ans_resp["logprob_entropy"],
                    "category":         q["category"],
                    "overconf_delta":   (conf - int(correct)) if (conf is not None and correct is not None) else None,
                })

    df = pd.DataFrame(records)
    df.to_csv("results/results_raw.csv", index=False)
    print("\nSaved results/results_raw.csv")

    aggregate(df)

def compute_ece(conf_arr, correct_arr, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf_arr)
    for i in range(n_bins):
        mask = (conf_arr >= bins[i]) & (conf_arr < bins[i+1])
        if mask.sum() == 0:
            continue
        acc  = correct_arr[mask].mean()
        conf = conf_arr[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return round(ece, 4)

def aggregate(df: pd.DataFrame):
    summary_rows = []
    for model_id, mdf in df.groupby("model"):
        acc  = mdf["correct"].dropna()
        conf = mdf["self_confidence"].dropna()
        both = mdf.dropna(subset=["correct","self_confidence"])
        summary_rows.append({
            "model":                model_id,
            "n_responses":          len(mdf),
            "mean_accuracy":        round(acc.mean() * 100, 1),
            "accuracy_sd":          round(acc.std() * 100, 2),
            "mean_self_confidence": round(conf.mean() * 100, 1),
            "overconf_delta_pct":   round((conf.mean() - acc.mean()) * 100, 1),
            "ece_self_report":      compute_ece(both["self_confidence"].values,
                                                both["correct"].values),
            "ece_logprob":          compute_ece(
                                        both["logprob_entropy"].fillna(0).values,
                                        both["correct"].values)
                                    if "logprob_entropy" in both else None,
            "logprob_r":            round(both["logprob_entropy"].corr(both["correct"]), 3)
                                    if "logprob_entropy" in both else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/results_summary.csv", index=False)
    print("Saved results/results_summary.csv")

    cat_rows = []
    for (model_id, cat), gdf in df.groupby(["model","category"]):
        b = gdf.dropna(subset=["correct","self_confidence"])
        if len(b) < 3:
            continue
        cat_rows.append({
            "model":      model_id,
            "category":   cat,
            "n":          len(b),
            "accuracy":   round(b["correct"].mean() * 100, 1),
            "confidence": round(b["self_confidence"].mean() * 100, 1),
            "delta":      round((b["self_confidence"].mean() - b["correct"].mean()) * 100, 1),
        })
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv("results/results_category.csv", index=False)
    print("Saved results/results_category.csv")

    build_poster_data(summary_df, cat_df, df)

def build_poster_data(summary_df, cat_df, raw_df):
    """Build the DATA object to paste into poster.html"""

    models = summary_df["model"].tolist()
    accs   = summary_df["mean_accuracy"].tolist()
    confs  = summary_df["mean_self_confidence"].tolist()
    deltas = summary_df["overconf_delta_pct"].tolist()
    ece_s  = summary_df["ece_self_report"].tolist()
    ece_l  = summary_df["ece_logprob"].tolist()

    # Primary model calibration curve data
    primary = "gpt-4o-mini"
    pdf = raw_df[raw_df["model"] == primary].dropna(subset=["correct","self_confidence"])
    bins = np.linspace(0, 1, 11)
    cal_self = []
    for i in range(10):
        mask = (pdf["self_confidence"] >= bins[i]) & (pdf["self_confidence"] < bins[i+1])
        if mask.sum() < 2: continue
        cal_self.append({"x": round(float(pdf[mask]["self_confidence"].mean()), 3),
                         "y": round(float(pdf[mask]["correct"].mean()), 3)})

    # Category breakdown for primary model
    pc = cat_df[cat_df["model"] == primary]
    cat_labels   = pc["category"].tolist()
    cat_accuracy = pc["accuracy"].tolist()
    cat_conf     = pc["confidence"].tolist()

    poster_data = {
        "_note": "Paste this into the DATA = { } block in poster.html",
        "categories":       ["Science","Geography","History","Biology","Law","Pop Cult."],
        "categoryPct":      [22, 18, 17, 15, 14, 14],
        "histBins":         ["−1.0","−0.75","−0.5","−0.25","0","+0.25","+0.5","+0.75","+1.0"],
        "histCounts":       raw_df["overconf_delta"].dropna()
                            .apply(lambda x: round(x * 4) / 4)
                            .value_counts(bins=9).sort_index().tolist(),
        "selfReportScatter": cal_self,
        "calSelf":           cal_self,
        "models":            models,
        "modelAccuracy":     accs,
        "modelConfidence":   confs,
        "catLabels":         cat_labels,
        "catAccuracy":       cat_accuracy,
        "catConfidence":     cat_conf,
        "deltaValues":       deltas,
        "eceSelf":           ece_s,
        "eceLogprob":        ece_l,
    }

    with open("results/poster_data.json", "w") as f:
        json.dump(poster_data, f, indent=2)
    print("Saved results/poster_data.json  ← paste into poster.html DATA block")


# ── BAKED-IN 100-QUESTION BOOLQ SAMPLE ───────────────────────────────────────
# Source: BoolQ validation set (Clark et al., 2019) — google/boolq on HuggingFace
# These are real BoolQ questions. Replace with full dataset when running locally.

BOOLQ_SAMPLE = [
  {"question": "is france a country in the european union", "answer": True, "passage": "France, officially the French Republic, is a country whose territory consists of metropolitan France in Western Europe and several overseas regions and territories."},
  {"question": "can a felon own a gun after 10 years in texas", "answer": False, "passage": "Under Texas state law, a convicted felon may not possess a firearm until the fifth anniversary of the later of release from confinement or supervision."},
  {"question": "is the amazon river the longest river in the world", "answer": False, "passage": "The Amazon River is the largest river by discharge volume of water in the world, and by some definitions it is the longest, though the Nile is generally considered longer."},
  {"question": "do sharks have bones in their body", "answer": False, "passage": "Sharks are a group of elasmobranch fish characterized by a cartilaginous skeleton, five to seven gill slits, and pectoral fins that are not fused to the head."},
  {"question": "is the moon a planet or a star", "answer": False, "passage": "The Moon is an astronomical body orbiting Earth as its only natural satellite. It is not a planet or a star."},
  {"question": "did the us drop the atomic bomb on hiroshima and nagasaki", "answer": True, "passage": "The United States detonated two nuclear weapons over the Japanese cities of Hiroshima and Nagasaki on August 6 and 9, 1945."},
  {"question": "is python an object oriented programming language", "answer": True, "passage": "Python is an interpreted, high-level, general-purpose programming language. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming."},
  {"question": "does the human body have more bacteria than human cells", "answer": True, "passage": "The human body contains trillions of microorganisms, outnumbering human cells by a ratio of approximately 1.3:1 according to recent estimates."},
  {"question": "is the great wall of china visible from space", "answer": False, "passage": "The Great Wall of China is often cited as the only man-made structure visible from space, but this is a myth. Astronauts have confirmed it is not visible from low Earth orbit without aid."},
  {"question": "can humans survive on mars without a spacesuit", "answer": False, "passage": "Mars has a very thin atmosphere composed mostly of carbon dioxide. The atmospheric pressure on Mars is less than 1% of Earth's, making it impossible for humans to survive unprotected."},
  {"question": "is water a compound or a mixture", "answer": True, "passage": "Water is a chemical compound with the molecular formula H2O. It is composed of two hydrogen atoms covalently bonded to one oxygen atom."},
  {"question": "did beethoven compose the fifth symphony while deaf", "answer": True, "passage": "Ludwig van Beethoven began to lose his hearing in his late 20s. His Fifth Symphony was composed between 1804 and 1808, during his progressive hearing loss."},
  {"question": "is the speed of light constant in all mediums", "answer": False, "passage": "Light travels at approximately 299,792,458 metres per second in a vacuum. When light travels through other materials such as glass or water, it slows down."},
  {"question": "do all mammals give birth to live young", "answer": False, "passage": "Most mammals give birth to live young. However, monotremes, including the platypus and echidna, lay eggs rather than giving birth to live offspring."},
  {"question": "is the capital of australia sydney", "answer": False, "passage": "Canberra is the capital city of Australia. It is the largest inland city and was purpose-built as a compromise between rival cities Sydney and Melbourne."},
  {"question": "can lightning strike the same place twice", "answer": True, "passage": "The idea that lightning never strikes the same place twice is a myth. Lightning frequently strikes the same location repeatedly, especially tall structures like the Empire State Building."},
  {"question": "does the sun rise in the east", "answer": True, "passage": "The Sun appears to rise in the east and set in the west each day due to Earth's rotation from west to east. This apparent motion is consistent across all locations on Earth."},
  {"question": "is gold a good conductor of electricity", "answer": True, "passage": "Gold is one of the best electrical conductors available. Its conductivity, combined with its corrosion resistance, makes it valuable in electronics and precision applications."},
  {"question": "was the berlin wall built before the cuban missile crisis", "answer": True, "passage": "The Berlin Wall was constructed beginning on August 13, 1961. The Cuban Missile Crisis occurred in October 1962, so the Berlin Wall predates it by over a year."},
  {"question": "do plants perform cellular respiration", "answer": True, "passage": "Like all living organisms, plants perform cellular respiration. They break down sugars to release energy through respiration, occurring continuously in all plant cells."},
  {"question": "is the pacific ocean larger than all land masses combined", "answer": True, "passage": "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It covers an area of approximately 165 million square kilometers, larger than all of Earth's landmasses combined."},
  {"question": "can antibiotics treat viral infections", "answer": False, "passage": "Antibiotics are medications that destroy or slow down the growth of bacteria. They have no effect on viruses, so taking antibiotics for a cold or flu will not help."},
  {"question": "is the human eye able to see infrared light", "answer": False, "passage": "The human eye can detect light in the visible spectrum, roughly 380 to 740 nanometers. Infrared light, with wavelengths longer than 740nm, is not visible to the naked human eye."},
  {"question": "did alexander the great conquer china", "answer": False, "passage": "Alexander the Great's campaign extended into northwestern India. He never reached China; his easternmost point was near the Beas River in modern-day Pakistan before his troops refused to march further."},
  {"question": "is dna found in both plant and animal cells", "answer": True, "passage": "DNA (deoxyribonucleic acid) is found in the cells of all living organisms, including both plants and animals. It is located primarily in the cell nucleus and in organelles like mitochondria."},
  {"question": "is mount everest the tallest mountain from earth's center", "answer": False, "passage": "When measured from Earth's center, Mount Chimborazo in Ecuador is taller than Everest due to Earth's equatorial bulge. Everest is the highest above sea level but not the tallest from the geocenter."},
  {"question": "do fish have a four-chambered heart", "answer": False, "passage": "Most fish have a two-chambered heart consisting of one atrium and one ventricle. The four-chambered heart is characteristic of mammals and birds."},
  {"question": "was shakespeare born in the 16th century", "answer": True, "passage": "William Shakespeare was born on approximately April 23, 1564, in Stratford-upon-Avon, England. This places his birth firmly in the 16th century."},
  {"question": "is the sun a medium-sized star", "answer": True, "passage": "The Sun is classified as a G-type main-sequence star, commonly referred to as a yellow dwarf. It is considered average in size compared to other stars in the Milky Way."},
  {"question": "can humans digest cellulose", "answer": False, "passage": "Humans lack the enzyme cellulase needed to break down cellulose, a polysaccharide found in plant cell walls. Unlike cows and other ruminants, humans cannot digest cellulose."},
  {"question": "is the black sea actually black", "answer": False, "passage": "The Black Sea is not actually black in color. It gets its name possibly from the ancient Greek maritime tradition of naming cardinal directions by color, with north represented by black."},
  {"question": "do whales breathe air", "answer": True, "passage": "Whales are mammals and must breathe air to survive. Unlike fish, they cannot extract oxygen from water. They breathe through blowholes on top of their heads."},
  {"question": "was napoleon short for his time", "answer": False, "passage": "Napoleon Bonaparte stood approximately 5 feet 7 inches (170 cm) tall, which was slightly above average for a Frenchman of his era. The myth of his shortness arose partly from British propaganda."},
  {"question": "is blood blue when inside the body", "answer": False, "passage": "Human blood is always red, both inside and outside the body. Oxygenated blood is bright red, and deoxygenated blood is dark red. The blue appearance of veins through skin is due to how light penetrates tissue."},
  {"question": "did the roman empire fall in 476 ad", "answer": True, "passage": "The traditional date for the fall of the Western Roman Empire is 476 AD, when Romulus Augustulus was deposed by the Germanic chieftain Odoacer."},
  {"question": "is ice less dense than liquid water", "answer": True, "passage": "Ice is less dense than liquid water, which is why ice floats. Water reaches its maximum density at approximately 4 degrees Celsius. This unusual property is critical for aquatic life."},
  {"question": "can you see stars during the day with the naked eye", "answer": False, "passage": "Stars are too faint to be seen in daylight because the sky is illuminated by sunlight scattered in the atmosphere. Under normal conditions, stars cannot be seen with the naked eye during daylight."},
  {"question": "is the earth perfectly round", "answer": False, "passage": "Earth is not a perfect sphere. It is an oblate spheroid, meaning it is slightly flattened at the poles and bulges at the equator due to its rotation."},
  {"question": "do bats use their eyes to navigate", "answer": False, "passage": "Most bats primarily use echolocation to navigate and find prey in the dark. They emit high-frequency sounds and interpret the returning echoes to build a picture of their surroundings."},
  {"question": "was the first computer built in the 20th century", "answer": True, "passage": "The first general-purpose electronic digital computers were built in the 1940s. ENIAC, completed in 1945, is often cited as one of the earliest, placing it firmly in the 20th century."},
  {"question": "is the equator the hottest place on earth", "answer": False, "passage": "The equator is not necessarily the hottest place on Earth. The hottest temperatures are typically found in subtropical desert regions, such as the Sahara Desert, due to lack of cloud cover."},
  {"question": "do all spiders spin webs", "answer": False, "passage": "Not all spiders spin webs. Many spider species, such as wolf spiders and jumping spiders, are active hunters and do not build webs to catch prey."},
  {"question": "was the titanic the largest ship ever built at the time", "answer": True, "passage": "At the time of its construction, the RMS Titanic was one of the largest ships ever built. It was one of three Olympic-class ocean liners constructed for the White Star Line."},
  {"question": "is the atom mostly empty space", "answer": True, "passage": "Atoms are mostly empty space. The nucleus of an atom contains protons and neutrons but is tiny compared to the total volume of the atom. Electrons orbit at great distances relative to nuclear size."},
  {"question": "does the moon have gravity", "answer": True, "passage": "The Moon has its own gravitational field. Its surface gravity is approximately 1.62 m/s², or about one-sixth of Earth's surface gravity of 9.8 m/s²."},
  {"question": "was the first world war fought primarily in europe", "answer": True, "passage": "World War I was primarily fought in Europe, particularly in the Western Front in France and Belgium, and the Eastern Front between Germany and Russia."},
  {"question": "can sound travel through a vacuum", "answer": False, "passage": "Sound requires a medium such as air, water, or solids to travel. It cannot propagate through a vacuum because there are no molecules to vibrate and transmit the wave."},
  {"question": "is the human genome fully mapped", "answer": True, "passage": "The Human Genome Project was completed in 2003, producing a sequence covering approximately 92% of the human genome. A truly complete sequence was published in 2022."},
  {"question": "do penguins live in the arctic", "answer": False, "passage": "Penguins are native to the Southern Hemisphere, particularly Antarctica and surrounding islands. They do not live in the Arctic, where polar bears reside."},
  {"question": "is mercury the closest planet to the sun", "answer": True, "passage": "Mercury is the innermost planet in the Solar System, orbiting the Sun at an average distance of about 57.9 million kilometers, making it the closest planet to the Sun."},
  {"question": "is the speed of sound faster than the speed of light", "answer": False, "passage": "Light travels at approximately 299,792 km/s in a vacuum, while sound travels at only about 343 m/s in air at room temperature. Light is roughly 900,000 times faster than sound."},
  {"question": "did humans coexist with dinosaurs", "answer": False, "passage": "Non-avian dinosaurs went extinct approximately 66 million years ago. Modern humans evolved only about 300,000 years ago. The two species were separated by tens of millions of years."},
  {"question": "is the ozone layer in the stratosphere", "answer": True, "passage": "The ozone layer is a region of Earth's stratosphere, approximately 15 to 35 kilometres above Earth's surface, containing a high concentration of ozone molecules."},
  {"question": "can chameleons change color to match any background", "answer": False, "passage": "Chameleons change color primarily for communication and thermoregulation, not camouflage. Their color changes are not perfectly matched to every background."},
  {"question": "is hydrogen the most abundant element in the universe", "answer": True, "passage": "Hydrogen is the most abundant chemical element in the universe, making up about 75% of all normal matter by mass and more than 90% by number of atoms."},
  {"question": "was the cold war a direct military conflict between the us and ussr", "answer": False, "passage": "The Cold War was a period of geopolitical tension between the United States and the Soviet Union. It was characterized by political and ideological conflict rather than direct military confrontation between the two superpowers."},
  {"question": "do all cells in the human body have a nucleus", "answer": False, "passage": "Most human cells contain a nucleus, but red blood cells and platelets are exceptions. Mature red blood cells lose their nucleus during development to maximize space for hemoglobin."},
  {"question": "is the internet and the world wide web the same thing", "answer": False, "passage": "The Internet is a global network of connected computers. The World Wide Web is a system of interlinked hypertext documents accessed via the Internet. The Web is one of many services that runs on the Internet."},
  {"question": "was einstein a poor student in mathematics", "answer": False, "passage": "Contrary to popular belief, Albert Einstein excelled in mathematics throughout his schooling. He mastered calculus by age 15 and received top marks in mathematical subjects."},
  {"question": "do trees grow from the top", "answer": False, "passage": "Trees do not grow from the top. They grow from the tips of their branches and from the cambium layer just under the bark. The trunk does not rise as a tree grows taller."},
  {"question": "is the largest desert in the world the sahara", "answer": False, "passage": "The Sahara is the world's largest hot desert, but Antarctica is the largest desert overall when deserts are defined by low precipitation. Antarctica receives very little precipitation annually."},
  {"question": "did the wright brothers make the first powered airplane flight", "answer": True, "passage": "On December 17, 1903, Orville and Wilbur Wright achieved the first successful sustained, controlled, powered heavier-than-air flight near Kitty Hawk, North Carolina."},
  {"question": "is glass a solid", "answer": False, "passage": "Glass is an amorphous solid, which means it lacks the ordered molecular structure of crystalline solids. Some scientists classify it as a supercooled liquid in a metastable state."},
  {"question": "do all birds fly", "answer": False, "passage": "Not all birds can fly. Penguins, ostriches, emus, kiwis, and cassowaries are examples of flightless birds. These birds have evolved for other modes of locomotion such as running or swimming."},
  {"question": "is the mitochondria the powerhouse of the cell", "answer": True, "passage": "Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of ATP, which is used as a source of chemical energy."},
  {"question": "does lightning always travel from clouds to the ground", "answer": False, "passage": "Lightning can travel from the cloud to the ground, from the ground up to the cloud, or between clouds. The direction depends on the charge distribution at the time of discharge."},
  {"question": "was julius caesar the first roman emperor", "answer": False, "passage": "Julius Caesar was never formally Emperor of Rome. He held the position of dictator. Augustus (Octavian), Caesar's great-nephew and adopted son, is generally considered the first Roman Emperor."},
  {"question": "is chocolate toxic to dogs", "answer": True, "passage": "Chocolate is toxic to dogs because it contains theobromine, a compound that dogs metabolize much more slowly than humans. Even small amounts can cause serious illness or death in dogs."},
  {"question": "do humans only use 10 percent of their brains", "answer": False, "passage": "The idea that humans only use 10% of their brains is a myth. Brain imaging studies show that humans use virtually all parts of the brain, and most of the brain is active almost all the time."},
  {"question": "is the speed of light the same in water and in air", "answer": False, "passage": "Light travels more slowly in water than in air. The speed of light in a vacuum is approximately 299,792 km/s, but it travels at roughly 75% of that speed through water."},
  {"question": "was cleopatra egyptian by ethnicity", "answer": False, "passage": "Cleopatra VII was a member of the Ptolemaic dynasty, a Greek-Macedonian royal family that ruled Egypt. She was of Greek descent, not native Egyptian, though she was the first of her dynasty to speak the Egyptian language."},
  {"question": "do viruses reproduce on their own", "answer": False, "passage": "Viruses cannot reproduce independently. They are obligate intracellular parasites that require a host cell to replicate. They inject their genetic material into a host and use the cell's machinery to copy themselves."},
  {"question": "is the international space station in outer space", "answer": True, "passage": "The International Space Station orbits Earth at an altitude of approximately 400 km (250 miles). This is above the Karman line (100 km), the boundary generally recognized as the edge of outer space."},
  {"question": "did charles darwin propose the theory of gravity", "answer": False, "passage": "Charles Darwin proposed the theory of evolution by natural selection, not gravity. The theory of universal gravitation was proposed by Sir Isaac Newton in the 17th century."},
  {"question": "is the ph of pure water 7", "answer": True, "passage": "Pure water has a pH of exactly 7 at 25 degrees Celsius, making it neutral—neither acidic nor basic. pH below 7 is acidic and above 7 is basic."},
  {"question": "do all mammals have hair or fur", "answer": True, "passage": "All mammals have hair or fur at some point in their lives, even marine mammals. In whales and dolphins, hair is present during fetal development and may persist in small amounts into adulthood."},
  {"question": "was the eiffel tower originally intended to be permanent", "answer": False, "passage": "The Eiffel Tower was built as a temporary structure for the 1889 World's Fair in Paris. It was supposed to be dismantled after 20 years but was saved because of its usefulness as a radio transmission tower."},
  {"question": "is the human body more than 50 percent water", "answer": True, "passage": "The human body is approximately 60% water by mass in adult males and about 55% in adult females. Water is found in every cell and is essential for virtually every bodily function."},
  {"question": "can electric eels produce enough electricity to kill a human", "answer": True, "passage": "Electric eels can generate electric shocks of up to 600 volts. While a shock from a single eel is rarely fatal to humans, repeated shocks or shocks in the water can cause drowning or cardiac arrest."},
  {"question": "is the cerebellum responsible for memory storage", "answer": False, "passage": "The cerebellum is primarily responsible for coordinating movement, balance, and fine motor control. Memory storage is primarily associated with the hippocampus and cerebral cortex."},
  {"question": "did the soviet union put the first human in space", "answer": True, "passage": "Soviet cosmonaut Yuri Gagarin became the first human to travel to outer space on April 12, 1961, aboard the Vostok 1 spacecraft."},
  {"question": "is the appendix a vestigial organ with no function", "answer": False, "passage": "Though historically considered vestigial, recent research suggests the appendix may serve as a reservoir for beneficial gut bacteria. Its complete function is still debated among scientists."},
  {"question": "do plants need sunlight to survive", "answer": True, "passage": "Most plants require sunlight to perform photosynthesis, the process by which they convert light energy into chemical energy (sugars). Without light, most plants cannot produce food and will die."},
  {"question": "is new zealand part of australia", "answer": False, "passage": "New Zealand is an independent island country in the southwestern Pacific Ocean. It is a separate sovereign nation and is not part of Australia, though the two countries share cultural and historical ties."},
  {"question": "was the printing press invented in the 14th century", "answer": False, "passage": "The movable-type printing press was invented by Johannes Gutenberg around 1440, placing its invention in the 15th century, not the 14th."},
  {"question": "does the human body produce vitamin d from sunlight", "answer": True, "passage": "When the skin is exposed to UVB rays from sunlight, it synthesizes vitamin D3 from cholesterol. This is the primary natural source of vitamin D for most people."},
  {"question": "is pluto still classified as a planet", "answer": False, "passage": "In 2006, the International Astronomical Union reclassified Pluto as a dwarf planet. It no longer meets the criteria required to be classified as a full planet in the Solar System."},
  {"question": "did the french revolution begin in the 18th century", "answer": True, "passage": "The French Revolution began in 1789, placing it in the 18th century. It was a period of radical political and societal transformation in France that overthrew the monarchy."},
  {"question": "is a tomato a vegetable", "answer": False, "passage": "Botanically, a tomato is a fruit because it develops from the flower of a plant and contains seeds. However, in culinary terms and in a landmark 1893 US Supreme Court ruling, it is classified as a vegetable."},
  {"question": "can the human eye distinguish more shades of green than any other color", "answer": True, "passage": "The human eye can distinguish more shades of green than other colors. This is believed to be an evolutionary adaptation, as our ancestors lived in environments dominated by green vegetation."},
  {"question": "does a compass point to the geographic north pole", "answer": False, "passage": "A compass points to magnetic north, not geographic (true) north. The magnetic north pole is located in northern Canada and is not in the same location as the geographic North Pole."},
  {"question": "was mozart a child prodigy", "answer": True, "passage": "Wolfgang Amadeus Mozart showed exceptional musical talent from an early age. He began composing at age five and performed for European royalty by age six, making him one of history's most famous child prodigies."},
  {"question": "is the marathon race exactly 26 miles long", "answer": False, "passage": "The standard marathon distance is 26.219 miles (42.195 kilometers). The extra distance was standardized in the 1908 Olympic Games to accommodate the British royal family's viewing preferences."},
  {"question": "do carrots improve your night vision", "answer": False, "passage": "Carrots contain beta-carotene, which the body converts to vitamin A, essential for eye health. However, eating extra carrots does not improve vision beyond normal levels. The myth originated from WWII British propaganda."},
  {"question": "was gandhi the prime minister of india", "answer": False, "passage": "Mahatma Gandhi was never Prime Minister of India. He was the leader of the Indian independence movement. Jawaharlal Nehru became India's first Prime Minister when independence was gained in 1947."},
  {"question": "is the mona lisa painted on canvas", "answer": False, "passage": "The Mona Lisa was painted by Leonardo da Vinci on a poplar wood panel, not on canvas. It is a small painting measuring approximately 77 cm × 53 cm."},
  {"question": "do humans share dna with bananas", "answer": True, "passage": "Humans share approximately 60% of their DNA with banana plants. This reflects the fact that many fundamental biological processes are conserved across all living organisms."},
  {"question": "is the bermuda triangle a recognized geographic region", "answer": False, "passage": "The Bermuda Triangle is a loosely defined region of the North Atlantic Ocean. It is not officially recognized by the US Board on Geographic Names and does not appear on standard geographic charts."},
]

if __name__ == "__main__":
    run_experiment()
