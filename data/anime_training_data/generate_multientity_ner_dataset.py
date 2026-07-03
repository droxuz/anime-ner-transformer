import argparse
import json
import random
import re
import shutil
import statistics
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SEED = 321
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|'[A-Za-z]+|[^\w\s]")
MARKER_PATTERN = re.compile(r"<<(TITLE|GENRE|THEME)(\d+)>>")
ENTITY_TYPES = ("TITLE", "GENRE", "THEME")
VALID_LABELS = {
    "O",
    "B-TITLE", "I-TITLE",
    "B-GENRE", "I-GENRE",
    "B-THEME", "I-THEME",
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "even", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

# A genre is a recognizable story/market category. A theme is an idea or question
# explored by the story. Similar concepts are intentionally assigned to only one side.
GENRE_TAXONOMY = [
    {"canonical": "action", "aliases": ["action"]},
    {"canonical": "adventure", "aliases": ["adventure"]},
    {"canonical": "comedy", "aliases": ["comedy"]},
    {"canonical": "drama", "aliases": ["drama"]},
    {"canonical": "fantasy", "aliases": ["fantasy"]},
    {"canonical": "science fiction", "aliases": ["science fiction", "sci-fi"]},
    {"canonical": "romance", "aliases": ["romance"]},
    {"canonical": "mystery", "aliases": ["mystery"]},
    {"canonical": "horror", "aliases": ["horror"]},
    {"canonical": "thriller", "aliases": ["thriller"]},
    {"canonical": "sports", "aliases": ["sports"]},
    {"canonical": "slice of life", "aliases": ["slice of life"]},
    {"canonical": "historical drama", "aliases": ["historical drama", "historical fiction"]},
    {"canonical": "military", "aliases": ["military", "military drama"]},
    {"canonical": "mecha", "aliases": ["mecha", "giant robot"]},
    {"canonical": "music", "aliases": ["music", "musical drama"]},
    {"canonical": "school comedy", "aliases": ["school comedy"]},
    {"canonical": "workplace comedy", "aliases": ["workplace comedy"]},
    {"canonical": "cooking", "aliases": ["cooking", "food-centered comedy"]},
    {"canonical": "detective", "aliases": ["detective", "detective fiction"]},
    {"canonical": "martial arts", "aliases": ["martial arts"]},
    {"canonical": "space opera", "aliases": ["space opera"]},
    {"canonical": "cyberpunk", "aliases": ["cyberpunk"]},
    {"canonical": "isekai", "aliases": ["isekai", "other-world fantasy"]},
    {"canonical": "crime drama", "aliases": ["crime drama", "crime thriller"]},
    {"canonical": "medical drama", "aliases": ["medical drama"]},
    {"canonical": "supernatural", "aliases": ["supernatural", "supernatural drama"]},
    {"canonical": "psychological thriller", "aliases": ["psychological thriller"]},
    {"canonical": "romantic comedy", "aliases": ["romantic comedy", "rom-com"]},
    {"canonical": "dark fantasy", "aliases": ["dark fantasy"]},
    {"canonical": "post-apocalyptic", "aliases": ["post-apocalyptic", "post-apocalyptic fiction"]},
    {"canonical": "samurai", "aliases": ["samurai", "samurai drama"]},
    {"canonical": "magical girl", "aliases": ["magical girl"]},
    {"canonical": "game-centered adventure", "aliases": ["game-centered adventure", "gaming adventure"]},
]

THEME_TAXONOMY = [
    {"canonical": "found family", "aliases": ["found family", "chosen family"]},
    {"canonical": "redemption", "aliases": ["redemption", "earning redemption"]},
    {"canonical": "personal identity", "aliases": ["personal identity", "identity"]},
    {"canonical": "grief and healing", "aliases": ["grief and healing", "coping with grief"]},
    {"canonical": "rivalry", "aliases": ["rivalry", "friendly rivalry"]},
    {"canonical": "political corruption", "aliases": ["political corruption", "corrupt institutions"]},
    {"canonical": "survival under pressure", "aliases": ["survival under pressure", "endurance under pressure"]},
    {"canonical": "growing up", "aliases": ["growing up", "the transition into adulthood"]},
    {"canonical": "friendship", "aliases": ["friendship", "loyalty between friends"]},
    {"canonical": "revenge and consequences", "aliases": ["revenge and consequences", "the cost of revenge"]},
    {"canonical": "moral ambiguity", "aliases": ["moral ambiguity", "uncertain morality"]},
    {"canonical": "ambition", "aliases": ["ambition", "the price of ambition"]},
    {"canonical": "isolation", "aliases": ["isolation", "social isolation"]},
    {"canonical": "responsibility", "aliases": ["responsibility", "accepting responsibility"]},
    {"canonical": "the cost of war", "aliases": ["the cost of war", "war and its consequences"]},
    {"canonical": "humanity and technology", "aliases": ["humanity and technology", "technology's effect on humanity"]},
    {"canonical": "environmental decline", "aliases": ["environmental decline", "ecological collapse"]},
    {"canonical": "family expectations", "aliases": ["family expectations", "pressure from family"]},
    {"canonical": "competition and teamwork", "aliases": ["competition and teamwork", "cooperation under competition"]},
    {"canonical": "travel and discovery", "aliases": ["travel and discovery", "discovery through travel"]},
    {"canonical": "memory", "aliases": ["memory", "the reliability of memory"]},
    {"canonical": "forgiveness", "aliases": ["forgiveness", "learning to forgive"]},
    {"canonical": "social class", "aliases": ["social class", "class inequality"]},
    {"canonical": "justice versus law", "aliases": ["justice versus law", "law and justice"]},
    {"canonical": "sacrifice", "aliases": ["sacrifice", "personal sacrifice"]},
    {"canonical": "leadership", "aliases": ["leadership", "learning to lead"]},
    {"canonical": "fear of failure", "aliases": ["fear of failure", "anxiety about failure"]},
    {"canonical": "trust", "aliases": ["trust", "learning to trust"]},
    {"canonical": "belonging", "aliases": ["belonging", "the search for belonging"]},
    {"canonical": "duty versus freedom", "aliases": ["duty versus freedom", "freedom against duty"]},
    {"canonical": "human connection", "aliases": ["human connection", "forming meaningful connections"]},
    {"canonical": "the meaning of home", "aliases": ["the meaning of home", "what makes a home"]},
    {"canonical": "power and corruption", "aliases": ["power and corruption", "the corrupting effect of power"]},
    {"canonical": "artistic expression", "aliases": ["artistic expression", "creativity and self-expression"]},
    {"canonical": "tradition versus change", "aliases": ["tradition versus change", "tradition and social change"]},
    {"canonical": "second chances", "aliases": ["second chances", "starting over"]},
    {"canonical": "self-acceptance", "aliases": ["self-acceptance", "accepting oneself"]},
    {"canonical": "teamwork", "aliases": ["teamwork", "working as a team"]},
    {"canonical": "obsession", "aliases": ["obsession", "the danger of obsession"]},
    {"canonical": "healing", "aliases": ["healing", "emotional recovery"]},
    {"canonical": "the value of ordinary life", "aliases": ["the value of ordinary life", "meaning in everyday life"]},
    {"canonical": "courage", "aliases": ["courage", "acting despite fear"]},
    {"canonical": "truth and deception", "aliases": ["truth and deception", "lies and hidden truths"]},
    {"canonical": "generational conflict", "aliases": ["generational conflict", "conflict between generations"]},
    {"canonical": "loss of innocence", "aliases": ["loss of innocence", "the end of childhood innocence"]},
    {"canonical": "hope during crisis", "aliases": ["hope during crisis", "maintaining hope in a crisis"]},
    {"canonical": "love and vulnerability", "aliases": ["love and vulnerability", "emotional vulnerability in love"]},
]

# Explicitly documents the distinctions used to prevent label drift.
CONTRAST_PAIRS = [
    {"genre": "romance", "theme": "love and vulnerability", "rule": "Romance names the story category; love and vulnerability names the idea explored."},
    {"genre": "sports", "theme": "competition and teamwork", "rule": "Sports names the activity-centered category; competition and teamwork names the underlying relationship or conflict."},
    {"genre": "military", "theme": "the cost of war", "rule": "Military names the setting/category; the cost of war names the idea examined."},
    {"genre": "mecha", "theme": "humanity and technology", "rule": "Mecha names the robot-centered category; humanity and technology names the philosophical concern."},
    {"genre": "historical drama", "theme": "tradition versus change", "rule": "Historical drama names the period-based category; tradition versus change names the central tension."},
    {"genre": "slice of life", "theme": "the value of ordinary life", "rule": "Slice of life names the narrative mode; the value of ordinary life names what the narrative may express."},
    {"genre": "crime drama", "theme": "justice versus law", "rule": "Crime drama names the category; justice versus law names the moral question."},
    {"genre": "psychological thriller", "theme": "personal identity", "rule": "Psychological thriller names the suspense category; personal identity names the subject explored."},
    {"genre": "dark fantasy", "theme": "power and corruption", "rule": "Dark fantasy names the category and tone; power and corruption names the idea."},
    {"genre": "isekai", "theme": "belonging", "rule": "Isekai names transport to another world; belonging names the character concern."},
]

TONES = [
    "quiet", "hopeful", "melancholic", "tense", "playful", "reflective", "bittersweet", "energetic",
    "restrained", "optimistic", "unsettling", "warm", "serious", "absurd", "mysterious", "character-driven",
    "fast-moving", "slow-burning", "emotionally grounded", "thoughtful", "darkly comic", "earnest", "atmospheric",
    "introspective", "suspenseful", "gentle", "bleak", "uplifting"
]
SETTINGS = [
    "a coastal town", "a crowded city", "a remote village", "a damaged space station", "a competitive school",
    "a fantasy kingdom", "a near-future workplace", "a quiet neighborhood", "a war-torn country", "an isolated island",
    "a traveling theatre", "a research laboratory", "a mountain community", "an underground society", "a sports club",
    "a family restaurant", "a detective agency", "a ruined megacity", "a rural farm", "a long-distance train journey",
    "an orbital colony", "a historic capital", "a small music venue", "a hospital", "an unfamiliar parallel world"
]
TRAITS = [
    "flawed but capable", "patient under pressure", "socially awkward", "morally conflicted", "quietly ambitious",
    "protective of others", "willing to admit mistakes", "clever without being perfect", "funny without becoming a joke",
    "driven by responsibility", "uncertain about the future", "slow to trust", "competitive but fair", "openly compassionate",
    "haunted by an earlier choice", "learning how to lead", "trying to rebuild a relationship", "curious and observant"
]
AVOIDS = [
    "repetitive exposition", "forced romance", "constant fan service", "shallow villains", "an endless tournament arc",
    "a rushed ending", "too many flashbacks", "predictable twists", "one-note side characters", "needless cruelty",
    "excessive narration", "a confusing opening", "unresolved subplots", "comedy that interrupts serious scenes",
    "power scaling without consequences", "a passive protagonist", "long recaps", "artificial misunderstandings"
]

# 80% of examples come from these clear entity contexts.
CLEAR_TEMPLATES = [
    "I finished <<TITLE1>> last night, and now I want a <<GENRE1>> series that explores <<THEME1>> without relying on {avoid}.",
    "After watching <<TITLE1>>, I realized that <<GENRE1>> works best for me when the story develops <<THEME1>> patiently.",
    "Could you recommend something like <<TITLE1>>, but with stronger <<GENRE1>> elements and more attention to <<THEME1>>?",
    "The part of <<TITLE1>> that stayed with me was <<THEME1>>; its use of <<GENRE1>> also kept the plot moving.",
    "<<TITLE1>> was interesting. My next show should combine <<GENRE1>> with <<THEME1>>, although it can use a different setting.",
    "When my friend suggested <<TITLE1>>, I expected ordinary <<GENRE1>>, but its treatment of <<THEME1>> surprised me.",
    "I have not seen much <<GENRE1>>, so should I begin with <<TITLE1>> if I also care about <<THEME1>>?",
    "My reference point is <<TITLE1>>: I want the same {tone} pacing, a clear <<GENRE1>> identity, and a focus on <<THEME1>>.",
    "Although <<TITLE1>> contains some {avoid}, I still liked the way its <<GENRE1>> plot handled <<THEME1>>.",
    "Before class, I watched <<TITLE1>>; afterward, I kept thinking about <<THEME1>> rather than the usual <<GENRE1>> spectacle.",
    "If I liked <<TITLE1>>, what should I watch next when I want <<GENRE1>>, <<THEME1>>, and characters who are {trait}?",
    "Please do not copy <<TITLE1>> exactly. I only want <<GENRE1>> with a similarly {tone} approach to <<THEME1>>.",
    "Is <<TITLE1>> mainly <<GENRE1>>, or does it spend more time exploring <<THEME1>> and character relationships?",
    "The title on my list is \"<<TITLE1>>,\" because several people said it balances <<GENRE1>> with <<THEME1>>.",
    "One show keeps appearing in recommendations: <<TITLE1>>. Does it fit someone who prefers <<GENRE1>> stories about <<THEME1>>?",
    "<<TITLE1>>—from what I have heard—mixes <<GENRE1>> with <<THEME1>>; that combination sounds promising.",
    "I may watch <<TITLE1>> this weekend, provided its <<GENRE1>> plot gives <<THEME1>> enough room to develop.",
    "For a short break, would <<TITLE1>> work, or is its <<GENRE1>> plot too dense despite its focus on <<THEME1>>?",
    "The last anime I completed was <<TITLE1>>; next, I want {tone} <<GENRE1>> where <<THEME1>> drives the conflict.",
    "I liked <<TITLE1>> for several reasons: the <<THEME1>>, the {tone} direction, and the <<GENRE1>> framework all helped.",
    "Even though I usually avoid {avoid}, <<TITLE1>> held my attention because its <<GENRE1>> story treated <<THEME1>> seriously.",
    "Someone described <<TITLE1>> as <<GENRE1>> with an emphasis on <<THEME1>>. Is that accurate?",
    "At first I ignored <<TITLE1>>, but after hearing that this <<GENRE1>> series explores <<THEME1>>, I added it to my list.",
    "What does <<TITLE1>> do with <<THEME1>>? I am trying to find <<GENRE1>> that does not simplify that subject.",
    "My ideal recommendation would not imitate <<TITLE1>>; it would simply combine <<GENRE1>> and <<THEME1>> with a {tone} mood.",
    "I watched <<TITLE1>>, took a break, and later decided that <<THEME1>> was more memorable than the <<GENRE1>> plot.",
    "Between homework and work, I have little free time. Is <<TITLE1>> worth starting if I enjoy <<GENRE1>> and <<THEME1>>?",
    "A recommendation does not need to resemble <<TITLE1>>, but its <<GENRE1>> story should give <<THEME1>> time to develop.",
    "Would you call <<TITLE1>> a strong example of <<GENRE1>>, especially for a viewer who cares about <<THEME1>>?",
    "I wrote down <<TITLE1>> in my notes beside the phrases <<GENRE1>> and <<THEME1>> so I would remember why it interested me.",
    "Because <<TITLE1>> ended well, I am now searching for another <<GENRE1>> series with a satisfying treatment of <<THEME1>>.",
    "No spoilers, please: does <<TITLE1>> eventually focus on <<THEME1>>, or does it remain mostly <<GENRE1>>?",
    "I know almost nothing about <<TITLE1>>. I only need to know whether its <<GENRE1>> approach to <<THEME1>> is consistent.",
    "For tonight, I need something easy to begin but not shallow; perhaps <<TITLE1>> fits because it joins <<GENRE1>> with <<THEME1>>.",
    "My friend called <<TITLE1>> a {tone} <<GENRE1>> story about <<THEME1>>, which immediately caught my attention.",
    "If <<TITLE1>> had less {avoid}, it would be close to my ideal: <<GENRE1>> built around <<THEME1>>.",
    "I am comparing my preferences with <<TITLE1>>: keep <<THEME1>>, change the setting, and add more <<GENRE1>>.",
    "The name <<TITLE1>> appeared in a discussion of <<THEME1>>. Is the show actually <<GENRE1>>, or was that comparison misleading?",
    "I want <<GENRE1>> that examines <<THEME1>> in {setting}; <<TITLE1>> is the closest example I can name.",
    "I am not asking for another <<TITLE1>>. I am asking for {tone} <<GENRE1>> whose central concern is <<THEME1>>.",

    "I enjoyed <<TITLE1>>, but <<TITLE2>> handled <<THEME1>> better; now I want <<GENRE1>> that combines their strengths.",
    "Between <<TITLE1>> and <<TITLE2>>, which is the stronger introduction to <<GENRE1>> and <<THEME1>>?",
    "<<TITLE1>> gave me the <<GENRE1>> energy I wanted, whereas <<TITLE2>> made <<THEME1>> more emotionally convincing.",
    "I watched <<TITLE1>> first and <<TITLE2>> later. Both use <<GENRE1>>, but they approach <<THEME1>> differently.",
    "My ideal recommendation would borrow the pacing of <<TITLE1>>, the characters of <<TITLE2>>, and the <<THEME1>> focus of good <<GENRE1>>.",
    "Do <<TITLE1>> and <<TITLE2>> belong to the same kind of <<GENRE1>>, or do they merely share <<THEME1>>?",
    "I mentioned <<TITLE1>> at lunch, and someone suggested <<TITLE2>> because both connect <<GENRE1>> with <<THEME1>>.",
    "Although <<TITLE1>> is shorter, <<TITLE2>> may suit me better because I care more about <<THEME1>> than typical <<GENRE1>> conventions.",
    "I have questions about <<TITLE1>> and <<TITLE2>>: are they accessible, are they truly <<GENRE1>>, and do they develop <<THEME1>>?",
    "First I saw <<TITLE1>>; then I found <<TITLE2>>. Together, they changed what I expect from <<GENRE1>> stories about <<THEME1>>.",
    "Neither <<TITLE1>> nor <<TITLE2>> needs to be perfect. I mainly want to know which one develops <<THEME1>> more naturally within <<GENRE1>>.",
    "My notes say, \"Watch <<TITLE1>>, then try <<TITLE2>> for <<GENRE1>> and <<THEME1>>.\" I no longer remember who recommended them.",
    "I would rewatch <<TITLE1>>, but I would rather discuss <<TITLE2>> because its <<GENRE1>> structure makes <<THEME1>> harder to resolve.",
    "The jump from <<TITLE1>> to <<TITLE2>> was strange, yet both showed how flexible <<GENRE1>> can be when it addresses <<THEME1>>.",
    "If <<TITLE1>> is the obvious <<GENRE1>> recommendation, would <<TITLE2>> be a less predictable choice for someone interested in <<THEME1>>?",

    "I want a {tone} <<GENRE1>> anime in which <<THEME1>> develops slowly and the characters are {trait}.",
    "Could you suggest <<GENRE1>> set in {setting}, with a focus on <<THEME1>>, that does not rely on {avoid}?",
    "The show can be old or new, but it should clearly be <<GENRE1>> and should treat <<THEME1>> as more than background decoration.",
    "I am looking for <<GENRE1>> that my friends and I can discuss after each episode, especially because of <<THEME1>>.",
    "Please recommend <<GENRE1>> with characters who are {trait}; the main conflict should arise from <<THEME1>>.",
    "What should I watch when I want a {tone} mood, a <<GENRE1>> setting, and a story centered on <<THEME1>>?",
    "Can you find <<GENRE1>> about <<THEME1>> that takes place in {setting} and avoids turning every conflict into a fight?",
    "A little romance is acceptable, but I am primarily asking for <<GENRE1>> in which <<THEME1>> shapes the characters' choices.",
    "I am interested in <<GENRE1>>, although I care more about <<THEME1>> and believable consequences than spectacle.",
    "I need a recommendation for someone who rarely watches anime but enjoys <<GENRE1>> and stories about <<THEME1>>.",
    "What is a good entry point for <<GENRE1>> if I care about <<THEME1>>, clear motivation, and a satisfying conclusion?",
    "Please find a {tone} <<GENRE1>> series where <<THEME1>> appears through choices rather than long speeches.",

    "I want <<GENRE1>> with some <<GENRE2>>, but the story should remain focused on <<THEME1>>.",
    "Could a series combine <<GENRE1>> and <<GENRE2>> without losing its interest in <<THEME1>>?",
    "My preferred mix is <<GENRE1>> for momentum, <<GENRE2>> for variety, and <<THEME1>> for emotional depth.",
    "I am searching for something between <<GENRE1>> and <<GENRE2>>; ideally, <<THEME1>> would connect the two styles.",
    "The recommendation can begin as <<GENRE1>> and gradually become <<GENRE2>>, provided <<THEME1>> remains coherent.",
    "Would <<TITLE1>> fit a viewer who likes both <<GENRE1>> and <<GENRE2>>, especially when <<THEME1>> is central?",
    "<<TITLE1>> looks like <<GENRE1>>, but I have heard that it also contains <<GENRE2>> and explores <<THEME1>>.",
    "After <<TITLE1>>, I want a blend of <<GENRE1>> and <<GENRE2>> that treats <<THEME1>> seriously.",

    "I want <<GENRE1>> that explores both <<THEME1>> and <<THEME2>> without reducing either idea to a speech.",
    "Could you recommend <<GENRE1>> where <<THEME1>> creates the external conflict and <<THEME2>> shapes the character arc?",
    "The category can remain <<GENRE1>>, but I would like the story to connect <<THEME1>> with <<THEME2>>.",
    "<<TITLE1>> made me interested in <<THEME1>>; now I want <<GENRE1>> that also examines <<THEME2>>.",
    "I liked the way <<TITLE1>> approached <<THEME1>>, although its <<GENRE1>> plot could have done more with <<THEME2>>.",
]

# 20% challenge cases: semantic contrasts, multiple entities, entity-like words used
# in ordinary senses, punctuation boundaries, and close genre/theme pairs.
CHALLENGE_TEMPLATES = [
    "I am not asking for <<GENRE1>> merely because it contains fights; I want the actual category, and I want <<THEME1>> to matter after the action ends.",
    "Do not confuse <<GENRE1>> with <<THEME1>>: one describes the kind of story, while the other describes what the story examines.",
    "People call <<TITLE1>> <<GENRE1>>, yet the reason I remember it is <<THEME1>>, not the category label.",
    "The review listed <<GENRE1>>, <<GENRE2>>, and <<THEME1>> in one sentence; which of those words describes the genre, and which describes the idea?",
    "I want <<GENRE1>>—not just a story containing a brief genre-like scene—with a sustained interest in <<THEME1>>.",
    "<<TITLE1>> begins like <<GENRE1>>; after the comma, however, its discussion of <<THEME1>> becomes more important.",
    "Is <<TITLE1>> really <<GENRE1>>? The trailer emphasizes <<THEME1>>, but trailers often hide the actual structure.",
    "My request has two separate parts: use <<GENRE1>> as the category, and use <<THEME1>> as the underlying concern.",
    "I enjoy <<GENRE1>>, but I do not automatically enjoy every story about <<THEME1>>; the recommendation needs both.",
    "A show may discuss <<THEME1>> without belonging to <<GENRE1>>, so please check the category rather than matching one keyword.",
    "The phrase <<THEME1>> describes what I want the characters to confront; <<GENRE1>> describes how I want the story organized.",
    "<<TITLE1>>, <<GENRE1>>, <<THEME1>>—those are three different kinds of information, even though they appear beside one another in my notes.",
    "I want <<GENRE1>>; specifically, I want it to explore <<THEME1>>. I do not want the model to merge those spans.",
    "In the sentence \"<<TITLE1>> is <<GENRE1>> about <<THEME1>>,\" the title, category, and idea should remain distinct.",
    "Would you label <<GENRE1>> as a genre and <<THEME1>> as a theme even when both phrases occur after the same colon: <<TITLE1>>?",
    "I compared <<TITLE1>> with <<TITLE2>>—both are <<GENRE1>>—but only one develops <<THEME1>> beyond a simple subplot.",
    "My list reads: <<TITLE1>>; <<GENRE1>>; <<THEME1>>. The semicolons separate the title from the two preference types.",
    "The recommendation should contain <<GENRE1>>, not merely mention it, and should develop <<THEME1>>, not merely name it.",
    "Even if <<TITLE1>> is marketed as <<GENRE1>>, I would skip it if <<THEME1>> disappears after the opening episodes.",
    "I want neither random spectacle nor a lecture: give me <<GENRE1>> whose characters reveal <<THEME1>> through decisions.",
]

# Ordinary uses of words that overlap the taxonomy. These are all O labels.
HARD_NEGATIVE_PROMPTS = [
    "The action taken by the committee was recorded in the meeting notes before anyone left the room.",
    "Their adventure through the airport ended when the missing suitcase appeared beside the information desk.",
    "The comedy of errors at work was frustrating for everyone who had to repair the schedule afterward.",
    "The family drama at dinner had nothing to do with television, animation, or a fictional series.",
    "Her fantasy football team lost because two players were injured during the same weekend.",
    "The science fiction section of the library was moved upstairs during the renovation.",
    "Their romance began in college, long before either person became interested in anime.",
    "The mystery of the missing keys was solved when we checked the pocket inside the winter coat.",
    "The horror on his face came from seeing the repair bill, not from watching a frightening show.",
    "The thriller novel was still on the table, but nobody had opened it during the study session.",
    "The sports equipment belongs in the storage room after the practice ends.",
    "She sliced the bread and continued with the rest of her ordinary life before work.",
    "The historical drama between the two departments was described in the annual report.",
    "Military personnel closed the road while the damaged bridge was inspected.",
    "The giant robot toy was placed on the highest shelf so the dog could not reach it.",
    "Music from the apartment next door continued until the building manager called.",
    "The school comedy performance raised money for new library computers.",
    "The workplace comedy was an accidental misunderstanding during a real office meeting, not a show title.",
    "Cooking dinner took longer because the oven needed to be cleaned first.",
    "The detective assigned to the case asked for a copy of every receipt.",
    "Martial arts classes are held in the community center on Tuesday evenings.",
    "The term space opera appeared in the essay as an example of genre vocabulary.",
    "Cyberpunk was written on the whiteboard during a discussion of visual design trends.",
    "The word isekai was misspelled in the student's notes and corrected before submission.",
    "The crime drama inside the office concerned missing funds and an argument between managers.",
    "The medical drama in the waiting room ended when the nurse explained that the results were normal.",
    "A supernatural explanation was unnecessary because the strange sound came from a loose window.",
    "The phrase psychological thriller was part of the assignment question, not a request for a recommendation.",
    "Their romantic comedy of mistakes happened during a real date and was retold at breakfast.",
    "The dark fantasy artwork was printed on a notebook cover sold near the register.",
    "The post-apocalyptic costume used cardboard, paint, and parts from an old bicycle.",
    "The samurai display at the museum included armor, documents, and a carefully restored sword.",
    "The magical girl costume was folded into a box after the convention ended.",
    "Our family found the missing photograph behind a cabinet while cleaning the living room.",
    "The employee earned redemption in the manager's opinion by fixing the mistake and explaining what happened.",
    "Personal identity information must be removed from the form before it is shared publicly.",
    "Grief and healing were headings in the counselor's workshop schedule rather than tags for entertainment.",
    "Friendly rivalry between the two local teams increased ticket sales for the final match.",
    "Political corruption was mentioned in the newspaper article that the class analyzed.",
    "Survival under pressure is a skill discussed in the emergency training manual.",
    "Growing up near the lake taught her how quickly the weather could change.",
    "Friendship takes effort when people move to different cities and maintain busy schedules.",
    "The cost of revenge was the title of a lecture, not the title of an anime.",
    "Moral ambiguity was written in the margin as a possible topic for the essay.",
    "His ambition was to finish the degree while continuing to work on weekends.",
    "Isolation of the damaged circuit prevented the failure from spreading to the other components.",
    "Responsibility for the equipment passes to the next shift at six in the evening.",
    "The cost of war was discussed in a history class using letters and government records.",
    "Humanity and technology appeared as the conference theme on the event poster.",
    "Environmental decline affected the lake over several decades according to the research report.",
    "Family expectations changed after everyone discussed the plan honestly.",
    "Competition and teamwork are both assessed during the group exercise.",
    "Travel and discovery were listed as goals in the museum's new educational program.",
    "Memory usage increased when the program loaded the entire file instead of reading it in chunks.",
    "Forgiveness was difficult, but the two friends eventually agreed to speak again.",
    "Social class was one variable in the research paper's analysis.",
    "Justice versus law was the debate topic assigned to the second group.",
    "Personal sacrifice was not required; the schedule could be changed for everyone.",
    "Leadership training begins next month and includes several practical exercises.",
    "Fear of failure made the exam feel more difficult than the questions actually were.",
    "Trust is important when several people share access to the same project files.",
    "The search for belonging was a phrase in her journal, not a media category.",
    "Duty versus freedom was printed on the slide as the topic of the philosophy seminar.",
    "Human connection improved after the group stopped using phones during dinner.",
    "The meaning of home changed after the family moved three times in five years.",
    "Power and corruption were discussed during the city council meeting.",
    "Artistic expression is encouraged in the community program for younger students.",
    "Tradition versus change became the central question in the local planning debate.",
    "Second chances are useful when a mistake can be corrected without causing harm.",
    "Self-acceptance took time and support from several close friends.",
    "Teamwork reduced the amount of time needed to move the furniture.",
    "Obsession with minor formatting details delayed the report for another day.",
    "Healing of the cut took longer because the bandage was removed too early.",
    "The value of ordinary life was discussed in the speaker's closing remarks.",
    "Courage helped her ask the question even though the room was full.",
    "Truth and deception were separate categories in the classroom exercise.",
    "Generational conflict became obvious when the family discussed how to run the business.",
    "Loss of innocence was one possible interpretation of the novel assigned in class.",
    "Hope during crisis was the slogan printed on the charity's information sheet.",
    "Love and vulnerability require honest communication outside fictional stories as well.",
    "I need one piece of paper for the printer, and the rest can stay in the drawer.",
    "The black cat slept under the table while everyone else was eating dinner.",
    "Please bleach the white towels, but do not mix them with the dark clothes.",
    "The air in the room felt heavy because the windows had been closed all afternoon.",
    "Another question came up after class, so I wrote it at the bottom of the page.",
    "The monster in the costume looked friendly once the performer removed the mask.",
]

FAMOUS_TRAIN_IDS = {
    1, 19, 20, 21, 30, 33, 43, 47, 57, 68, 101, 120, 121, 136, 164,
    199, 205, 223, 226, 232, 245, 249, 269, 2904, 5114, 9253, 11061, 16498,
    30276, 31964, 38000, 40748
}


def tokenize_with_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def normalize_alias(alias):
    return unicodedata.normalize("NFKC", alias).strip()


def valid_title_alias(alias):
    alias = normalize_alias(alias)
    if not alias or not alias.isascii() or "�" in alias or "\x90" in alias:
        return False
    tokens = TOKEN_PATTERN.findall(alias)
    if not tokens or len(tokens) > 16:
        return False
    return any(re.search(r"[A-Za-z0-9]", token) for token in tokens)


def load_title_entries(path):
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    entries = []
    seen_ids = set()
    for anime in raw:
        mal_id = anime.get("mal_id")
        if mal_id is None or mal_id in seen_ids:
            continue
        aliases = []
        seen_aliases = set()
        for alias in anime.get("aliases", []):
            alias = normalize_alias(alias)
            if not valid_title_alias(alias):
                continue
            key = alias.casefold()
            if key in seen_aliases:
                continue
            seen_aliases.add(key)
            aliases.append(alias)
        if not aliases:
            continue
        aliases.sort(key=lambda a: (len(TOKEN_PATTERN.findall(a)), len(a)))
        entries.append({
            "mal_id": int(mal_id),
            "canonical_title": anime.get("canonical_title") or aliases[0],
            "aliases": aliases[:4],
            "length": len(TOKEN_PATTERN.findall(aliases[0])),
        })
        seen_ids.add(mal_id)
    return entries


def bucket_name(length):
    if length == 1:
        return "1"
    if length <= 3:
        return "2-3"
    if length <= 6:
        return "4-6"
    if length <= 9:
        return "7-9"
    return "10+"


def stratified_select(entries, total, rng, pinned_ids=None):
    pinned_ids = set(pinned_ids or [])
    by_id = {entry["mal_id"]: entry for entry in entries}
    selected = [by_id[mid] for mid in pinned_ids if mid in by_id]
    selected_ids = {e["mal_id"] for e in selected}
    quotas = {"1": round(total * 0.18), "2-3": round(total * 0.42), "4-6": round(total * 0.27), "7-9": round(total * 0.09)}
    quotas["10+"] = total - sum(quotas.values())
    buckets = defaultdict(list)
    for entry in entries:
        if entry["mal_id"] not in selected_ids:
            buckets[bucket_name(entry["length"])].append(entry)
    for values in buckets.values():
        rng.shuffle(values)
    current = Counter(bucket_name(e["length"]) for e in selected)
    for name, quota in quotas.items():
        need = max(0, quota - current[name])
        chosen = buckets[name][:need]
        selected.extend(chosen)
        selected_ids.update(e["mal_id"] for e in chosen)
    if len(selected) < total:
        leftovers = [e for e in entries if e["mal_id"] not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    rng.shuffle(selected)
    return selected[:total]


def split_title_pools(entries, rng):
    # 80/10/10 title pool split, matching the example split.
    train_size = 4000
    val_size = 500
    test_size = 500

    train = stratified_select(entries, train_size, rng, pinned_ids=FAMOUS_TRAIN_IDS)
    used = {e["mal_id"] for e in train}

    remaining = [e for e in entries if e["mal_id"] not in used]
    val = stratified_select(remaining, val_size, rng)
    used.update(e["mal_id"] for e in val)

    remaining = [e for e in entries if e["mal_id"] not in used]
    test = stratified_select(remaining, test_size, rng)

    return {"train": train, "val": val, "test": test}


def variant(entry, rng):
    alias = rng.choice(entry["aliases"])
    roll = rng.random()
    if roll < 0.08:
        return alias.lower()
    if roll < 0.11 and sum(ch.isalpha() for ch in alias) <= 28:
        return alias.upper()
    return alias


def concept_variant(entry, rng):
    return rng.choice(entry["aliases"])


def fill_context(text, rng):
    text = text.format(
        tone=rng.choice(TONES), setting=rng.choice(SETTINGS), trait=rng.choice(TRAITS), avoid=rng.choice(AVOIDS)
    )
    return re.sub(r"\ba (?=[aeiouAEIOU])", "an ", text)


def render_template(template, values):
    spans = []
    cursor = 0
    out = []
    out_len = 0
    for match in MARKER_PATTERN.finditer(template):
        prefix = template[cursor:match.start()]
        out.append(prefix)
        out_len += len(prefix)
        entity_type = match.group(1)
        index = int(match.group(2)) - 1
        value = values[entity_type][index]
        start = out_len
        out.append(value)
        out_len += len(value)
        spans.append((start, out_len, entity_type, index))
        cursor = match.end()
    out.append(template[cursor:])
    text = "".join(out)
    token_data = tokenize_with_spans(text)
    tokens = [token for token, _, _ in token_data]
    labels = ["O"] * len(tokens)
    span_metadata = []
    occupied = set()
    for start_char, end_char, entity_type, value_index in spans:
        indices = [i for i, (_, start, end) in enumerate(token_data) if start >= start_char and end <= end_char]
        if not indices:
            raise ValueError(f"Entity produced no tokens: {entity_type} {values[entity_type][value_index]!r}")
        if occupied.intersection(indices):
            raise ValueError("Overlapping marker spans")
        occupied.update(indices)
        labels[indices[0]] = f"B-{entity_type}"
        for i in indices[1:]:
            labels[i] = f"I-{entity_type}"
        span_metadata.append({
            "entity": entity_type,
            "text": values[entity_type][value_index],
            "start_token": indices[0],
            "end_token": indices[-1] + 1,
        })
    return {"prompt": text, "tokens": tokens, "labels": labels}, span_metadata


def build_o_example(text):
    tokens = TOKEN_PATTERN.findall(text)
    return {"prompt": text, "tokens": tokens, "labels": ["O"] * len(tokens)}, []


def distinct_choices(pool, count, rng):
    if count == 0:
        return []
    return rng.sample(pool, count)


def marker_counts(template):
    counts = Counter()
    for entity, index in MARKER_PATTERN.findall(template):
        counts[entity] = max(counts[entity], int(index))
    return counts


def make_marked_example(template, title_pool, rng, family, difficulty):
    counts = marker_counts(template)
    title_entries = distinct_choices(title_pool, counts["TITLE"], rng)
    genre_entries = distinct_choices(GENRE_TAXONOMY, counts["GENRE"], rng)
    theme_entries = distinct_choices(THEME_TAXONOMY, counts["THEME"], rng)
    values = {
        "TITLE": [variant(e, rng) for e in title_entries],
        "GENRE": [concept_variant(e, rng) for e in genre_entries],
        "THEME": [concept_variant(e, rng) for e in theme_entries],
    }
    template = fill_context(template, rng)
    example, spans = render_template(template, values)
    meta = {
        "family": family,
        "difficulty": difficulty,
        "mal_ids": [e["mal_id"] for e in title_entries],
        "genre_canonicals": [e["canonical"] for e in genre_entries],
        "theme_canonicals": [e["canonical"] for e in theme_entries],
        "spans": spans,
    }
    return example, meta


NEGATIVE_WRAPPERS = [
    "{text}",
    "For context, {text}",
    "In an unrelated conversation, {text}",
    "While reviewing my notes, I wrote: {text}",
    "This sentence is not an anime request: {text}",
    "Outside a media discussion, {text}",
    "During class, someone observed that {text_lower}",
    "The example on the worksheet said, \"{text}\"",
    "Before the meeting ended, {text_lower}",
    "As a plain-language example, {text_lower}",
    "The paragraph continued: {text}",
    "In ordinary conversation, {text_lower}",
]


def make_negative_example(rng):
    base = rng.choice(HARD_NEGATIVE_PROMPTS)
    wrapper = rng.choice(NEGATIVE_WRAPPERS)
    text = wrapper.format(text=base, text_lower=base[0].lower() + base[1:])
    example, spans = build_o_example(text)
    return example, {"family": "semantic_hard_negative", "difficulty": "challenge", "mal_ids": [], "genre_canonicals": [], "theme_canonicals": [], "spans": spans}


def generate_split(name, count, title_pool, rng, global_prompts):
    # 80% clear contexts, 20% challenge contexts. Within the challenge block,
    # half are marked contrast cases and half are all-O semantic hard negatives.
    clear_count = round(count * 0.80)
    challenge_count = count - clear_count
    contrast_count = round(count * 0.15)
    negative_count = challenge_count - contrast_count
    plan = (["clear"] * clear_count + ["contrast"] * contrast_count + ["negative"] * negative_count)
    rng.shuffle(plan)
    examples, metadata = [], []
    attempts = 0
    for kind in plan:
        created = False
        while not created:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"Unable to generate {count} unique examples for {name}")
            if kind == "clear":
                template = rng.choice(CLEAR_TEMPLATES)
                example, meta = make_marked_example(template, title_pool, rng, "clear_entity_context", "clear")
            elif kind == "contrast":
                template = rng.choice(CHALLENGE_TEMPLATES)
                example, meta = make_marked_example(template, title_pool, rng, "contrast_or_boundary", "challenge")
            else:
                example, meta = make_negative_example(rng)
            key = example["prompt"].strip().casefold()
            if key in global_prompts:
                continue
            if not (5 <= len(example["tokens"]) <= 96):
                continue
            global_prompts.add(key)
            examples.append(example)
            metadata.append(meta)
            created = True
    paired = list(zip(examples, metadata))
    rng.shuffle(paired)
    examples, metadata = map(list, zip(*paired))
    return examples, metadata


def validate(examples):
    errors = []
    for row, example in enumerate(examples):
        tokens, labels = example["tokens"], example["labels"]
        if len(tokens) != len(labels):
            errors.append(f"row {row}: token/label mismatch")
            continue
        for i, label in enumerate(labels):
            if label not in VALID_LABELS:
                errors.append(f"row {row}, token {i}: invalid label {label}")
                continue
            if label.startswith("I-"):
                entity = label[2:]
                if i == 0 or labels[i - 1] not in {f"B-{entity}", f"I-{entity}"}:
                    errors.append(f"row {row}, token {i}: invalid transition {label}")
    return errors


def count_spans(labels):
    return Counter(label[2:] for label in labels if label.startswith("B-"))


def collect_report(name, examples, metadata):
    labels = Counter(label for ex in examples for label in ex["labels"])
    span_counts = Counter()
    entity_token_counts = Counter()
    stop_inside = Counter()
    stop_outside = 0
    lengths = []
    family = Counter(m["family"] for m in metadata)
    difficulty = Counter(m["difficulty"] for m in metadata)
    genre_usage = Counter()
    theme_usage = Counter()
    mal_ids = set()
    for ex, meta in zip(examples, metadata):
        lengths.append(len(ex["tokens"]))
        span_counts.update(count_spans(ex["labels"]))
        genre_usage.update(meta["genre_canonicals"])
        theme_usage.update(meta["theme_canonicals"])
        mal_ids.update(meta["mal_ids"])
        for token, label in zip(ex["tokens"], ex["labels"]):
            if label != "O":
                entity_token_counts[label.split("-", 1)[1]] += 1
            if token.casefold() in STOPWORDS:
                if label == "O":
                    stop_outside += 1
                else:
                    stop_inside[label.split("-", 1)[1]] += 1
    stop_inside_total = sum(stop_inside.values())
    total_stop = stop_outside + stop_inside_total
    return {
        "examples": len(examples),
        "tokens": sum(lengths),
        "average_tokens": round(statistics.mean(lengths), 3),
        "median_tokens": statistics.median(lengths),
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "label_counts": dict(labels),
        "entity_span_counts": dict(span_counts),
        "entity_token_counts": dict(entity_token_counts),
        "family_counts": dict(family),
        "difficulty_counts": dict(difficulty),
        "unique_mal_ids_used": len(mal_ids),
        "genre_usage": dict(genre_usage),
        "theme_usage": dict(theme_usage),
        "stopword_stats": {
            "outside_entities": stop_outside,
            "inside_entities": dict(stop_inside),
            "outside_percentage": round(100 * stop_outside / total_stop, 3) if total_stop else 100.0,
            "outside_to_inside_ratio": round(stop_outside / max(1, stop_inside_total), 3),
        },
    }


def write_jsonl(path, examples):
    with Path(path).open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazetteer", default="/mnt/data/1214b96d-0550-4959-a356-f72fdde60177.json")
    parser.add_argument("--output", default="/mnt/data/anime_multientity_ner_80_10_10")
    parser.add_argument("--zip", default="/mnt/data/anime_multientity_ner_80_10_10.zip")
    parser.add_argument("--total", type=int, default=15000)
    args = parser.parse_args()

    rng = random.Random(SEED)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    title_entries = load_title_entries(args.gazetteer)
    pools = split_title_pools(title_entries, rng)
    train_count = round(args.total * 0.80)
    val_count = round(args.total * 0.10)
    test_count = args.total - train_count - val_count
    global_prompts = set()
    train, train_meta = generate_split("train", train_count, pools["train"], rng, global_prompts)
    val, val_meta = generate_split("val", val_count, pools["val"], rng, global_prompts)
    test, test_meta = generate_split("test", test_count, pools["test"], rng, global_prompts)

    errors = validate(train) + validate(val) + validate(test)
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:20]))

    train_ids = {mid for meta in train_meta for mid in meta["mal_ids"]}
    val_ids = {mid for meta in val_meta for mid in meta["mal_ids"]}
    test_ids = {mid for meta in test_meta for mid in meta["mal_ids"]}
    leakage = {
        "train_val": sorted(train_ids.intersection(val_ids)),
        "train_test": sorted(train_ids.intersection(test_ids)),
        "val_test": sorted(val_ids.intersection(test_ids)),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Title leakage detected: {leakage}")

    all_examples = train + val + test
    report = {
        "seed": SEED,
        "source_gazetteer_entries": len(json.loads(Path(args.gazetteer).read_text(encoding="utf-8"))),
        "usable_ascii_title_entries": len(title_entries),
        "total_examples": len(all_examples),
        "split_rule": "80% train / 10% validation / 10% test",
        "difficulty_rule": "Within each split: 80% clear entity contexts / 20% challenge cases",
        "title_pool_rule": "80% train / 10% validation / 10% test title pools, disjoint by MAL ID",
        "title_pool_sizes": {k: len(v) for k, v in pools.items()},
        "title_leakage": leakage,
        "exact_prompt_duplicates": len(all_examples) - len({e["prompt"].casefold() for e in all_examples}),
        "taxonomy": {
            "genre_concepts": len(GENRE_TAXONOMY),
            "theme_concepts": len(THEME_TAXONOMY),
            "contrast_pairs": len(CONTRAST_PAIRS),
        },
        "splits": {
            "train": collect_report("train", train, train_meta),
            "validation": collect_report("validation", val, val_meta),
            "test": collect_report("test", test, test_meta),
        },
    }

    label_map = {
        "label_to_id": {
            "O": 0,
            "B-TITLE": 1, "I-TITLE": 2,
            "B-GENRE": 3, "I-GENRE": 4,
            "B-THEME": 5, "I-THEME": 6,
        },
        "id_to_label": {
            "0": "O", "1": "B-TITLE", "2": "I-TITLE",
            "3": "B-GENRE", "4": "I-GENRE",
            "5": "B-THEME", "6": "I-THEME",
        },
    }
    taxonomy = {
        "definition_rule": {
            "GENRE": "A recognizable story, market, or structural category.",
            "THEME": "An idea, conflict, or question that the story explores.",
        },
        "genres": GENRE_TAXONOMY,
        "themes": THEME_TAXONOMY,
        "contrast_pairs": CONTRAST_PAIRS,
    }

    write_jsonl(output / "anime_ner_train_80.jsonl", train)
    write_jsonl(output / "anime_ner_validation_10.jsonl", val)
    write_jsonl(output / "anime_ner_test_10.jsonl", test)
    write_jsonl(output / "anime_ner_all.jsonl", all_examples)
    (output / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (output / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(__file__, output / "generate_multientity_ner_dataset.py")

    readme = f"""# Anime multi-entity NER dataset

This dataset recognizes three entity types:

- `TITLE`: anime titles
- `GENRE`: story/market categories
- `THEME`: ideas or questions explored by a story

## Labels

`O`, `B-TITLE`, `I-TITLE`, `B-GENRE`, `I-GENRE`, `B-THEME`, `I-THEME`

Multiword genres and themes require `I-GENRE` and `I-THEME`.

## 80/10/10 split

- {train_count:,} examples (80%) are in `anime_ner_train_80.jsonl`.
- {val_count:,} examples (10%) are in `anime_ner_validation_10.jsonl`.
- {test_count:,} examples (10%) are in `anime_ner_test_10.jsonl`.
- Train, validation, and test title pools are disjoint by MAL ID.
- Within each split, 80% are clear contexts and 20% are challenge cases.
- Challenge cases include ordinary non-entity uses of taxonomy words, close genre/theme contrasts, multiple entities, and punctuation boundaries.

## Taxonomy rule

A genre is a recognizable category such as `romance`, `sports`, or `mecha`.
A theme is an underlying idea such as `love and vulnerability`, `competition and teamwork`, or `humanity and technology`.
See `taxonomy.json` for normalized concepts, aliases, and explicit contrast pairs.

## Files

- `anime_ner_train_80.jsonl`
- `anime_ner_validation_10.jsonl`
- `anime_ner_test_10.jsonl`
- `anime_ner_all.jsonl`
- `label_map.json`
- `taxonomy.json`
- `dataset_report.json`
- `generate_multientity_ner_dataset.py`

The anime gazetteer was used only as an offline source of real title strings. It is not required during inference.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    zip_path = Path(args.zip)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, arcname=f"{output.name}/{path.name}")

    summary = {
        "output": str(output),
        "zip": str(zip_path),
        "total_examples": len(all_examples),
        "train_examples": len(train),
        "validation_examples": len(val),
        "test_examples": len(test),
        "total_tokens": sum(len(e["tokens"]) for e in all_examples),
        "train_stopwords_outside_percentage": report["splits"]["train"]["stopword_stats"]["outside_percentage"],
        "validation_stopwords_outside_percentage": report["splits"]["validation"]["stopword_stats"]["outside_percentage"],
        "test_stopwords_outside_percentage": report["splits"]["test"]["stopword_stats"]["outside_percentage"],
        "duplicates": report["exact_prompt_duplicates"],
        "title_leakage": {k: len(v) for k, v in leakage.items()},
        "train_labels": report["splits"]["train"]["label_counts"],
        "validation_labels": report["splits"]["validation"]["label_counts"],
        "test_labels": report["splits"]["test"]["label_counts"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
