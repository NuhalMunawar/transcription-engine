# Transcription & Tagging Instructions

## 1. Segmentation & General Rules
* **Continuous Speech:** We can keep multiple sentences in the same segment if the person speaks continuously. If there is a gap after a sentence or if the person is not speaking continuously, create separate segments for each sentence.
* **Sentence Integrity:** Do not split a single sentence into separate segments.
    * *Wrong:* "The hotel was beautiful" / "and had a great view."
    * *Right:* "The hotel was beautiful and had a great view."
* **Numbers:** Default to numerals unless there is a context-specific reason to spell it out (e.g., "I had 2 sandwiches for lunch", "Lunch was quite expensive, it was 50 dollars").
* **Unfinished Segments:** Put `...` at the end of the segment.
* **Incomplete Words:** Use hyphens to indicate incomplete words (e.g., `everywhe-`).
* **Distortion:** Do not distort words. Write "Yes" instead of "Yeeeeeeeeessssssssss" or "No" instead of "Nooooooooo".
* **Stuttering:** Write it exactly as you hear it. Example: `M-m-my name is.` Do NOT use the `[stuttering]` tag.
* **Punctuation:** * Punctuation should reflect tone: Questions end with `?`, Exclamations end with `!`.
    * It should only be used at the end of sentences.
* **Grammar & Formatting:** Names of people or places should be capitalized. Special characters should be typed out instead of using symbols (e.g., write "at", "dash", "underscore", "hash").
* **Unintelligible Speech:** Write it verbatim if possible, otherwise use `[inaudible noise]`, `[mumbling]`, or `[unintelligible]`. Use the view prompt feature to check context.

## 2. Fillers & Non-Speech Sounds
* **Filler Words:** All filler words must be in square brackets. Example: `[uh]`, `[uh-huh]`, `[hmm]`, `[mhm]`, `[ah]`.
    * *More Examples:* `[woah]`, `[phew]`, `[pff]`, `[huh]`, `[eh]`, `[heh]`, `[aw]`, `[uff]`, `[beep]`, `[purr]`, `[duh]`, `[eek]`, `[neh]`, `[ye]`, `[blah]`, `[yada yada yada]`, `[ding]`, `[boop]`, `[yay]`, `[boom]`, `[ooh la la]`, `[boo]`, `[boink]`, `[dang]`, `[oof]`, `[aha]`, `[poof]`, `[meh]`, `[ehm]`, `[ouch]`, `[eww]`, `[yuck]`.
* **Not Fillers:** yum, yo, gosh, geez, whoops, oops, wow, ayo, peek-a-boo.
* **Exceptions:** "Oh" and "Nah" can be transcribed as normal words. Sounds starting with ooo/uuu (like oooohhh) should just be written as `oh` without brackets.
* **Human Sounds:**
    * Laughing, coughing, sneezing, etc., use: `[laughing]`, `[laughs]`, `[chuckling]`, `[chuckles]`, `[coughs]`, `[sneezes]`, `[clears throat]`, `[scoffs]`, `[tongue click]`.
    * If someone laughs and talks simultaneously, write `[laughing]` or `[chuckling]` and then write the word.
    * Singing/Humming: Use `[singing]` (and transcribe lyrics), `[humming]`, or `[scatting]` (for sounds like da da da).
* **Inhuman Sounds:** Do not transcribe `[mouse clicking]` or any other inhuman sounds (only transcribe sounds caused by the mouth).

## 3. Breathing & Pauses
* **Breathing:** Normal breaths before speaking do not count. Inhale/exhale shouldn't be transcribed at the beginning/end of segments unless it conveys meaning (e.g., `[deep breath]`, `[sigh]`). Breathing relevant to the context (nervousness, yoga, workouts) should be transcribed.
* **Pauses:**
    * If a break breaks natural flow and is < 2 seconds, use `[short pause]`.
    * If it lasts 2-3 seconds, use `[pause]`.
    * If the gap is > 3 seconds, create separate segments.
    * Pauses should **never** happen at the very start or end of a segment/sentence; they should always be in the middle.
* **Mouth Noises:** `[sigh]`, `[gasp]`, `[claps]`, contextual inhale/exhale, sucks teeth must be segmented and transcribed. *(Note: no need for lipsmack or tongue click).*

## 4. Bracket Rules `[]`
* Don't use punctuation (except hyphens) inside the square brackets.
* Don't use a full stop after `[]`.
* The highest amount of focus is on square brackets for filler words and pauses.

## 5. Non-English Speech
* First write `[speaking in 'specific language']` and then transcribe the language.
* Example: `[speaking in Spanish] Hola, mi nombre es Nafee.`
* Do not write the name of the language inside quotation marks.

---

## 6. Third Review: Tags (Emotion, Delivery, Prosody)
* **Definitions:**
    * **Emotion:** E.g., `[happy]`, `[sad]`, `[frustrated]`, `[angry]`, `[excited]`.
    * **Delivery:** E.g., `[loudly]`, `[quietly]`, `[softly]`.
    * **Prosody:** E.g., `[Slow down]`, `[Speeds up]`. (Related to rhythm; no longer using the "drawn out" tag).
* **Tagging Rules:**
    * Always put the tag **before** the word/sentence, never after.
    * Do not use "tone" after the tag (e.g., avoid "soft tone", "loud tone").
    * Emphasized words next to each other should be grouped: `**I am angry.**`
    * Not every segment needs to be tagged. It is rare for a single segment to have all three dimensions, though they can overlap (e.g., `[loudly] [speeds up]`).
    * Use synonyms if you feel comfortable. Ballpark reasonable tagging is acceptable.
    * **Context Matters:** Don't automatically start tagging laughter as `[joyful]` or `[jolly]`. Context determines the emotion behind the laugh.
* **Banned Tags:** inquisitive, pondering, suggestive, squeaky, funnily, questioning, affirming, agreeing, commanding, perceptive, strongly/firmly (use `**` emphasis instead), POLITE.

### Example Emotions

#### Positive:
Happy, Content/satisfied/relaxed/calm/serene, Curious/interested, Optimistic, Amused, Delighted, Surprised, Excited/ecstatic, Amazed, Eager, Desperate, Trusting, Relieved, Hopeful, Proud, Courageous, Caring/tender, Compassionate, Joyful/glee, Sentimental, Enthusiastic, Blissful, Astonished.

#### Negative:
Angry/outraged/furious, Disbelief, Remorseful, Regretful, Anxious/nervous, Aggressive, Bored, Envious, Pessimistic, Embarrassed/humiliated/mortified, Jealous, Frustrated/annoyed/irritated/agitated, Impatient, Disappointed, Helpless, Insecure, Confused/unsure/doubtful, Sad, Resentful, Contempt/loathing, Scared/fearful/apprehensive, Alarm, Panic, Sceptical/dismissive, Suspicious, Lonely, Guilty, Worried, Insulted, Hostile.

#### Other Emotions:
Confused, Baffled, Puzzled, Agitated, Surprised, Relief, Anticipation, Disgust, Nostalgia, Appreciation, Emphatic, Boredom, Dramatic, Sarcastic, Thoughtful, Playful, Assertive, Supportive, Compassionate, Affectionate, Fear, Anxiety, Embarrassment, Guilt, Shame, Pride, Envy, Jealousy, Helplessness, Nervousness, Disbelief, Emptiness, Mockery, Suspicion, Confidence, Indifference, Lazy.
