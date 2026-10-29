# Project memory

Running log so any AI or person can pick up where we left off. Newest entries go on top of each section.

## Current status (2026-10-29)

- **Tablet UI: FINAL v1.0 (frozen, git tag `ui-v1.0`).** The user approved the safety check pipeline in `pepper_ppe_ui/html/index.html` as final. Don't change its flow, screens or texts unless the user explicitly asks. `navigation.html` is a separate, standalone page.
- **Backend:** next phase. The user will give the requirements, and only after that is the backend built and its behaviour checked. Right now there are only prototypes in the `UI_Safety` repo (branch `feat/safety-face-detection`), with no HTTP API.
- **Next up:** collect the user's backend requirements, then implement the contract in `pepper_ppe_ui/README.md` (`/live.mjpg`, `/frame`, `/check`, `/help`), test it, then try it on the real robot.

## User requirements (latest wins)

1. A person starts the check with **one button**. The start page shows nothing else.
2. **Positioning page with a live camera view**, so people see where to stand. Pepper **recognises the person here** (no separate greeting page). Once they stand right: **3‑2‑1 timer, then the photo**.
3. **No scanning page.** Straight after the photo, show **only the missing items** (symbol + name). No "almost there" or "grab these" text, and nothing about the items that passed.
4. **No directions** to the workplace.
5. **Second failure** (same person comes back, still missing): ask them to **wait at a certain spot**, call the helper after a **10 s timer**, say on the **same page** that the helper was called, then a timer for the next person.
6. **All pass:** just "have a great day" (Pepper says "Rock on" in the voice). No pass symbol, no extra buttons. The *next person* timer and button start right away.
7. **German is the default** on all pages. The DE/EN switch **carries over** to the following pages until someone switches it again.
8. **Voice = Pepper's built-in TTS.** No language model or external TTS.
9. No "I'm wearing it now" confirmation. The screen resets for the next person, and Pepper remembers each face's missing gear.
10. Fun, friendly look. Never industrial or alarming.
11. Test checklist: **gloves** and **helmet** scenarios must work.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-29 | Voice is **only Pepper's built-in `ALAnimatedSpeech`**, with pitch `\vct=115\` for a cheerier sound. Language is set via `ALTextToSpeech.setLanguage`. Piper/TTS models dropped. | User decision. |
| 2026-10-29 | The language choice is saved (`localStorage`, `pepper-ppe-lang`) and is never reset by the flow. German only at first power-on. | "Carry over until changed again." |
| 2026-10-29 | Flow cut to **Start → Position (live + recognise + photo) → Result**. Greeting, scan, directions and calling pages are removed. The helper call happens inside the still-missing page. | User wanted it minimal. |
| 2026-10-29 | Backend gives `/frame` (hint + person) during positioning, and `/check` (photo + identify + YOLO) at the end. The live view comes from `/live.mjpg`. | Recognition happens on the positioning page; one call at the end gives the result. |
| 2026-10-29 | Every result page has a *Nächste Person* button that counts down (pass 10 s, missing 15 s, after the helper call 20 s). | User: "timer with a button for the next person". |
| 2026-10-29 | The still-missing page keeps a small **Jetzt rufen** button during the 10 s countdown. | The original brief asks for a Help button. It only speeds up the automatic call. |
| 2026-10-29 | Pepper remembers missing gear **per face id** for 60 min (tablet `localStorage`). Visitors get a temporary id from the backend. | Needed for the no-confirmation flow. |
| 2026-10-29 | Light, playful style: Pepper's face, coral/mint/sunny/lavender colours, Baloo 2 + Nunito fonts, confetti on pass. | The industrial look was rejected as alarming. |
| 2026-10-29 | UI and backend talk only through the HTTP contract. ES5, single HTML file. | Separate repos; old tablet WebView. |

## Voice notes (Pepper built-in TTS)

- Works offline with no delay. `ALAnimatedSpeech` adds gestures that stay in sync with the speech.
- German needs the **German language pack** on the robot. Check with `ALTextToSpeech.getAvailableLanguages()`. If it's missing, German text gets read with the English voice.
- Make it cheerier with `CONFIG.voice.pitch` (`\vct=`, around 110–120) and `speed` (`\rspd=`, around 95–105).
- In the browser demo, the browser's own voice (de-DE / en-US) stands in for Pepper.

## Backend analysis (UI_Safety @ feat/safety-face-detection, commit 644173c)

**Works:** the YuNet + SFace face recognition (`face_greeting.py`), grabbing frames from Pepper's camera with NAOqi (`ppe_detection_pepper.py`), and the YOLO webcam test bench (`ppe_webcam.py`).
**Missing:**
1. No HTTP API. The UI needs `/live.mjpg`, `/frame`, `/check` and `/help`, with CORS.
2. The default model (Roboflow construction-site-safety, YOLOv8n) has **no gloves, goggles or boots** classes, so the glove test fails. The YOLOv11 weights aren't in the repo.
3. It relies on "NO-*" boxes. It should instead check whether a PPE box is inside the person's box.
4. Results aren't tied to one person when several are in frame.
5. Decisions come from single frames. Combine several frames, or a short burst around the photo.
6. Nothing builds the `/frame` positioning hint. It can come from the Person box.
7. Camera framing: 640×480 at 2 m probably can't see head to feet. Tilt the head down and use a higher resolution for the photo.
8. The scripts' own speech (`ALTextToSpeech` warning, `pyttsx3`) duplicates the UI's voice. Remove it.
9. Smaller issues: `qi` is missing from requirements, the docstring names a script that doesn't exist, empty frames cause a busy loop, and `faces/` isn't in `.gitignore` (privacy risk).

## Open problems / TODO

- [ ] Backend HTTP server implementing the contract in `pepper_ppe_ui/README.md`, including the MJPEG live stream.
- [ ] Retrain YOLOv11 with helmet, glasses, vest, gloves and boots. Until then set `CONFIG.required = ['helmet','vest']`.
- [ ] Check that the German TTS pack is on the actual Pepper.
- [ ] Mark the physical waiting spot ("gelbe Markierung rechts neben Pepper") and footprints on the floor, and adjust the `waitWhere` text to match.
- [ ] Hook `PPECheck/HelpRequested` to a real notification for the helper.
- [ ] Decide where "who was missing what" should live long-term (tablet `localStorage` today, backend later).
- [ ] Privacy: face data and the memory of missing gear may need consent or a notice (GDPR).
- [ ] Try it on the real tablet: does MJPEG work in its WebView, timing, font sizes.

## Change log

### 2026-10-29: room assistant, round 2 (user feedback)
- **Safety check now runs inside room.html** (same app). Screens `sc-*`: position (live view, recognise, 3‑2‑1 photo) → pass / missing / still-missing + helper → "Nächste Person" returns to the **room home**. The pipeline, texts and look are copied 1:1 from `index.html`, with CSS scoped under `.sc`. A small "Zurück" button was added on the positioning screen. It shares the `pepper-ppe-pending` memory with `index.html`. **`index.html` and `navigation.html` are not changed** (verified against `ui-v1.0`).
- **Real safety-sign images, not restyled:** `assets/signs/*.svg` are the official ISO 7010 graphics from Wikimedia Commons: W012 voltage, W017 hot, M009 gloves, M004 goggles, M011 wash, M030 trash, E001 exit, E003 first aid, E010 AED, E007 assembly point, F001 extinguisher. The requirements PDF with the user's own images is **not on disk**. If their images differ, drop them in with the same file names.
- **Escape map is now the original site plan.** `assets/lageplan.jpg` was extracted from `Downloads/Layout-plan-of-a-workshop.pdf` (1024×1024, German version). The route, current-room box, "Du bist hier" and the E007 assembly-point sign are drawn on top in the image's pixel coordinates, with moving green arrows. Routes were checked by drawing them onto the plan: the west doors of WB1/WB2 are at y≈380/440, and sanitary goes corridor → foyer door (y≈575) → main entrance. The assembly point is at the end of the front walkway by the street, which is an **assumption** because the plan has none.
- **Read-aloud now stops** on every screen change, when the sign popup is closed, and when the lunch reminder is dismissed (`stopSpeech()`: `ALTextToSpeech.stopAll()` / `speechSynthesis.cancel()`).

### 2026-10-29: room assistant UI (new file, safety check untouched)
- New `pepper_ppe_ui/html/room.html`, built from the user's "New UI design requirements" PDF and the site plan (Werkstatt für Menschen mit Behinderung, ground floor).
- One file with one address per room (`?room=werkstatt-1|werkstatt-2|speisesaal|sanitaer`). Screens:
  - launch page with only the Start button
  - room home laid out like the wireframe: time & date + room name header, "Hilfe rufen" top-left, "Notausgang" top-right, 2×2 tiles (Piktogramme, Sicherheits-Check, Sicherheits-Regeln, Tagesplan)
- **Piktogramme:** 4 colours (blue = must, red = forbidden/fire, green = escape/first aid, yellow = warning), then that room's signs, then a text popup that reads aloud. "Schließen" goes back to the room home (as the spec says). Colours with no signs in the room are greyed out.
- **Sign texts** are taken from the PDF. Small grammar fixes: "kann *ein* schlimmer Unfall", "*Legen* Sie Ihre Hand nicht…", "*Wirf* deinen Müll…". First aid and the defibrillator are split into two signs. The PDF mixes "du" and "Sie"; that's left as given (open question).
- **Sicherheits-Check** tile is a plain link to `index.html`. The safety check itself is unchanged (still `ui-v1.0`).
- **Tagesplan:** example day per room, with the current block highlighted and a lunch banner ("noch 1 Std. 20 Min."). A lunch reminder pops up (and is spoken) 10 min before lunch.
- **Notausgang:** schematic ground-floor map redrawn from the site plan, with the current room highlighted, "Du bist hier", green moving arrows to the assembly point, and 4 easy-language steps, spoken automatically.
- **Hilfe rufen:** raises `Room/HelpRequested`, POSTs `/help` `{room, reason:'assistant', lang}`, and shows "Maria kommt gleich!".
- **Language:** DE default, with the same shared `pepper-ppe-lang` key as the safety check, so the choice carries over between the two.
- **Example content to replace with real data:** daily plans, safety rules, lunch times, and the escape routes and assembly point. **The escape routes are assumptions** from the sample plan and must be checked against the official Flucht- und Rettungsplan.
- **Known bug found in frozen `index.html`, not fixed:** it has no `[hidden]{display:none!important}` rule. On the real tablet, the "Moment…" overlay and the "Nächste Person" button on the still-missing page would show while they're supposed to be hidden. The artifact previews hide the bug because the preview page adds that rule. Needs the user's OK to patch.

### 2026-10-29: UI frozen as v1.0
- The user confirmed the safety check UI pipeline is right. It's committed and tagged `ui-v1.0`. From here on the work is the backend.

### 2026-10-29: pass message + standalone directions page
- Pass page now reads "Sicherheitscheck bestanden!" with "Einen schönen Tag, <Name>!" below (EN: "Safety check passed!" / "Have a great day, <Name>!"). Pepper says "Rock on, … Sicherheitscheck bestanden …".
- New **standalone** `pepper_ppe_ui/html/navigation.html`, not linked from the PPE flow. It shows the factory map with a dotted route and **moving arrows** that point the way (positions computed in JS along the route; static when reduced motion is on), a pulsing goal, numbered steps, DE/EN (shares the `pepper-ppe-lang` key with the check), and a "Bring mich hin" button that raises `PPECheck/GuideTo`. Destinations live in `ZONES` (`line-b-bay4`, `briefing`, `line-a`), chosen with `?zone=` or the demo dropdown.
- Why: user wants the directions UI as a separate page to adapt for another UI.

### 2026-10-29: built-in voice only
- Removed the Piper/`/tts` path and the `ALAudioPlayer` code. `say()` now only uses Pepper's `ALAnimatedSpeech` (browser voice in the demo). `CONFIG.voice` is just `pitch` and `speed`.
- Why: user said they'll use the robot's default TTS, not a model.

### 2026-10-29: big simplification + language carry-over
- Start page is only the button. The header, step dots and Pepper speech bubble are gone (the DE/EN switch stays, top right).
- Removed the greeting, scan, directions and separate calling pages.
- Positioning page has a live view: MJPEG from the backend on the robot, the webcam in the demo, or an illustration otherwise. It shows a name tag when the person is recognised, a 3‑2‑1 countdown, a flash, and "Moment…" while checking.
- Missing page shows only the missing symbols and names, plus *Nächste Person* (15 s).
- Still-missing page: missing symbols, "Bitte warte am Wartepunkt", a 10 s ring before the helper is called (plus *Jetzt rufen*), then "Maria ist informiert…" and *Nächste Person* (20 s).
- Pass page: "Einen schönen Tag, Francesca!", confetti, and *Nächste Person* (10 s).
- The language now persists across screens and reloads. Switching it re-renders the open screen.
- Backend contract changed: `/identify` + `/detect` became `/frame` (with person) + `/check`, and `/live.mjpg` was added.
- Why: user request.

### 2026-10-29: German default, demo name
- UI starts in German, with a DE/EN switch. Demo person renamed Lena Hoffmann → Francesca (first name only).

### 2026-10-29: no-confirmation flow, EN/DE, agent docs
- Removed the "I'm back · Check again" buttons and added per-person memory of missing gear.
- Added EN/DE text tables, `AGENTS.md` and this file.

### 2026-10-29: friendly redesign
- Replaced the industrial signage look with a light, playful Pepper style.

### 2026-10-29: first UI prototype
- Single-file tablet app with demo panel. Backend branch analysed.
