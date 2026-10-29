# AGENTS.md

Instructions for any AI assistant (Claude, Copilot, ChatGPT, etc.) working on this project.
**Read [PROJECT_MEMORY.md](PROJECT_MEMORY.md) first.** It holds the decision log, open problems and where we left off.

## What this project is

A PPE safety check on a **SoftBank Pepper robot** (CAIRO Challenge 3). A worker taps Start on Pepper's tablet and stands on footprints, watching themselves in a live view. Pepper recognises them, takes a photo after a 3‑2‑1 countdown, and a YOLO model checks their safety gear. Then Pepper either wishes them a great day, or shows what's missing. If the same person comes back and something is still missing, Pepper asks them to wait at a spot and calls a helper.

## Repos

| Repo | What |
|---|---|
| this repo (`cairo_challenge_3`) | Tablet UI: `pepper_ppe_ui/html/index.html` |
| `github.com/CAIRO-Challenge-3/UI_Safety`, branch `feat/safety-face-detection` | Backend prototypes: YOLO PPE detection, face recognition. **Do not merge it into the UI code.** The two only meet through the HTTP contract in `pepper_ppe_ui/README.md`. |

## Layout

```
AGENTS.md              this file
PROJECT_MEMORY.md      decision log + status; update it after every change
pepper_ppe_ui/
  README.md            flow, deploy steps, backend HTTP contract
  html/index.html      the whole tablet app (HTML + CSS + JS in one file)
  html/navigation.html standalone directions page (map + moving arrows); NOT part of the PPE flow
  html/room.html       room assistant, one URL per room: room.html?room=werkstatt-1|werkstatt-2|speisesaal|sanitaer
                       (pictograms by colour, safety rules, daily plan + lunch reminder, emergency escape map,
                       call a helper). Its "Sicherheits-Check" tile only LINKS to index.html.
```

The repo is `CAIRO-Challenge-3/UI_Safety` (`main`). The backend prototypes are on branch `feat/safety-face-detection`.

## Hard rules

0. **The UI is frozen at v1.0** (git tag `ui-v1.0`). Don't change `index.html` unless the user asks. Backend work must fit the existing HTTP contract. If the contract has to change, say so first.

1. **Minimal screens.** Start = only the button. Position = live view + one hint. Results = only what matters, plus a *Nächste Person* countdown button. No greeting page, no scanning page, no directions, no "you passed" symbols, no explanatory copy.
2. **German first.** German at power-on. The DE/EN switch applies to every following screen until switched again. All user-facing strings live in `T.de` / `T.en` (and `PPE[k].de/.en`). Never hard-code UI text.
3. **Voice = Pepper's built-in TTS** (`ALAnimatedSpeech` / `ALTextToSpeech`). No external TTS or language models.
4. **Tone: fun and friendly.** No red, no alarms, no "violation" wording.
5. **No confirmation screens.** Results reset for the next person on their own. A returning person is recognised by face, and Pepper remembers what they were missing.
6. **Old WebView.** The tablet (NAOqi 2.5) runs an old Android Chromium: **ES5 only** (`var`, `function`, no arrow functions, template literals or `Array.includes`). In CSS: no CSS variables, no grid, no flex `gap`.
7. **One file, no build step.** **Demo mode must keep working** in a normal browser.

## Run

- **Browser demo:** open `pepper_ppe_ui/html/index.html`. The demo panel is on the right.
- **Pepper:** see `pepper_ppe_ui/README.md`.

## After each change

Add a dated entry to `PROJECT_MEMORY.md` → **Change log**: what changed, why (the user's request or the problem), and any follow-ups.
