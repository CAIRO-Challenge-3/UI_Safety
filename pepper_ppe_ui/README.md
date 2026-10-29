# Pepper PPE Check – tablet UI

`html/index.html` is the whole app: one file, no build step. It works in two modes:

- **Browser (demo):** open `html/index.html` (or add `?demo=1`). A demo panel fakes the backend. Opened as a local file, the live view uses your webcam.
- **Pepper:** when the page loads on the tablet it connects to NAOqi through `/libs/qimessaging/2/qimessaging.js`, hides the demo panel, and calls the real backend.

**Language:** German at power-on. The `DE | EN` switch (top right) changes every screen, and the choice stays until someone switches it again. It is saved, so it also survives a reload.

**Voice:** Pepper's built-in text-to-speech (`ALAnimatedSpeech`), with the language set through `ALTextToSpeech.setLanguage("German" | "English")`. Pitch and speed are in `CONFIG.voice`. German needs the German language pack installed on the robot.

## Flow (no confirmation buttons)

1. **Start:** only the start button.
2. **Position:** live camera view with the footprint guide and hints ("Bitte einen kleinen Schritt zurück"). The person is recognised here, and a name tag shows "Hallo Francesca!" or "Willkommen zurück…". Once they stand right: 3‑2‑1, photo, check.
3. **Result**, straight after the photo:
   - **All good:** "Einen schönen Tag, Francesca!", plus a *Nächste Person* button that counts down 10 s.
   - **First visit, missing:** only the symbols and names of the missing items, plus *Nächste Person* (15 s). Pepper remembers what this face was missing (60 min).
   - **Return visit, still missing:** missing symbols, plus "Bitte warte am Wartepunkt". The helper is called after a 10 s countdown (or right away with *Jetzt rufen*). The same page then shows "Maria ist informiert…" and *Nächste Person* (20 s).

## Deploy to Pepper (NAOqi 2.5)

1. Create a Choregraphe project and copy `html/` into it.
2. In the behaviour, use the *Show App* box, or call `ALTabletService.loadApplication("<package-id>")` and then `showWebview()`.
3. Set `CONFIG.backendUrl` in `index.html` to your laptop/server IP.

## Backend contract (JSON over HTTP, CORS enabled)

| Call | Response |
|---|---|
| `GET /live.mjpg` | MJPEG stream of Pepper's head camera, for the live view |
| `GET /frame` | `{"hint": "none" \| "step_back" \| "come_closer" \| "move_left" \| "move_right" \| "ok", "person": {"id","name"} \| null}`. For an unknown face, return a **temporary id** (e.g. `"visitor-7f3"`) with `name: null`, so the return visit is still recognised. |
| `POST /check` body `{"required": [...]}` | Takes the photo now, identifies the person and runs YOLO: `{"person": {"id","name"}, "items": {"helmet": true, "goggles": false, "vest": true, "gloves": false, "boots": true}}` |
| `POST /help` body `{"person","name","missing","visit","lang"}` | `{"eta_min": 2}` |

## ALMemory events raised by the page

- `PPECheck/Result`: `{person, visit, missing}`
- `PPECheck/HelpRequested`: `{person, name, missing, visit, lang}`. Hook your notification (Slack, pager, etc.) here.

## Things to edit

`CONFIG` (required items, timers, voice pitch/speed, helper), `T.de` / `T.en` (all texts, including the wait-spot text), `PPE` (labels per language).
