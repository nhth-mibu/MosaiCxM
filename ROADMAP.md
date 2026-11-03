# Roadmap

Planned work, in order. Nothing here is built yet.

## 1. AI features

### Offline (runs on your computer)

These keep the app fully offline, as it is today.

- **Smart content matching.** An image model such as CLIP sorts tiles by what they show, not only by colour. Faces go in the eyes, sky photos go in the sky, and so on.
- **Face-aware cropping.** Tiles crop around faces and subjects, so nobody's head gets cut off.
- **Quality filter.** Skip blurry, near-duplicate and badly exposed photos automatically.

### Cloud (optional add-on)

These need a cloud language model, such as Claude. They are off unless you turn them on.

- **Motif suggestions.** Pick the best photos in your library to use as the main picture, for example a sharp portrait with strong contrast.
- **Search by description.** Type "beach photos from 2023" to build a themed mosaic.
- **Shapes from text.** Type "a cat silhouette" and get an outline for the Shape picker.
- **Auto story.** A written summary of the statistics, for example "mostly shot at f/1.8 on your Fuji, peak year 2019".

## 2. iOS and Android apps

The Rust engine runs on both platforms as it is, and Slint can build for both.

- **Photo access.** Read from the phone's photo library (Apple Photos, Android media store) instead of folders.
- **Responsive layouts.** One app with two layouts, chosen by screen width:
  - Wide screens (desktop, iPad, Android tablets) keep the current side-by-side layout.
  - Narrow screens (phones) get a single column, one step at a time.
  - The layout switches by itself, for example when you rotate a tablet. A manual toggle is also possible.
- **Foldables.** Switch layout when the device folds or unfolds.
- **Memory limits.** Phones have less RAM, so very large print exports get tighter limits or stay desktop-only.
