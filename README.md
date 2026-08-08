# MalariaCheck — Malaria Symptom Risk Assessment System

MalariaCheck is a web application that helps people decide how urgently they
should seek a malaria test, based on the symptoms they are experiencing. It
uses a machine learning model trained on real hospital records to estimate the
likelihood that a person's symptoms are caused by malaria, and then gives
clear, practical guidance on what to do next.

> **Medical disclaimer:** MalariaCheck is a screening aid, not a diagnosis.
> Malaria can only be confirmed with a laboratory blood test. Anyone who feels
> unwell should visit a health facility regardless of what the tool says.

## Why we built it

Malaria remains one of the most common and dangerous illnesses in
sub-Saharan Africa, and it can become life-threatening within 24 hours if
left untreated. At the same time, its early symptoms — fever, headache,
body weakness — are easy to confuse with flu and other common illnesses, so
many people wait too long before getting tested. MalariaCheck bridges that
gap: it gives a person an evidence-based first assessment in under a minute
and, most importantly, tells them when testing is urgent.

## What we did

We started from a dataset of 1,622 anonymised hospital admission records
with laboratory-confirmed outcomes: 1,167 confirmed malaria cases and 455
patients whose illness turned out to be something else. Each record contains
the patient's age and eleven commonly reported symptoms, such as fever,
headache, vomiting, dizziness and joint pain.

We cleaned the data by removing personal identifiers, admission dates and
other columns that carry no medical signal. We also deliberately excluded the
dataset's pre-computed risk score from training, because it is derived from
the symptoms themselves and would have let the model "cheat" (a problem known
as data leakage).

On this cleaned data we trained a Decision Tree classifier — a model that
learns a series of yes/no questions about symptoms, much like the mental
checklist a clinician follows. We chose it because its reasoning is fully
transparent: the application can show users and reviewers the exact tree of
decisions behind every assessment. The model was evaluated on a held-out 30%
of the records that it never saw during training.

Finally, we wrapped the model in a carefully designed web interface and
shaped the whole experience around safety rather than raw prediction.

## How the system works

A user answers eleven simple yes/no questions about their current symptoms
and enters their age. The model compares this profile against the patterns it
learned from the hospital records and produces an estimated likelihood of
malaria, shown on a colour-coded gauge together with a plain-language
recommendation.

Two safety rules sit above the model:

- **Danger signs override everything.** If the user reports confusion or
  repeated vomiting — possible signs of severe malaria — the app displays an
  urgent-care warning and the emergency number, no matter what the model
  predicts.
- **Fever is never dismissed.** Even when the model considers malaria
  unlikely, a user who reports fever is advised to get tested within 24
  hours, because in a malaria-endemic area any fever deserves a test.

## What the application offers

- **Self-Assessment** — the guided symptom questionnaire with a
  probability-based result, risk gauge and recommended next steps
- **Example cases** — pre-filled patient profiles that demonstrate how the
  model responds to different symptom combinations
- **Data Insights** — interactive charts showing the diagnosis distribution,
  age patterns, and how symptom prevalence differs between malaria and
  non-malaria patients
- **Model Performance** — the honest numbers: accuracy, sensitivity,
  specificity and precision on the held-out test set, with a confusion matrix
  and feature importance
- **Decision Tree view** — a full visualisation of the trained model, so the
  reasoning behind every assessment can be inspected
- **Help & About** — malaria facts, prevention advice, when to seek care
  immediately, and the system's limitations

## How well it performs

On the held-out test set the model reaches about 89% overall accuracy. More
importantly for a screening tool, its sensitivity is about 89% — meaning it
correctly flags roughly nine out of ten real malaria cases — and its
specificity is about 88%, meaning it rarely alarms people whose illness is
not malaria. For screening, sensitivity is the number that matters most:
missing a real case is far more dangerous than sending a healthy person for
a quick, inexpensive test.

## Privacy

Nothing the user enters is stored or transmitted. Every assessment is
processed only within the user's own session and disappears when the page is
closed.

## Limitations

- Malaria symptoms overlap with flu, typhoid and other febrile illnesses;
  only a blood test can confirm the diagnosis
- The training data comes from one region's hospital records, so performance
  may differ for other populations
- The tool does not account for pregnancy, chronic illness or medication —
  situations that need professional judgement

## How to run it

1. **Get the project.** Clone this repository or download it as a ZIP file
   and extract it anywhere on your computer.
2. **Have Python ready.** Any recent version of Python (3.10 or newer) works.
3. **Install the required packages.** Open a terminal inside the project
   folder and run `pip install -r requirements.txt`. This installs everything
   the app needs in one step.
4. **Start the application.** In the same terminal, run
   `streamlit run malaria_app.py`.
5. **Use it in your browser.** The app opens automatically at
   `http://localhost:8501`. Use the sidebar to move between the six sections,
   and start with **Self-Assessment** to try the symptom questionnaire.

To stop the application, return to the terminal and press `Ctrl + C`.
