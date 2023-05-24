---
title: Qg Generation
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.19.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

This is a question-generating application where the user writes a piece of scientific text and the model outputs questions based on that text.
At the moment we have fine-tuned the FLAN-T5-Base and BART-Large models on the r/AskScience dataset and also include the zero-shot output from FLAN-T5-XXL and GPT-3.5.
For more information and to test this application yourself, click <a href="https://huggingface.co/spaces/dhmeltzer/qg_generation">here</a>.
