# Aman Jain

Machine learning engineer working on real-time computer vision and applied GenAI, and on the Python services and interfaces that carry them into production.

Most of my work sits between a model and a working product: quantising models so they run on constrained hardware, designing the threading around them so latency stays flat, and building the API or dashboard an operator actually uses. B.Tech in Information Technology, 2026.

---

## Projects

### Hybrid RAG Search
**[Code](https://github.com/Amanjaiiinnn/Hybrid-Rag-Search)**

A retrieval-augmented search engine over a product catalog, with the retrieval maths written from scratch rather than pulled from a framework.

- **BM25 built by hand:** tokeniser, inverted index, IDF and the full saturation/length-normalisation scoring formula, with `k1` and `b` adjustable live.
- **Three retrievers, compared:** keyword, vector (cosine similarity) and hybrid, fused with Reciprocal Rank Fusion and with normalised score fusion.
- **Measured, not assumed:** an evaluation harness scores every retriever against hand-labelled queries using Precision@K, Recall@K, MRR and NDCG@K. Hybrid RRF reached **NDCG@3 of 0.784** against **0.735** for BM25 alone, and BM25 kept a perfect MRR.
- **Grounded answers** generated with Llama 3.3 on Groq, restricted to the retrieved products.

`Python` · `NumPy` · `Streamlit` · `Groq` · `Hugging Face`

### SnapClass — AI Attendance
**[Code](https://github.com/Amanjaiiinnn/supaclass-ai-attendance)** · **[Live demo](https://supaclass.streamlit.app/)**

A full-stack attendance system where students sign in with their face and teachers mark a whole class from classroom photos or a recording.

- **Face recognition:** dlib detection and 128-dimensional embeddings, classified by a linear SVM, with a distance check so an unknown face is never forced onto the nearest student.
- **Speaker recognition:** a second identification path that splits classroom audio at silences and matches each segment against enrolled students' voice embeddings.
- **Backed by Postgres:** a five-table Supabase schema covering teachers, students, subjects, enrolments and attendance sessions, with bcrypt-hashed passwords and QR join links.

`Python` · `Streamlit` · `Supabase / PostgreSQL` · `dlib` · `scikit-learn` · `Resemblyzer`

### AI Real-Time Gym Trainer
**[Code](https://github.com/Amanjaiiinnn/AI_GYM_TRAINER)** · **[Live demo](https://gymaitrainer.streamlit.app/)**

A webcam coach that counts your reps, checks your form and talks back.

- **Pose analysis in the browser** over WebRTC, with MediaPipe landmarks processed per frame.
- **Five exercises, each with its own rules:** rep counting as a joint-angle state machine, plus form checks such as squat depth, hip sag in push-ups, elbow drift in curls and lower-back arch in presses.
- **A coach that knows when to speak:** metrics are turned into a specific fault, sent to Llama 3.3 for a short cue, spoken aloud, and rate-limited so it corrects rather than nags.

`Python` · `MediaPipe` · `streamlit-webrtc` · `OpenCV` · `Groq` · `SQLite`

### Neural Style Transfer (AdaIN)
**[Code](https://github.com/Amanjaiiinnn/Style_transfer_Vgg)** · **[Live demo](https://huggingface.co/spaces/Amanprime/NST)**

Repaint any photo in the style of any artwork in one forward pass, with a decoder I trained against a frozen VGG-19 encoder.

- **Adaptive Instance Normalization:** content features are re-normalised to the style's per-channel statistics, so unseen styles work without retraining.
- **Style strength control** by interpolating between content and stylised features before decoding.
- About **5–6 seconds per 512×512 image on CPU**; deployed as a Hugging Face Space.

`PyTorch` · `Torchvision` · `Flask` · `Gradio`

---

## Machine Learning Intern — EDS-INDIA (R&D)
*April 2026 – present · sole developer on two client-facing edge-AI products*

**Metro platform safety system** — a real-time system on an i.MX93 board with an Ethos-U65 NPU that raises an alert when a passenger crosses the platform safety line. INT8 YOLO11 detection, pose estimation and a garment classifier, now running in pilot with an enterprise transit client.

- Designed the concurrency model around a single NPU-owning worker thread pinned to its own core, fed by drop-oldest channels, so a slow consumer drops stale frames instead of building a backlog and end-to-end latency stays flat under load.
- Cut false alerts by roughly **40%** with a multi-frame voting layer and a garment-colour filter, backed by train-presence suppression that invalidates votes at both edges of a train's stay.
- Took the pipeline from **4–5 FPS to a sustained 11 FPS** end to end on the same hardware through post-training INT8 quantisation and TFLite/Vela compilation, cutting model size about 4×.
- Built the operator dashboard in Flask: live MJPEG stream, on-image virtual-line configuration, hot config reload without a restart, and bilingual (English/Hindi) spoken alerts.

**Driver Monitoring System (STM32MP2)** — face detection, facial landmark and iris tracking, eye- and mouth-ratio drowsiness and yawn detection, and a custom-trained YOLOv8n for phone use and smoking, combined into a rolling driver safety score. Trained on a 97k-image dataset, reaching **93% accuracy** on drowsiness/attention and distraction detection.

- Diagnosed a memory leak of roughly **47 MB per 500 inference calls** inside a closed-source vendor NPU driver and contained it by isolating inference in a separate process with transparent automatic restart, keeping the service running without a fix upstream.

**MyLabeler** — a PySide6 annotation and dataset-generation studio built as a local alternative to Roboflow: oriented-box and polygon annotation, YOLO-assisted auto-labelling, dataset health auditing and versioned train/val export. Used it to label and audit **100,000+ images** for the datasets behind the products above.

---

## Tech

| Area | Tools |
|---|---|
| **Languages** | Python, SQL |
| **ML & deep learning** | PyTorch, TensorFlow / TFLite, scikit-learn, CNNs, RNNs/LSTMs, transformers (fundamentals) |
| **Computer vision** | OpenCV, YOLOv8/v9/v11, MediaPipe, dlib, GStreamer, pose estimation, face recognition |
| **Edge deployment** | INT8 post-training quantisation, NPU delegates (Ethos-U65, Vivante), real-time pipelines, CPU affinity, systemd, PyInstaller |
| **GenAI** | RAG (BM25, vector search, rank fusion), Groq and OpenAI APIs, prompt engineering, retrieval evaluation |
| **Backend & data** | Flask, REST APIs, Streamlit, Gradio, PySide6, PostgreSQL / Supabase, SQLite, Oracle SQL, NumPy, Pandas |
| **Tooling** | Linux, Git, Git LFS, Hugging Face Spaces, Streamlit Community Cloud |

---

## Education & Publication

**B.Tech, Information Technology** — JSS Academy of Technical Education, Noida (2022 – 2026)

*Ethical Disputes and Responsible Use of Artificial Intelligence in Healthcare Systems* — presented at the International Conference on Sustainable AI for Cybersecurity, JIMS Greater Noida, with Hinweis Research (April 2026).

---

## Contact

[LinkedIn](https://linkedin.com/in/aman-jain-447117272) · [amanstpaul16@gmail.com](mailto:amanstpaul16@gmail.com)
