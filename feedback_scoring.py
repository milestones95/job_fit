"""
Feedback scoring — learns from kept/dismissed postings to improve ranking.

Two modes:
  - "centroid": nearest-neighbor penalty against dismissed embeddings. Works
    immediately with a handful of examples. Used until you have ~15+ labeled
    examples per class.
  - "classifier": logistic regression trained on kept vs. dismissed
    embeddings. More powerful once you have enough data, since it learns
    the actual decision boundary rather than relying on a hand-picked
    penalty weight.

Storage: a simple local JSON file mapping job identifiers to their
embedding vector + label. Swap this for a real DB later if needed.

Requires: pip install scikit-learn --break-system-packages (for classifier mode)
"""

import json
import os
import numpy as np

FEEDBACK_STORE_PATH = "job_feedback.json"

# Minimum labeled examples per class before we trust the classifier over
# the centroid heuristic. Below this, centroid mode is used regardless of
# what the caller asks for.
MIN_PER_CLASS_FOR_CLASSIFIER = 15

# Once we have at least this many per class, hold out a slice to report a
# quick accuracy estimate — purely informational, doesn't gate the switch.
MIN_PER_CLASS_FOR_VALIDATION = 10


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def load_feedback():
    if not os.path.exists(FEEDBACK_STORE_PATH):
        return {"kept": [], "dismissed": []}
    with open(FEEDBACK_STORE_PATH, "r") as f:
        return json.load(f)


def save_feedback(feedback):
    with open(FEEDBACK_STORE_PATH, "w") as f:
        json.dump(feedback, f)


def _unlabel(feedback, job_id):
    """Strip any existing label for job_id from both lists, so relabeling
    (e.g. changing your mind on a job) doesn't leave duplicate/conflicting
    entries."""
    feedback["kept"] = [k for k in feedback["kept"] if k["id"] != job_id]
    feedback["dismissed"] = [d for d in feedback["dismissed"] if d["id"] != job_id]


def record_dismissal(job_id, embedding, reason=None, reason_embedding=None, feedback=None):
    """Call this when the user dismisses a job from the dashboard.

    reason / reason_embedding are optional but preferred: a short sentence
    explaining WHY (e.g. "We don't want Product Manager roles") embeds much
    more precisely than the raw title, which falsely collides with anything
    sharing a word (e.g. "Agent") regardless of actual role type. See
    nearest_dismissed_similarity, which uses the reason embedding when
    present and only falls back to the title embedding otherwise."""
    feedback = feedback or load_feedback()
    _unlabel(feedback, job_id)
    entry = {"id": job_id, "embedding": embedding.tolist()}
    if reason:
        entry["reason"] = reason
        entry["reason_embedding"] = reason_embedding.tolist()
    feedback["dismissed"].append(entry)
    save_feedback(feedback)
    return feedback


def record_keep(job_id, embedding, feedback=None):
    """Call this when the user explicitly marks a job as relevant (keep
    button, saves it, applies). Kept examples make the classifier mode
    much stronger, since it needs both classes."""
    feedback = feedback or load_feedback()
    _unlabel(feedback, job_id)
    feedback["kept"].append({"id": job_id, "embedding": embedding.tolist()})
    save_feedback(feedback)
    return feedback


def reset_feedback():
    """Permanently clear all recorded kept/dismissed labels."""
    empty = {"kept": [], "dismissed": []}
    save_feedback(empty)
    return empty


# ---------------------------------------------------------------------------
# Mode 1: centroid / nearest-neighbor penalty
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def nearest_dismissed_similarity(embedding, feedback):
    """Compares against each dismissed job's reason embedding when one was
    given (much more precise — see record_dismissal), falling back to the
    raw title embedding for older dismissals that don't have a reason."""
    dismissed = feedback.get("dismissed", [])
    if not dismissed:
        return 0.0
    sims = [
        cosine_similarity(embedding, np.array(d.get("reason_embedding") or d["embedding"]))
        for d in dismissed
    ]
    return max(sims)


def score_with_centroid_penalty(embedding, ideal_role_similarity, feedback, penalty_weight=0.1):
    """
    ideal_role_similarity: the original score (similarity to your IDEAL_ROLE
    description), computed the same way as before.
    penalty_weight: how strongly to punish closeness to dismissed postings.
    Start around 0.3-0.5 and tune by eye against your own judgment.
    """
    nearest_dismissed = nearest_dismissed_similarity(embedding, feedback)
    return ideal_role_similarity - (penalty_weight * nearest_dismissed)


# ---------------------------------------------------------------------------
# Mode 2: trained classifier (use once you have ~30+ labeled examples total)
# ---------------------------------------------------------------------------

def train_classifier(feedback):
    from sklearn.linear_model import LogisticRegression

    kept = feedback.get("kept", [])
    dismissed = feedback.get("dismissed", [])

    if len(kept) < 5 or len(dismissed) < 5:
        raise ValueError(
            f"Not enough labeled data yet (kept={len(kept)}, dismissed={len(dismissed)}). "
            "Use centroid mode until you have at least ~5 of each, ideally 15+."
        )

    X = np.array([k["embedding"] for k in kept] + [d["embedding"] for d in dismissed])
    y = np.array([1] * len(kept) + [0] * len(dismissed))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def score_with_classifier(embedding, clf):
    """Returns probability (0-1) that this posting is a 'keep', per the
    trained classifier. Use this in place of ideal-role similarity once
    the classifier is trained."""
    prob = clf.predict_proba(embedding.reshape(1, -1))[0][1]
    return float(prob)


# ---------------------------------------------------------------------------
# Auto mode selection + validation
# ---------------------------------------------------------------------------

def choose_mode(feedback):
    """Decide which scoring mode to use based on how much labeled data exists."""
    n_kept = len(feedback.get("kept", []))
    n_dismissed = len(feedback.get("dismissed", []))
    if n_kept >= MIN_PER_CLASS_FOR_CLASSIFIER and n_dismissed >= MIN_PER_CLASS_FOR_CLASSIFIER:
        return "classifier"
    return "centroid"


def validate_classifier(feedback):
    """
    Quick, informational accuracy check via a train/test split. Only runs
    when there's enough data to spare a held-out slice without starving
    the training set. Returns None if there isn't enough data yet.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    kept = feedback.get("kept", [])
    dismissed = feedback.get("dismissed", [])
    if len(kept) < MIN_PER_CLASS_FOR_VALIDATION or len(dismissed) < MIN_PER_CLASS_FOR_VALIDATION:
        return None

    X = np.array([k["embedding"] for k in kept] + [d["embedding"] for d in dismissed])
    y = np.array([1] * len(kept) + [0] * len(dismissed))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return {"holdout_accuracy": accuracy, "n_train": len(X_train), "n_test": len(X_test)}


# ---------------------------------------------------------------------------
# Single entry point — the dashboard/script calls this and doesn't need to
# know or care which mode is active.
# ---------------------------------------------------------------------------

def score_jobs_auto(jobs_with_embeddings, ideal_vector, verbose=True, force_mode=None):
    """
    jobs_with_embeddings: list of dicts, each with an 'embedding' (np.array)
    key already computed, same as in job_fit_finder.py.

    force_mode: pass "centroid" or "classifier" to skip automatic mode
    selection entirely and always use that mode. Leave as None for the
    normal auto-switch-at-15-per-class behavior.

    Returns (ranked_jobs, mode_used, validation_info).
    Retrains from job_feedback.json on every call — cheap at this scale, so
    there's no need for incremental updates or model persistence.
    """
    feedback = load_feedback()
    mode = force_mode or choose_mode(feedback)
    validation_info = None
    clf = None

    if mode == "classifier":
        try:
            validation_info = validate_classifier(feedback)
            clf = train_classifier(feedback)
        except Exception as e:
            if verbose:
                print(f"[feedback_scoring] classifier training failed ({e}), falling back to centroid")
            mode = "centroid"

    if verbose:
        n_kept, n_dismissed = len(feedback.get("kept", [])), len(feedback.get("dismissed", []))
        msg = f"[feedback_scoring] mode={mode} (kept={n_kept}, dismissed={n_dismissed})"
        if validation_info:
            msg += f", holdout_accuracy={validation_info['holdout_accuracy']:.2f}"
        print(msg)

    scored = []
    for job in jobs_with_embeddings:
        embedding = job["embedding"]
        if mode == "classifier":
            score = score_with_classifier(embedding, clf)
        else:
            ideal_similarity = cosine_similarity(embedding, ideal_vector)
            score = score_with_centroid_penalty(embedding, ideal_similarity, feedback)
        scored.append({**job, "score": score})

    scored.sort(key=lambda j: j["score"], reverse=True)
    return scored, mode, validation_info
