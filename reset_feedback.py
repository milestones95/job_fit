"""
Reset feedback — permanently clears all Keep/Dismiss labels and rebuilds
the dashboard from scratch (back to centroid mode, scored only against
IDEAL_ROLE).

Usage:
  python reset_feedback.py            # asks for confirmation
  python reset_feedback.py --yes      # skips the confirmation prompt
"""

import sys

import feedback_scoring as fs
import build_dashboard


def main():
    skip_confirm = "--yes" in sys.argv

    feedback = fs.load_feedback()
    n_kept, n_dismissed = len(feedback["kept"]), len(feedback["dismissed"])

    if n_kept == 0 and n_dismissed == 0:
        print("No feedback recorded — nothing to reset.")
        return

    print(f"This will permanently delete {n_kept} kept and {n_dismissed} dismissed label(s).")
    if not skip_confirm:
        answer = input("Type 'yes' to permanently delete these labels: ").strip().lower()
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    fs.reset_feedback()
    print("Feedback cleared. Rebuilding dashboard in centroid mode...\n")
    build_dashboard.main()


if __name__ == "__main__":
    main()
