# Making improvements to limbtrack

Lass maintains the main version and the main features. Other contributors add and improve the program.
Lass has authorised this work without requiring his approval for each change. Reviews are welcome, but are optional.

## Keep each change easy to understand

Start a separate branch from the current `main` for each clear problem or improvement. Open a pull request (a proposed change on
GitHub) with the supplied template. Use a draft while the work is still being checked.

Write pull request descriptions, commit messages, changelog entries and review comments in plain English. Explain what the operator
will see, why it helps, and whether any measured number can change. Say what is known and what has not been checked. A screenshot or
a small before-and-after example often explains more than code.

Keep unrelated changes separate. Add a short entry under `Unreleased` in `CHANGELOG.md`. Only tick a check in the template when it
was actually done. If a measured number can change, name the number and the situations affected; report a measured difference when
one is available, otherwise say it is not known yet.

## Check the proposed version

The `Automatic tests` check runs on Windows with Python 3.12 for every pull request to `main`, including drafts. It also runs after
a change reaches `main`. It checks `protocol.json` and runs the existing tests. It does not change the tests or regenerate their
saved tracking reference.

Open the check's Details link to see the result. The run summary lists passed, failed and skipped tests. Under Artifacts, download
`windows-python312-test-results` for the test report and the installed package versions. A failed check must be investigated; a
timing-sensitive failure must be reported rather than hidden by removing the test or treating failure as success.

These checks use generated pictures and simulated signals. Passing them does not validate the real camera, load cell, sound,
landmark placement or agreement with a goniometer. State separately which of these were tried.

For a manual check, use a separate folder and follow README's setup instructions there. Start with the included demo video.
Keep test settings and test recordings separate from a participant session. Note the branch or commit tested.

## Protect the recordings and measurement rules

Never commit participant video, `recordings/`, `audit/`, PDFs or `settings.json`: this repository is public.
Do not upload these as test reports either.

Preserve raw voltages and dot positions. Display and analysis smoothing must stay separate from the saved raw samples.
Do not fill in missing dot positions. Keep settings fixed during a recording and record them with the trial.
Only regenerate `tests/golden_tracking.npz` for an intentional tracking change, and explain the reason and effect.
Changes to angle definitions or the protocol must be clearly described as changes to what is measured.

## Optional review and repository settings

`.github/CODEOWNERS` lists Lass as the contact for the program. GitHub can request his review when a pull request is ready, but that
file alone does not make his approval compulsory.

This proposal does not change repository settings. An administrator can protect `main` by requiring pull requests and the
`Windows / Python 3.12` test check after it has run once. Do not enable required approvals or required code-owner reviews for this
workflow: Lass's separate approval is not required.

After checking the result and explaining any limits, an authorised contributor may merge the pull request. GitHub Actions will not
merge changes automatically. A draft must be marked ready before it can be merged. If a proposed change is dropped, close its pull
request; the change will not enter `main`.
