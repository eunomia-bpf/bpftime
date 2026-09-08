# Warp eligibility fixture checkpoint

These draft fixtures and their CMake registration are preserved as a source
checkpoint at the user's request. They have **not been compiled or run**;
do not cite them as passing verifier or eligibility evidence. In particular,
the atomic-helper-emulation section currently has no assertion, and the
synthetic instruction streams still require implementation follow-up.

No runtime source is changed by this checkpoint. The existing runtime
libraries used by the queued Table1 measurement were built from `c4c83cd`;
this test-only commit does not replace or rebuild those libraries. No GPU
experiment or additional validation campaign was started here.
