## Seed / Nonce Guidance (PP-Mark v2.1)

- **Production**: use a 256-bit random nonce (e.g., 64 hex chars) for the semantic anchor seed. This makes brute-force infeasible (CWE-339 mitigation).
- **Demo/Test**: shorter seeds (e.g., 32-bit ints) work functionally but are weak and should not be used for security claims.
- Seed is concatenated into the semantic anchor `h = H(prompt, seed, model_id, timestamp, parent_manifest_hash?)` and then fed to Poseidon with the secret key, so increasing seed entropy does not change runtime cost meaningfully.
