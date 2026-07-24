# Third-Party Notices

## AGaLiTe

ArcMind's POBAX benchmark suite contains a modified JAX port of AGaLiTe.

- Project: AGaLiTe
- Repository: https://github.com/subho406/agalite
- Audited revision: `101acbecc121a258ad8f7e58e2f782f546674979`
- License: Apache License 2.0
- License copy: `licenses/AGALITE-APACHE-2.0.txt`

The port preserves the released finite-channel recurrence and GTrXL-style
policy block. ArcMind modifications include a batched recurrent policy-core
interface, explicit parameter dictionaries, an operationally frozen
LayerNorm epsilon, source and shared comparison lanes, parameter accounting,
and integration with the shared POBAX PPO learner.

The upstream repository does not contain a NOTICE file at the audited
revision. AGaLiTe and its authors are identified here for attribution. This
notice does not imply endorsement by the upstream authors.
