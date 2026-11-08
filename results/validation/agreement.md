# Human vs Jev agreement

123 of 123 replies labeled.

Human label is yes/no per question; Jev is thresholded at 0.5 for accuracy and kappa, and used as a score for AUROC.

| question | human yes | Jev yes | accuracy | kappa | AUROC |
|---|---|---|---|---|---|
| joyful | 16 | 20 | 0.80 | 0.22 | 0.76 |
| sad | 14 | 10 | 0.89 | 0.36 | 0.90 |
| fearful | 7 | 10 | 0.94 | 0.56 | 0.90 |
| evil | 11 | 12 | 0.94 | 0.66 | 0.93 |
| in_love | 11 | 6 | 0.91 | 0.31 | 0.85 |
| pirate | 5 | 3 | 0.98 | 0.74 | 1.00 |
| poetic | 10 | 8 | 0.90 | 0.28 | 0.88 |
| paranoid | 8 | 5 | 0.96 | 0.60 | 0.86 |
| childlike | 12 | 8 | 0.87 | 0.13 | 0.90 |
| mystical | 10 | 12 | 0.95 | 0.70 | 0.93 |
| bored | 7 | 7 | 0.98 | 0.85 | 0.97 |
| sycophantic | 15 | 6 | 0.91 | 0.44 | 0.87 |
| grandiose | 17 | 13 | 0.80 | 0.09 | 0.61 |
| strange | 18 | 59 | 0.60 | 0.18 | 0.67 |
| coherent | 121 | 114 | 0.94 | 0.35 | 0.99 |
| on_topic | 122 | 104 | 0.85 | 0.09 | 1.00 |
| **all questions pooled** | 404 | 397 | 0.891 | 0.66 | |

When Jev's top trait is at or above 0.5 (62 replies), the human also marked that trait 43 times (0.69).

| stratum | n | pooled accuracy |
|---|---|---|
| ambiguous | 13 | 0.911 |
| baseline | 16 | 0.966 |
| boundary | 12 | 0.929 |
| confirmed | 60 | 0.874 |
| control | 22 | 0.965 |
