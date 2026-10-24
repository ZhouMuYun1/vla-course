# Representative videos

Formal B-class evaluation: Base vs Final (LoRA r8 step300), 2 tasks × 10 episodes each.

| File | Model | Task | Seed | Outcome |
|---|---|---|---|---|
| videos/base_t1_seed1119_success.mp4 | Base | T1 spoon_on_towel | 1119 | success |
| videos/base_t2_seed2219_fail.mp4   | Base | T2 eggplant_in_basket | 2219 | fail |
| videos/final_t1_seed1111_fail.mp4  | Final | T1 spoon_on_towel | 1111 | fail |
| videos/final_t2_seed2219_success.mp4 | Final | T2 eggplant_in_basket | 2219 | success |

Failed episodes are included per spec (9.1 item 5: "在存在失败时包含失败样例").
Full per-episode records: episodes.csv
