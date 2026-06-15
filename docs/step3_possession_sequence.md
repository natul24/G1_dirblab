# Step 3: Smoothed Possession Sequence

Step 3 follows the project instruction:

> Assign the ball to the nearest player, smooth it over time, and track when
> possession changes. This is the skeleton of most events.

The raw nearest-player estimate from Step 2 remains in the table. Step 3 adds
clearer columns for the smoothed possession assignment, possession changes, and
possession sequences.

## Input

```text
data/processed/model_base/master_join_table.parquet
```

The input is still one row per reliable live-play tracking frame across all
matches. It includes `match_id`, tracking/ball features, nearest-player features,
and joined event labels.

## Match Splits

Splits are stored in:

```text
config.yaml
```

The split is by complete `match_id`, never by row. This avoids leakage because
nearby 10 Hz frames from the same match are highly correlated. The table keeps a
single `data_split` column with values `train`, `validation`, `test`, or
`unassigned`.

Use `train` and `validation` to choose rule thresholds. Use `test` only for final
reported evaluation.

Current split:

| Split | Match IDs |
| --- | --- |
| `train` | `678949`, `679026`, `679053`, `679072`, `679075`, `679088`, `679104`, `679128`, `682607`, `682717`, `682815`, `683132`, `683190`, `683253`, `683261`, `683309`, `683425`, `684014`, `684119`, `684139`, `684141`, `684147`, `689340` |
| `validation` | `689526`, `689552`, `689556`, `713886`, `713893` |
| `test` | `713910`, `713946`, `713998`, `714000`, `745399` |

## Logic

1. Build `raw_possession_key` from Step 2's `possessing_team_id` and
   `possessing_player_id` when `has_possession=True`.
2. Fill short no-possession gaps when the same player has the ball before and
   after the gap.
3. Remove very short possession flips when they are surrounded by the same
   possessor.
4. Extract smoothed possession team/player IDs, names, and sequence IDs.
5. Mark possession changes:
   - `smoothed_possession_change`
   - `possession_team_change`
   - `possession_player_change`
   - `possession_change_type`

The word skeleton refers to how these possession changes support later event
rules. For example, a same-team player change can become a pass candidate, while
an opponent team change can become an interception or tackle candidate.

## Output

```text
data/processed/possession_sequence/possession_sequence_table.parquet
data/processed/possession_sequence/possession_sequence_summary.csv
data/processed/possession_sequence/possession_sequence_metadata.json
```

Run it with:

```bash
python main.py step3
```

For a single-match inspection run:

```bash
python main.py step3 --match-id 679026
```

The single-match run reads the all-match master join table, filters to that
match, and writes suffixed files such as
`possession_sequence_table_679026.parquet`. Rerunning Step 3 overwrites the
same output paths.
