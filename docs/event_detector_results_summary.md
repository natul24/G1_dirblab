# Event Detector Results Summary

## Overview

This project tested both a binary pass detector and a multi-class event detector using tracking-data features. The goal was to detect football events from frame-level tracking data and convert those predictions into event-level detections using temporal post-processing.

## Key Findings

### 1. Binary pass detector improved after threshold tuning

The initial binary pass detector used a fixed probability threshold of 0.50. This was too conservative and missed most pass events. After introducing validation-based threshold tuning, the pass detector improved substantially, showing that the model was producing useful probabilities but needed a better decision threshold.

### 2. Extra features gave only marginal improvement

Additional tracking-based features were added, including ball movement, ball acceleration, closest-player distance, second-closest player context, longer-window ball speed, and related temporal indicators. These features only improved performance slightly. This suggests that the main limitation was not simply the absence of a specific feature.

### 3. Multi-class detector works end-to-end

A multi-class XGBoost event detector was implemented to detect multiple event types at once: PASS, BALL TOUCH, AERIAL, TACKLE, BALL RECOVERY, FOUL, and TAKEON. The system runs end-to-end and produces frame-level predictions, event-level detections, saved model artifacts, reports, and figures.

### 4. Minority classes struggle because of class imbalance and label noise

PASS and no event are the strongest classes because they have the most training examples. Smaller event types such as FOUL, TAKEON, AERIAL, and BALL RECOVERY have far fewer examples, making them harder for the model to learn. These classes also appear more sensitive to noisy or ambiguous labels, especially because some football actions can look similar in tracking data.

### 5. Stricter NMS reduces false positives but recall drops quickly

Temporal Non-Maximum Suppression was used to convert frame-level predictions into event-level detections. A stricter per-class NMS search reduced false positives for several minority event classes, especially TACKLE, FOUL, and TAKEON. However, recall dropped quickly when thresholds became stricter, so F1 only improved slightly for some classes and decreased for others.

Overall, NMS helps clean the output, but the main performance ceiling is likely related to class imbalance, label quality, and the difficulty of separating similar football actions from tracking data alone.

## Interpretation

The experiments suggest that the pipeline is technically working, but further improvement would likely require deeper changes to the labelling logic or training setup rather than simply adding more features. Possible next steps include event-specific labelling windows, improved class balancing, event-specific post-processing, and deeper error analysis of false positives and false negatives.
