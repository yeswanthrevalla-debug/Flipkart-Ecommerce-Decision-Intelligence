Domain: E-commerce

## How to run

Run these commands from the repository root after cloning the `main` branch and opening a terminal in the project folder:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The scripts read `data/records.csv` and write their result files into `outputs/`. `common.py` contains a side-effect-free stable-ordering helper imported by Parts 2 and 4. The data generator recreates the synthetic dataset supplied for this assignment. The implementation uses Python and NumPy; no external dataset or paid service is required.

## Part 1 findings

The sample mean order value is 8.9382 hundreds of INR, which is 0.9382 hundreds (INR 93.82) above the planning assumption of 8.0 (INR 800). The test statistic is z = 2.8930, whose absolute value exceeds the two-tailed 5% critical value 1.96, so H0 is rejected. A Type I error here would mean concluding that the population's average order value differs from INR 800 when in truth it is INR 800. The difference is statistically significant in this sample; whether INR 93.82 is commercially important depends on margins, category mix and operating costs.

The estimated sample size needed for a margin of error of 0.5 hundreds of INR is 1,293, compared with the 800 rows available. The current sample is therefore smaller than the calculated target for that precision.

The gradient-descent slope gap is 0.0000. With the specified learning rate and 500 full-batch updates, gradient descent converged to the same simple-regression line as the closed-form calculation.

The raw residual standard-deviation ratio is 3.7525, indicating a strong funnel: residual spread is much larger in the upper fitted-value half. After the log-log transformation the ratio is 1.0396, so the residual spread is much more even and the funnel is substantially reduced.

Using the required mechanical absolute-coefficient threshold of 0.05 in the multiple regression, the predictors above the threshold are ln(X4) (coefficient 1.1576), X2 complaint count (0.0940), X3 discount-order share (0.1067), and X6 COD flag (0.1518). The adjusted R² is 0.7403 for the multiple regression, compared with 0.6561 for the one-predictor log-log model. On this measure the additional predictors improve fit, although the coefficients are on different scales and the threshold is not a ranking of causal or practical importance.

## Part 2 findings

The 200-update training losses, from lowest to highest, are: learning rate 1.0 (0.3703), 0.1 (0.3776), and 0.01 (0.5091). The smaller learning rate is slower to make progress within 200 updates, while the larger rate reaches a lower loss here; a rate that is too large can also overshoot or become unstable, so this result is specific to the tested rates and data.

The three largest absolute logistic-regression weights are X3 discount-order share (0.9013), X1 customer tenure (-0.5554), and X2 complaint count (0.5505). The largest odds ratio is X3 at 2.4629: holding the other predictors fixed, a one-training-standard-deviation increase in discount-order share multiplies the modelled odds of churn by about 2.46.

The always-negative baseline has accuracy 0.7875, while logistic regression at τ = 0.5 has accuracy 0.8042. Because only 51 of the 240 test records are positive, accuracy can look high even when a model misses many churners. Lowering τ from 0.5 to 0.3 increases recall from 0.4706 to 0.7451, while precision decreases from 0.5455 to 0.4634.

At τ = 0.3, logistic regression has F1 = 0.5714 and recall = 0.7451. KNN (K = 5) has F1 = 0.3696 and recall = 0.3333, so logistic regression performs better on both reported measures.

I would deploy the logistic-regression model with the cost-selected threshold τ = 0.15 for a controlled retention pilot. Its test cost is INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 if nobody is flagged. The threshold was selected using training-row costs so that the test set remains an independent evaluation set rather than being used to tune the decision.

## Part 3 findings

The successive WCSS drops are 659.8282 (K = 2 to 3), 108.0739 (3 to 4), 171.0575 (4 to 5), and 45.5466 (5 to 6). The specified elbow rule selects K = 3 because the drop from 3 to 4 is less than half the drop from 2 to 3. The later drops do not shrink steadily; the pinned starting rows give a reproducible solution, not necessarily the global optimum for every K.

- **Segment A — Established, higher-value shoppers:** 322 records, mean tenure 47.5028 months, mean order value 3.5976 hundreds of INR and outcome rate 0.0186. A practical action is to use low-cost loyalty recognition and avoid unnecessary discounting.
- **Segment B — High-spend, service-friction shoppers:** 255 records, mean complaint count 6.0706, mean order value 17.1248 hundreds of INR and outcome rate 0.2353. A practical action is proactive complaint resolution and targeted service recovery.
- **Segment C — Discount-led, higher-risk shoppers:** 223 records, discount-order share 0.7118, loyalty score 38.5215 and outcome rate 0.4664. This is the highest-outcome-rate segment (46.64%); a practical action is to test targeted retention offers while checking whether the economics justify the discount.

At the single-linkage cut, the four clusters are {E1}, {E2, E3, E6}, {E4}, and {E5}. At the complete-linkage cut, the two clusters are {E1, E4, E5} and {E2, E3, E6}. Single linkage uses the closest pair between clusters, so one short bridge can join groups early; complete linkage uses the farthest pair, so it delays a merge until all cross-cluster pairs are relatively close. That difference produces different largest gaps and therefore different cut counts for this table.

The first principal component is driven most strongly in the positive direction by X1 tenure (loading 0.5317) and in the negative direction by X3 discount share (-0.5187), with X5 loyalty score also positive (0.4418) and X2 complaints negative (-0.4100). Three components are needed to retain at least 80% of the variance (cumulative 0.8616). The mean PC1 scores are 1.7773 for A, -0.7831 for B and -1.6709 for C, separating the established segment from the more discount-led and higher-risk groups along this component.

## Part 4 findings

S4 (X1 tenure ≤ 24 months) is the strongest of the four tested one-question splits: it has the largest information gain (0.1252) and Gini gain (0.0606), compared with the other candidate splits. A tree grown until every leaf is pure can memorize noise and individual training records, which harms generalisation. A maximum-depth limit is one rule that can stop a tree from growing too deep.

Among the five Task 4.2 models, logistic regression has the highest F1 (0.5714) with recall 0.7451; the stump has recall 0.7647 but a lower F1 of 0.5532. KNN has F1 0.3696 and recall 0.3333. Neither hard voting (F1 0.5344, recall 0.6863) nor soft voting (F1 0.5507, recall 0.7451) improves on logistic regression's F1, so combining these models did not improve the chosen F1 measure.

After round 1 of AdaBoost, the two points misclassified by h1 (points 3 and 7) each have normalised weight 0.2500, for a combined weight of 0.5000. Round 2 gives h2 a larger coefficient (α2 = 0.8047) than h1 (α1 = 0.5493) because h2's weighted error is lower (0.1667 versus 0.2500).

**Model A — image classifier architecture**

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv 1 (5 × 5, 6 filters, padding 2) | 28 × 28 × 6 | 156 |
| Max-pool 1 (2 × 2, stride 2) | 14 × 14 × 6 | 0 |
| Conv 2 (5 × 5, 16 filters, no padding) | 10 × 10 × 16 | 2,416 |
| Max-pool 2 (2 × 2, stride 2) | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Fully connected hidden layer | 32 | 12,832 |
| Fully connected output layer | 10 | 330 |

The fully connected 784 → 16 → 16 → 10 option has 13,002 parameters; the convolutional option has 15,734 parameters for the specified architecture. For the separate 128 × 128 × 3 comparison, the fully connected layer has 50,332,672 parameters while the 3 × 3 convolution with 64 filters has 1,792 parameters, illustrating the parameter-sharing advantage of convolution.

**Model B — review-text recurrent network**

The parameter shapes are W_xh = 128 × 64, W_hh = 128 × 128, W_hy = 16 × 128, b_h = 128 and b_y = 16; the total is 26,768 parameters. The parameter count does not grow with sequence length because the same recurrent weights are reused at every time step. For a 10-class image task, ReLU is a suitable hidden-layer activation and softmax with categorical cross-entropy is a suitable output/loss combination. A very deep plain network can suffer vanishing gradients, leaving early layers with tiny updates; skip connections provide a shorter path for information and gradients to flow through the network.

## Model-choice memo

**To: Head of the Retention Team**

I recommend piloting the logistic-regression model on the structured customer data, using the training-cost-selected threshold τ = 0.15. On the held-out test set, the cost at this threshold is INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 if no customers are flagged. At τ = 0.3, logistic regression achieves F1 = 0.5714 and recall = 0.7451; it also has the highest F1 among the five Task 4.2 models. KNN has F1 = 0.3696, while hard and soft voting do not improve on logistic regression's F1. The threshold was chosen on training rows, not the test rows, to avoid tuning the decision against the evaluation set. The evidence is limited by synthetic data and a single held-out split, so the cost estimate should be validated in a live, controlled pilot before broad deployment. Model A (the image classifier) should wait until the structured retention model is validated; it addresses a different merchandising use case rather than the immediate retention decision.

## Limitations

- The 800 records are synthetic and generated from a fixed process; they are not real Flipkart customer behaviour, so patterns and estimated costs may not transfer to production.
- Classification is evaluated on one stratified 70/30 split. A different split or time-based validation could produce different estimates.
- There are only 51 positive churn outcomes in the test set, so precision, recall, F1 and cost estimates are sensitive to a small number of records.
- The cost model uses fixed assumptions of INR 50 per false alarm and INR 400 per missed churner; actual coupon redemption, margin and customer response may differ.
- K-means uses the pinned starting rows for reproducibility; those starts are not guaranteed to find the global minimum WCSS.
- The neural-network sections are parameter-count exercises only. No image or text model was trained or evaluated.
