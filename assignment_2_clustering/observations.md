# Observations — Assignment 2: Customer Segmentation

Dataset: Credit Card Customer dataset (`CC_GENERAL.csv`), 8,950 customers, 17 numeric behavioral features available.
Features selected for clustering (8, chosen for interpretability over using all 17): `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `PURCHASES_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `TENURE`.

## Data preparation

- Dropped `CUST_ID` (identifier, irrelevant to distance calculations).
- Imputed missing values with the **median**: `CREDIT_LIMIT` (1 missing) and `MINIMUM_PAYMENTS` (313 missing, not used in final feature set but cleaned anyway). Median chosen over mean because these financial columns are right-skewed (a small number of very high spenders would pull the mean upward).
- No duplicate rows found.
- All 8 selected features scaled with `StandardScaler`.

### Why scaling matters here

Unscaled, `PURCHASES` ranges roughly 0–49,000 while `PURCHASES_FREQUENCY` ranges 0–1. In a raw Euclidean distance calculation, differences in `PURCHASES` alone would dominate completely and `PURCHASES_FREQUENCY` would have almost no effect on which cluster a customer is assigned to — even though both describe genuinely important, independent behavior. Scaling puts every feature on the same footing so the clustering reflects the customer's overall behavioral profile, not just whichever feature happens to have the largest raw numbers.

## KMeans

### Choosing K

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 56,231 | **0.418** |
| 3 | 48,371 | 0.228 |
| 4 | 41,650 | 0.261 |
| 5 | 35,821 | 0.271 |
| 6 | 32,192 | 0.264 |
| 7 | 29,147 | 0.271 |
| 8 | 27,066 | 0.256 |
| 9 | 25,289 | 0.253 |
| 10 | 24,060 | 0.253 |

Inertia decreases smoothly with no sharp elbow — typical for real-world behavioral data with overlapping groups rather than perfectly separated blobs. Silhouette score is highest at K=2 (0.418), but two clusters would be too coarse to be useful for a segmentation task (effectively just "low spenders" vs "high spenders," with no nuance on cash-advance or repayment behavior). **K=4 was chosen** as the practical balance: silhouette is still reasonable (0.261, well above the K=3 dip), and four segments is a size that's genuinely actionable for a marketing or credit-risk team without being so many that each segment loses meaning.

### KMeans cluster profile (K=4)

| Cluster | Balance | Purchases | Cash Advance | Credit Limit | Payments | Purchase Freq | Cash Adv Freq | Tenure | Size | % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 971 | 1,622 | 165 | 4,767 | 1,655 | 0.87 | 0.03 | 11.9 | 3,556 | 39.7% |
| 1 | 5,533 | 1,992 | 4,594 | 9,525 | 5,364 | 0.42 | 0.42 | 11.7 | 1,048 | 11.7% |
| 2 | 837 | 433 | 1,037 | 2,468 | 615 | 0.43 | 0.19 | 7.4 | 719 | 8.0% |
| 3 | 1,144 | 224 | 720 | 3,175 | 982 | 0.15 | 0.14 | 11.9 | 3,627 | 40.5% |

**What distinguishes each cluster:**
- **Cluster 0 — "Active regular spenders" (39.7%)**: High purchase frequency (0.87), moderate balance, very low cash-advance usage. These customers use their card for regular purchases and rarely borrow cash.
- **Cluster 1 — "High-value revolvers" (11.7%)**: By far the highest balance, cash advance, credit limit, and payments. Smaller group but clearly the highest-value / highest-risk segment — heavy cash-advance users with high credit limits.
- **Cluster 2 — "Newer, low-engagement customers" (8.0%)**: Notably lower tenure (7.4 vs ~12 for others), lower balance and purchases — likely newer cardholders who haven't built up much activity yet.
- **Cluster 3 — "Low-engagement long-tenured customers" (40.5%)**: Longest tenure but the lowest purchase frequency (0.15) and lowest purchases overall — customers who've held the card a long time but barely use it.

## DBSCAN

### Sensitivity to eps / min_samples

| eps | min_samples | Clusters | Noise % | Silhouette |
|---|---|---|---|---|
| 0.5 | 5 | 35 | 31.8% | -0.19 |
| 0.5 | 20 | 7 | 46.5% | 0.01 |
| 1.0 | 5 | 7 | 9.0% | 0.26 |
| 1.0 | 10+ | 1 | 11-15% | N/A |
| **1.5** | **5** | **4** | **2.9%** | **0.36** |
| 1.5 | 10+ | 1 | 4-5% | N/A |
| 2.0 | any | 1 | 1-2% | N/A |
| 2.5 | 5 | 2 | 0.7% | 0.64 |

DBSCAN is **highly sensitive** to both parameters. Small `eps` (0.5) fragments the data into 35 tiny clusters with heavy noise (32%) — the density radius is too tight to connect genuinely related points. As `eps` increases, clusters merge; by `eps=2.0`, nearly everything collapses into one giant cluster regardless of `min_samples`. Increasing `min_samples` at a fixed `eps` consistently increases noise, since it raises the bar for what counts as a "dense" region.

**Final configuration: eps=1.5, min_samples=5** — chosen because it's the smallest `eps` that produces multiple genuine clusters (4) with low noise (2.9%) and the best silhouette score among multi-cluster configurations (0.36).

### DBSCAN cluster profile

| Cluster | Balance | Purchases | Cash Advance | Credit Limit | Payments | Size | % |
|---|---|---|---|---|---|---|---|
| -1 (noise) | 5,625 | 5,193 | 6,212 | 11,891 | 11,469 | 261 | 2.9% |
| 0 | 1,441 | 869 | 820 | 4,263 | 1,431 | 8,677 | **97.0%** |
| 1 | 1,739 | 13,534 | 0 | 15,900 | 14,761 | 5 | 0.06% |
| 2 | 4,077 | 1,885 | 4,146 | 5,600 | 790 | 4 | 0.04% |
| 3 | 2,677 | 2,219 | 3,768 | 10,333 | 8,412 | 3 | 0.03% |

DBSCAN found one dominant dense cluster (97% of customers — the "typical" behavioral profile) and three extremely small clusters (3-5 customers each) representing tight pockets of unusual-but-consistent behavior (e.g., cluster 1 = very high purchases with zero cash advance). The **noise points (261, 2.9%) are the most interesting group** — they have the highest average balance, purchases, cash advance, and payments of any group, higher even than the small extreme clusters. These are customers whose behavior is unusually intense but not consistent enough with any nearby group to form a dense region — in practice, exactly the kind of high-value, high-activity outliers a bank might want to review individually rather than lump into a generic segment.

## KMeans vs DBSCAN — comparison

- KMeans forces every point into one of K roughly-shaped clusters, producing 4 reasonably balanced segments (8-40% each) — useful for building a clean segmentation scheme for marketing.
- DBSCAN instead reveals that **most customers (97%) actually belong to one broad, dense behavioral group**, with a small number of distinct outlier pockets and a meaningful noise set. This is a genuinely different, and arguably more honest, picture: it suggests the "natural" structure in this data isn't four evenly-sized segments, but one common profile plus various flavors of unusual, high-activity customers.
- **KMeans is more useful for building an actionable segmentation** (e.g., four marketing personas). **DBSCAN is more useful for outlier/anomaly detection** — flagging the 2.9% of customers whose behavior doesn't fit the norm, which KMeans cannot do (it would have force-assigned them to the nearest of the 4 clusters regardless of fit).

## PCA

### Explained variance

| Component | Variance Ratio | Cumulative |
|---|---|---|
| PC1 | 33.81% | 33.81% |
| PC2 | 24.24% | 58.05% |
| PC3 | 12.52% | 70.58% |
| PC4 | 9.37% | 79.94% |

The first two components explain **58.05%** of total variance. This means the 2D PCA plots used for visualization capture a majority — but far from all — of the structure in the original 8-dimensional feature space. Roughly 42% of the variance (spread across 6 more components) is not visible in the 2D scatter plots. Two points that appear close together in the PCA plot could still differ meaningfully on dimensions not captured by PC1/PC2 (e.g., `TENURE` or `CASH_ADVANCE_FREQUENCY`), so the plots should be read as a useful summary, not a complete picture of similarity.

## Required observations — answered

- **How did scaling change the clustering result?** Without scaling, `PURCHASES` and `CASH_ADVANCE` (both in the thousands) would dominate distance calculations, effectively drowning out `PURCHASES_FREQUENCY` and `CASH_ADVANCE_FREQUENCY` (both 0-1). Scaling ensures every behavioral dimension contributes proportionally.
- **How was the final K selected?** By balancing silhouette score against practical usefulness — K=2 had the highest silhouette (0.418) but was too coarse to be actionable; K=4 kept a reasonable silhouette (0.261) while producing four genuinely distinct, interpretable segments.
- **What characteristics distinguish each KMeans cluster?** See cluster profile table above — differentiated mainly by cash-advance reliance, purchase frequency, and tenure.
- **How sensitive was DBSCAN to eps and min_samples?** Very sensitive — cluster count ranged from 35 (eps=0.5) down to 1 (eps≥2.0), and noise percentage ranged from under 1% to over 46% depending on the combination.
- **Which points were considered noise and why might that be useful?** The 261 points (2.9%) with unusually high balance, purchases, cash advance, and payments simultaneously. These are useful to flag separately because they represent high-value, high-activity customers whose behavior doesn't fit a common pattern — worth individual review rather than folding into a generic segment.
- **How much variance did the first two PCA components explain?** 58.05%.
- **Which clustering algorithm was more useful for this dataset and why?** KMeans for building an actionable, balanced segmentation scheme; DBSCAN for surfacing outlier customers that a forced-partition method like KMeans would otherwise hide inside an ill-fitting cluster.
